import os
import numpy as np
import pandas as pd
import joblib
import urllib.parse
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from .loadmodel import get_model
from .production_status import get_production_times
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def time_features(df):
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['day'] = df['date_time'].dt.day
    df['month'] = df['date_time'].dt.month
    df['year'] = df['date_time'].dt.year
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

def stat_features(df):
    df['total_reject'] = df['total_reject'].fillna(0)
    df['ma_3'] = df['total_reject'].rolling(window=3, min_periods=1).mean()
    df['lag_1'] = df['total_reject'].shift(1).fillna(0)
    df['lag_2'] = df['total_reject'].shift(2).fillna(0)
    return df

# Konfigurasi koneksi database
def get_database_engine():
    db_config = {
        "host": os.getenv("DB_HOST"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
        "port": int(os.getenv("DB_PORT") or 3306),
    }

    user = urllib.parse.quote_plus(db_config["user"])
    password = urllib.parse.quote_plus(db_config["password"])
    engine_url = f"mysql+mysqlconnector://{user}:{password}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    return create_engine(engine_url)


def fetch_data(engine, query):
    return pd.read_sql(query, engine)

def get_target_value(target_df):
    if target_df.empty:
        return None, None
    row = target_df.iloc[-1]
    try:
        value = int(str(row['target']).replace(",", ""))
        time = row['tanggal'].strftime('%Y-%m-%dT%H:%M:%S')
        return value, time
    except Exception as e:
        logger.warning("Failed to parse target: %s", e)
        return None, None

def get_total_hours(row):
    try:
        prod_start = pd.to_datetime(row['prod_start'])
        prod_end = pd.to_datetime(row['prod_end'])
        return (prod_end - prod_start).total_seconds() / 3600
    except:
        return None

def predict_db():
    try:
        engine = get_database_engine()
        model = get_model()
        x_scaler = joblib.load('model/x_scaler.pkl')
        y_scaler = joblib.load('model/y_scaler.pkl')

        query = """
            SELECT 
                DATE_FORMAT(tanggal, '%Y-%m-%d %H:00:00Z') AS date_time,
                MAX(int_pressure_measurement) AS pressure_measurement,
                MAX(displacement_measurement) AS displacement_measurement,
                MAX(int_pressure_upper) AS pressure_upper,
                MAX(int_pressure_lower) AS pressure_lower,
                MAX(displacement_upper) AS displacement_upper,
                MAX(displacement_lower) AS displacement_lower,
                MAX(total) AS total_reject,
                MAX(total_can_tested) AS total_can,
                CASE WHEN (lotno) = 'NOT ACTIVE' THEN 'NOT PRODUCTION' ELSE 'PRODUCTION' END AS status
            FROM tr_process_cpd
            WHERE tanggal >= NOW() - INTERVAL 12 HOUR AND total > 0 
            GROUP BY DATE_FORMAT(tanggal, '%Y-%m-%d %H:00:00')
            ORDER BY date_time ASC;
        """
        df = fetch_data(engine, query)
        if df.empty:
            return {"error": "Data kosong. Tidak ditemukan record terbaru dari database."}

        df = time_features(df)
        df = stat_features(df)

        features = [
            "pressure_measurement", "displacement_measurement",
            "pressure_lower", "pressure_upper", "displacement_upper",
            "year", "month", "day_of_week", "day",
            "hour", "hour_sin", "hour_cos",
            "lag_1", "lag_2", "ma_3"
        ]

        if df[features].isnull().any().any():
            logger.error("Missing values detected:\n%s", df[features].isnull().sum())
            return {"error": "Nilai NaN terdeteksi di feature input."}

        target_df = fetch_data(engine, """
            SELECT 
            tgl AS tanggal, 
            target, 
            prod_start, 
            prod_end
            FROM mst_prodidentity 
            WHERE tgl >= NOW() AND isActive = 'ACTIVE'
        """)
        last_target, last_target_time = get_target_value(target_df)
        
        print("[DEBUG] Target value:", last_target)
        print("[DEBUG] Target time:", last_target_time)

        actual_series, prediction_series = [], []
        for i in range(len(df)):
            row = df.iloc[[i]]
            input_scaled = x_scaler.transform(row[features].values).reshape((1, 1, len(features)))
            pred_scaled = model.predict(input_scaled)
            pred_original = y_scaler.inverse_transform(pred_scaled)[0][0]
            pred_time = row.iloc[0]['date_time'] + timedelta(hours=1)
            actual_val = int(row.iloc[0]['total_reject'])
            prediction_val = int(round(pred_original))
            actual_series.append({"x": row.iloc[0]['date_time'].strftime("%Y-%m-%dT%H:%M:%S"), "y": actual_val})
            prediction_series.append({"x": pred_time.strftime("%Y-%m-%dT%H:%M:%S"), "y": prediction_val})
    
        prev_prediction = prediction_series[-2] if len(prediction_series) > 1 else None
        error_mae = abs(actual_series[-1]['y'] - prev_prediction['y']) if prev_prediction else None

        last_pred_time = pd.to_datetime(prediction_series[-1]['x'])
        # hour_target = next((get_total_hours(row, last_pred_time) for _, row in target_df.iterrows() if get_total_hours(row, last_pred_time)), None)
        
        hour_target = None
        if not target_df.empty:
            row = target_df.iloc[-1]
            total_hours = get_total_hours(row)
            if total_hours and total_hours > 0:
                hour_target = float(row['target']) / total_hours

        latest = df.iloc[[-1]]
        reject = int(latest.iloc[0]['total_reject'])
        total_can = int(latest.iloc[0].get('total_can', 1000))
        rejection = round((reject / total_can) * 100, 2) if total_can > 0 else 0

        input_latest_scaled = x_scaler.transform(latest[features].values).reshape((1, 1, len(features)))
        pred_latest = int(round(y_scaler.inverse_transform(model.predict(input_latest_scaled))[0][0]))
        
        percentage_error = None
        if prev_prediction and 'y' in prev_prediction:
            pred_val = prev_prediction['y']
            if pred_val > 0 or reject > 0:
                percentage_error = round(abs(pred_val - reject) / max(pred_val, reject) * 100, 2)

        if hour_target and hour_target > 0:
            standard_hourly_target = round(prev_prediction['y'] / hour_target * 100, 2)  
        else:
            standard_hourly_target = None

        production_times = get_production_times(engine)

        return {
            "series": {"actual": actual_series, "prediction": prediction_series},
            "latest": {
                "actual": reject,
                "actual_datetime": latest.iloc[0]['date_time'].strftime("%Y-%m-%d %H:%M"),
                "prediction": pred_latest,
                "prediction_datetime": (latest.iloc[0]['date_time'] + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
                "pressure_measurement": float(latest.iloc[0]['pressure_measurement']),
                "displacement_measurement": float(latest.iloc[0]['displacement_measurement']),
                "pressure_lower": float(latest.iloc[0]['pressure_lower']),
                "pressure_upper": float(latest.iloc[0]['pressure_upper']),
                "displacement_lower": float(latest.iloc[0]['displacement_lower']),
                "displacement_upper": float(latest.iloc[0]['displacement_upper']),
                "total_can_tested": "{:,}".format(int(latest.iloc[0]['total_can'])).replace(",", "."),
                "rejection": rejection,
                "status": latest.iloc[0]['status']
            },
            "target_and_standard": {
                "target_production_hour": last_target_time,
                "target_value": last_target,
                "target_hour": round(hour_target) if hour_target is not None else None,
                "standard_hour": prediction_series[-1]['x'],
                "standard_percent": rejection,
                "percentage_error": percentage_error,
                "standard_hourly_target": standard_hourly_target,
                "standard_percent_date": last_target_time[:10] if last_target_time else None,
                "standard_over_limit": rejection > 0.10
            },
            "prev_prediction": prev_prediction,
            "error_mae": error_mae,
            "production_start": pd.to_datetime(production_times.get('production_start')).strftime('%Y-%m-%d %H:%M:%S') if production_times.get('production_start') else None,
            "production_stop": pd.to_datetime(production_times.get('production_stop')).strftime('%Y-%m-%d %H:%M:%S') if production_times.get('production_stop') else None
        }

    except Exception as e:
        logger.exception("Unhandled server error:")
        return {"error": f"Unhandled server error: {str(e)}"}
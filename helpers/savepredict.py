from sqlalchemy import create_engine, text
import os
import urllib.parse


def _get_dev_engine():
    """Utility untuk membuat koneksi ke database dev."""
    dev_db_config = {
        "host": os.getenv("DB_DEV_HOST"),
        "user": os.getenv("DB_DEV_USER"),
        "password": os.getenv("DB_DEV_PASSWORD"),
        "database": os.getenv("DB_DEV_NAME"),
        "port": int(os.getenv("DB_DEV_PORT") or 3306),
    }
    user_enc = urllib.parse.quote_plus(dev_db_config["user"])
    pwd_enc = urllib.parse.quote_plus(dev_db_config["password"])
    host = dev_db_config["host"]
    port = dev_db_config["port"]
    db = dev_db_config["database"]

    return create_engine(f"mysql+mysqlconnector://{user_enc}:{pwd_enc}@{host}:{port}/{db}")


def save_prediction_to_dev_db(datetime_actual, reject_actual, datetime_prediction, reject_prediction, standard_percent):
    """Menyimpan SATU data prediction ke database dev."""
    engine = _get_dev_engine()
    insert_query = text("""
        INSERT INTO prediction (datetime_actual, reject_actual, datetime_prediction, reject_prediction, standard_percent)
        VALUES (:datetime_actual, :reject_actual, :datetime_prediction, :reject_prediction, :standard_percent)
    """)
    try:
        with engine.begin() as conn:
            conn.execute(
                insert_query,
                {
                    "datetime_actual": datetime_actual,
                    "reject_actual": reject_actual,
                    "datetime_prediction": datetime_prediction,
                    "reject_prediction": reject_prediction,
                    "standard_percent": standard_percent,
                },
            )
        print("[INFO] Berhasil simpan 1 data prediction.")
    except Exception as e:
        print("[ERROR] Gagal menyimpan 1 data prediction:", e)
        raise


def save_bulk_predictions_to_dev_db(data_list):
    """Menyimpan BANYAK data prediction (batch) ke database dev."""
    engine = _get_dev_engine()
    insert_query = text("""
        INSERT INTO prediction (datetime_actual, reject_actual, datetime_prediction, reject_prediction, standard_percent)
        VALUES (:datetime_actual, :reject_actual, :datetime_prediction, :reject_prediction, :standard_percent)
    """)
    try:
        with engine.begin() as conn:
            conn.execute(insert_query, data_list)  
        print(f"[INFO] Berhasil simpan {len(data_list)} data prediction.")
    except Exception as e:
        print("[ERROR] Gagal simpan batch data prediction:", e)
        raise
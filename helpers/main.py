import os
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

from helpers.predict import predict_db
from helpers.schemas import FullPredictionResponse
from helpers.savepredict import save_prediction_to_dev_db
from datetime import datetime, timedelta
from helpers.savepredict import save_prediction_to_dev_db, save_bulk_predictions_to_dev_db

load_dotenv()

app = FastAPI(
    title="LSTM Prediction API",
    description="Get latest actual reject value & prediction for next hour."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def auto_save_predictions():
    result = predict_db()
    if isinstance(result, dict) and "error" in result:
        print(f"[AUTO SAVE] Gagal dapatkan prediction dari predict_db(): {result['error']}")
        return

    actual_series = result.get("series", {}).get("actual", [])
    prediction_series = result.get("series", {}).get("prediction", [])

    if not actual_series or not prediction_series:
        print("[AUTO SAVE] Tidak ada data 'actual' atau 'prediction' untuk disimpan.")
        return

    prediction_lookup = {
        item["x"]: item["y"] for item in prediction_series
    }

    data_for_db = []
    for actual_item in actual_series:
        actual_dt = actual_item["x"]
        reject_actual = actual_item["y"]

        actual_dt_obj = datetime.fromisoformat(actual_dt)
        prediction_dt = (actual_dt_obj + timedelta(hours=1)).isoformat()

        reject_prediction = prediction_lookup.get(prediction_dt)
        if reject_prediction is None:
            continue  

        data_for_db.append({
            "datetime_actual": actual_dt.replace("T", " "),
            "reject_actual": reject_actual,
            "datetime_prediction": prediction_dt.replace("T", " "),
            "reject_prediction": reject_prediction,
            "standard_percent": result.get("target_and_standard", {}).get("standard_percent", 0.0),
        })

    if not data_for_db:
        print("[AUTO SAVE] Tidak ada pasangan data actual -> prediction yang dapat disimpan.")
        return

    try:
        save_bulk_predictions_to_dev_db(data_for_db)
        print(f"[AUTO SAVE] Berhasil simpan {len(data_for_db)} data prediction ke database.")
    except Exception as e:
        print(f"[AUTO SAVE] Gagal simpan batch data prediction: {e}")

@app.get("/predict", response_model=FullPredictionResponse, tags=["Prediction"])
async def predict():
    result = predict_db()
    if isinstance(result, dict) and "error" in result:
        err_msg = result["error"].lower()
        if "data kosong" in err_msg or "tidak ditemukan" in err_msg:
            raise HTTPException(status_code=404, detail="No Data. Not Production.")
        raise HTTPException(status_code=500, detail=result["error"])

    if not isinstance(result, dict) or 'series' not in result or 'latest' not in result:
        raise HTTPException(status_code=500, detail="Prediction result missing or invalid.")

    return result

@app.post("/savepredict", tags=["Prediction"])
async def save_prediction():
    result = predict_db()
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    try:
        latest = result["latest"]
        save_prediction_to_dev_db(
            datetime_actual=latest["actual_datetime"],
            reject_actual=latest["actual"],
            datetime_prediction=latest["prediction_datetime"],
            reject_prediction=latest["prediction"],
            standard_percent=latest["rejection"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan prediksi: {e}")

    return {"message": "Hasil prediksi berhasil disimpan ke DB development."}

@app.on_event("startup")
def startup_event():
    """Menyalakan scheduler ketika app mulai."""
    print("[STARTUP] Menyalakan background scheduler...")
    scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    """Mematikan scheduler ketika app mati."""
    print("[SHUTDOWN] Mematikan background scheduler...")
    scheduler.shutdown()

from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(auto_save_predictions, "interval", hours=1)
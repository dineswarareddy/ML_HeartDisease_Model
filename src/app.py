import os
import time
import json
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# Configure structured application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("heart_disease_api")

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# 1. Prometheus Metrics Instrumentation (Exposes /metrics endpoint)
Instrumentator().instrument(app).expose(app)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
model = None


@app.on_event("startup")
def load_model_artifact():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model successfully loaded from {MODEL_PATH}")
    else:
        logger.error(f"Model file not found at {MODEL_PATH}")


# 2. Logging Middleware: Captures latencies, status codes, and request sizes
@app.middleware("http")
def monitor_requests(request: Request, call_next):
    start_time = time.time()

    # Process request
    response: Response = call_next(request)

    latency = time.time() - start_time
    logger.info(
        f"Path: {request.url.path} | Method: {request.method} | "
        f"Status: {response.status_code} | Latency: {latency:.4f}s"
    )
    return response


class PatientFeatures(BaseModel):
    age: float = Field(..., example=58.0)
    sex: float = Field(..., example=1.0)
    cp: float = Field(..., example=2.0)
    trestbps: float = Field(..., example=140.0)
    chol: float = Field(..., example=211.0)
    fbs: float = Field(..., example=1.0)
    restecg: float = Field(..., example=0.0)
    thalach: float = Field(..., example=165.0)
    exang: float = Field(..., example=0.0)
    oldpeak: float = Field(..., example=0.0)
    slope: float = Field(..., example=2.0)
    ca: float = Field(..., example=0.0)
    thal: float = Field(..., example=2.0)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(payload: PatientFeatures):
    if model is None:
        logger.error("Inference attempted while model was unavailable")
        raise HTTPException(status_code=503, detail="Model artifact is not loaded.")

    try:
        input_data = pd.DataFrame([payload.model_dump()])
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(probabilities[prediction])

        # Log incoming feature payload and resulting classification
        logger.info(
            f"Prediction payload: {payload.model_dump()} -> "
            f"Prediction: {prediction}, Confidence: {confidence:.4f}"
        )

        return {
            "prediction": prediction,
            "prediction_label": "Heart Disease Present" if prediction == 1 else "Absence of Disease",
            "confidence": round(confidence, 4)
        }
    except Exception as exc:
        logger.error(f"Inference error: {str(exc)}")
        raise HTTPException(status_code=400, detail=str(exc))
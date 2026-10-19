import os
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api")

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

# Instrument Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model artifact from {MODEL_PATH}")
    else:
        logger.error(f"Model file not found at {MODEL_PATH}")

class PatientData(BaseModel):
    age: float = Field(..., example=55.0)
    sex: float = Field(..., example=1.0)
    cp: float = Field(..., example=0.0)
    trestbps: float = Field(..., example=130.0)
    chol: float = Field(..., example=250.0)
    fbs: float = Field(..., example=0.0)
    restecg: float = Field(..., example=1.0)
    thalach: float = Field(..., example=145.0)
    exang: float = Field(..., example=0.0)
    oldpeak: float = Field(..., example=1.2)
    slope: float = Field(..., example=1.0)
    ca: float = Field(..., example=0.0)
    thal: float = Field(..., example=2.0)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(payload: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        input_dict = payload.model_dump()
        input_df = pd.DataFrame([input_dict])

        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        confidence = float(probabilities[prediction])

        logger.info(f"Input: {input_dict} | Output: {prediction} | Confidence: {confidence:.4f}")

        return {
            "prediction": prediction,
            "prediction_label": "Disease Present" if prediction == 1 else "Absence of Disease",
            "confidence": confidence
        }
    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
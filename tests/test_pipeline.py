import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from fastapi.testclient import TestClient
from src.preprocess import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.train import create_preprocessor
from src.app import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "age": 63.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0,
    "chol": 233.0, "fbs": 1.0, "restecg": 0.0, "thalach": 150.0,
    "exang": 0.0, "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0
}

def test_preprocessor_transformation():
    preprocessor = create_preprocessor()
    sample_df = pd.DataFrame([SAMPLE_PAYLOAD])
    transformed = preprocessor.fit_transform(sample_df)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 0

def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_api_predict_validation():
    # Sending empty body should trigger a 422 Unprocessable Entity
    response = client.post("/predict", json={})
    assert response.status_code == 422
import os
import joblib
import pandas as pd
import pytest

@pytest.fixture
def dummy_patient():
    return pd.DataFrame([{
        "age": 55.0, "sex": 1.0, "cp": 0.0, "trestbps": 130.0,
        "chol": 245.0, "fbs": 0.0, "restecg": 1.0, "thalach": 150.0,
        "exang": 0.0, "oldpeak": 1.2, "slope": 1.0, "ca": 0.0, "thal": 2.0
    }])

def test_model_artifact_loading():
    model_path = "models/best_model.joblib"
    assert os.path.exists(model_path), "Model artifact does not exist. Train the model first."
    model = joblib.load(model_path)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

def test_prediction_output_range(dummy_patient):
    model = joblib.load("models/best_model.joblib")
    pred = model.predict(dummy_patient)
    proba = model.predict_proba(dummy_patient)

    assert pred[0] in [0, 1]
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0][0] <= 1.0
    assert 0.0 <= proba[0][1] <= 1.0
    assert pytest.approx(proba[0][0] + proba[0][1], 0.001) == 1.0
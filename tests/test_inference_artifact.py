import joblib
import pandas as pd

def test_exported_pipeline_inference():
    # 1. Load persisted pipeline
    model_path = "models/best_model.joblib"
    pipeline = joblib.load(model_path)

    # Confirm both preprocessing and classification stages exist
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    # 2. Create raw patient input without manual scaling or one-hot encoding
    raw_sample = pd.DataFrame([{
        "age": 55.0,
        "sex": 1.0,
        "cp": 2.0,
        "trestbps": 130.0,
        "chol": 240.0,
        "fbs": 0.0,
        "restecg": 1.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 1.2,
        "slope": 1.0,
        "ca": 0.0,
        "thal": 2.0
    }])

    # 3. Perform prediction directly on raw inputs
    prediction = pipeline.predict(raw_sample)
    probabilities = pipeline.predict_proba(raw_sample)

    assert prediction[0] in [0, 1]
    assert len(probabilities[0]) == 2
    assert 0.0 <= probabilities[0][1] <= 1.0
    print("Pipeline artifact validation successful! Predictions generated on raw inputs.")

if __name__ == "__main__":
    test_exported_pipeline_inference()
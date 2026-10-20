import joblib
import pandas as pd

# 1. Load the serialized end-to-end pipeline
model_path = "models/best_model.joblib"
pipeline = joblib.load(model_path)

print(f"Loaded pipeline steps: {list(pipeline.named_steps.keys())}")
# Expected output: ['preprocessor', 'classifier']

# 2. Raw sample input (matching UCI features directly, without scaling or encoding)
raw_patient = pd.DataFrame([{
    "age": 58.0,
    "sex": 1.0,
    "cp": 2.0,
    "trestbps": 140.0,
    "chol": 211.0,
    "fbs": 1.0,
    "restecg": 0.0,
    "thalach": 165.0,
    "exang": 0.0,
    "oldpeak": 0.0,
    "slope": 2.0,
    "ca": 0.0,
    "thal": 2.0
}])

# 3. Predict directly on raw features
prediction = pipeline.predict(raw_patient)
probabilities = pipeline.predict_proba(raw_patient)

print(f"Raw Input Prediction: {prediction[0]}")
print(f"Prediction Probabilities [P(0), P(1)]: {probabilities[0]}")
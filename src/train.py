import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from preprocess import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL

def create_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

def evaluate_and_log(name, model_pipeline, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=name):
        # Cross-validation
        cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring="roc_auc")
        mlflow.log_metric("cv_roc_auc_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_roc_auc_std", float(cv_scores.std()))

        # Train full model
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)
        y_proba = model_pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_roc_auc": auc
        })

        # Log scikit-learn model artifact
        mlflow.sklearn.log_model(model_pipeline, artifact_path=name)
        print(f"[{name}] AUC: {auc:.4f} | Accuracy: {acc:.4f}")
        return auc

def run_training():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train, y_train = train_df.drop(columns=[TARGET_COL]), train_df[TARGET_COL]
    X_test, y_test = test_df.drop(columns=[TARGET_COL]), test_df[TARGET_COL]

    preprocessor = create_preprocessor()
    mlflow.set_experiment("Heart_Disease_Classification")

    # Model 1: Logistic Regression
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_auc = evaluate_and_log("Logistic_Regression", lr_pipeline, X_train, y_train, X_test, y_test)

    # Model 2: Random Forest
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])
    rf_auc = evaluate_and_log("Random_Forest", rf_pipeline, X_train, y_train, X_test, y_test)

    # Export best model pipeline locally for Docker/Serving
    best_pipeline = rf_pipeline if rf_auc >= lr_auc else lr_pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, "models/best_model.joblib")
    print(f"Exported best pipeline to models/best_model.joblib")

if __name__ == "__main__":
    run_training()
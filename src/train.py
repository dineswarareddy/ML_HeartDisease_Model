import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Ensure destination directory for artifact plots exists
ARTIFACTS_DIR = "screenshots/mlflow_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


def build_preprocessor():
    """Builds numerical and categorical feature transformation pipeline."""
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    """Calculates 5-fold CV ROC-AUC and standard test set metrics."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
    mean_cv_auc = cv_scores.mean()

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'cv_mean_roc_auc': mean_cv_auc
    }
    return pipeline, metrics, y_pred, y_proba


def train_and_log_model(model_name, model_estimator, preprocessor, X_train, y_train, X_test, y_test):
    """Trains pipeline, logs plots/artifacts, metrics, and models to MLflow."""
    print(f"\n--- Running Experiment: {model_name} ---")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_estimator)
    ])

    trained_pipeline, metrics, y_pred, y_proba = evaluate_model(
        pipeline, X_train, y_train, X_test, y_test
    )

    print(f"5-Fold CV Mean ROC-AUC: {metrics['cv_mean_roc_auc']:.4f}")
    print(
        f"Test Scores -> Accuracy: {metrics['accuracy']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"ROC-AUC: {metrics['roc_auc']:.4f}"
    )

    with mlflow.start_run(run_name=model_name):
        # 1. Log metrics
        for key, val in metrics.items():
            mlflow.log_metric(key, val)

        # 2. Log parameters
        mlflow.log_param("model_type", model_name)
        if hasattr(model_estimator, "get_params"):
            for param, val in model_estimator.get_params().items():
                mlflow.log_param(f"classifier__{param}", str(val))

        # 3. Generate and save Confusion Matrix plot
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm, cmap="Blues")
        ax_cm.set_title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        cm_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_confusion_matrix.png")
        fig_cm.savefig(cm_path, dpi=300)
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")

        # 4. Generate and save ROC Curve plot
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax_roc, name=model_name)
        ax_roc.set_title(f"ROC Curve - {model_name}")
        plt.tight_layout()
        roc_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_roc_curve.png")
        fig_roc.savefig(roc_path, dpi=300)
        plt.close(fig_roc)
        mlflow.log_artifact(roc_path, artifact_path="evaluation_plots")

        # 5. Log the scikit-learn pipeline model
        sample_input = X_train.iloc[:5]
        sample_output = trained_pipeline.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)

        mlflow.sklearn.log_model(
            sk_model=trained_pipeline,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=sample_input
        )

    return metrics['roc_auc'], trained_pipeline


def execute_pipeline():
    """Loads processed data, trains candidate models, registers top model."""
    experiment_name = "Heart_Disease_Prediction_Experiment"
    mlflow.set_experiment(experiment_name)

    data_path = "data/processed/heart_cleaned.csv"
    if not os.path.exists(data_path):
        data_path = "data/raw/heart_raw.csv"

    df = pd.read_csv(data_path)

    target_col = 'target' if 'target' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    # Model 1: Logistic Regression
    lr_estimator = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear')
    lr_auc, trained_lr = train_and_log_model(
        "Logistic_Regression", lr_estimator, preprocessor, X_train, y_train, X_test, y_test
    )

    # Model 2: Random Forest
    rf_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_auc, trained_rf = train_and_log_model(
        "Random_Forest", rf_estimator, preprocessor, X_train, y_train, X_test, y_test
    )

    # Select and export the best model
    if lr_auc >= rf_auc:
        best_name = "Logistic_Regression"
        best_auc = lr_auc
        best_model = trained_lr
    else:
        best_name = "Random_Forest"
        best_auc = rf_auc
        best_model = trained_rf

    export_path = "models/best_model.joblib"
    joblib.dump(best_model, export_path)

    print("\n================ Selection Summary ================")
    print(f"Top-Performing Model: {best_name} (ROC-AUC: {best_auc:.4f})")
    print(f"Serialized production pipeline exported to: {export_path}")
    print("===================================================")


if __name__ == "__main__":
    execute_pipeline()
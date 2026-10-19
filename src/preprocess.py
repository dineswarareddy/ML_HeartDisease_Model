import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COL = "target"

def fetch_and_clean_data(output_dir="data"):
    os.makedirs(f"{output_dir}/raw", exist_ok=True)
    os.makedirs(f"{output_dir}/processed", exist_ok=True)

    # 1. Fetch dataset from UCI
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    # Save raw snapshot
    raw_df = pd.concat([X, y], axis=1)
    raw_df.to_csv(f"{output_dir}/raw/heart_raw.csv", index=False)

    # 2. Binary target conversion: UCI target is 0 (healthy) or 1..4 (disease)
    # Binary classification: 0 = Absence, 1 = Presence
    target_series = y.iloc[:, 0].apply(lambda val: 1 if val > 0 else 0)
    df = pd.concat([X, target_series.rename(TARGET_COL)], axis=1)

    # 3. Clean and handle missing values
    # Convert potential string representations of nulls to NaN
    df = df.replace("?", pd.NA)

    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Train / Test Split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET_COL]
    )

    train_df.to_csv(f"{output_dir}/processed/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/processed/test.csv", index=False)
    print(f"Data processed successfully. Train: {len(train_df)}, Test: {len(test_df)}")

if __name__ == "__main__":
    fetch_and_clean_data()
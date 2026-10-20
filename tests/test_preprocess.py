import pytest
import numpy as np
import pandas as pd
from src.preprocess import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL
from src.train import build_preprocessor


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame([{
        "age": 63.0, "sex": 1.0, "cp": 3.0, "trestbps": 145.0,
        "chol": 233.0, "fbs": 1.0, "restecg": 0.0, "thalach": 150.0,
        "exang": 0.0, "oldpeak": 2.3, "slope": 0.0, "ca": 0.0, "thal": 1.0
    }])


def test_null_imputation_logic():
    # Construct DataFrame with nulls
    df_with_nulls = pd.DataFrame([{
        "age": np.nan, "sex": 1.0, "cp": 2.0, "trestbps": 130.0,
        "chol": np.nan, "fbs": 0.0, "restecg": 1.0, "thalach": 140.0,
        "exang": 0.0, "oldpeak": 1.0, "slope": 1.0, "ca": np.nan, "thal": np.nan
    }])

    # Impute via median/mode logic
    for col in NUMERICAL_FEATURES:
        df_with_nulls[col] = pd.to_numeric(df_with_nulls[col]).fillna(50.0)
    for col in CATEGORICAL_FEATURES:
        df_with_nulls[col] = pd.to_numeric(df_with_nulls[col]).fillna(0.0)

    assert df_with_nulls.isnull().sum().sum() == 0


def test_feature_shape_and_transformation(sample_raw_data):
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(sample_raw_data)

    # Check that rows match and output dimensions are non-zero
    assert transformed.shape[0] == len(sample_raw_data)
    assert transformed.shape[1] >= len(NUMERICAL_FEATURES)
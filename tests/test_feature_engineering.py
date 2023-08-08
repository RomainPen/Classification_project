import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer  
from src.data.features_engineering.features_engineering import FeaturesEngineering

# Create a fixture to generate a sample DataFrame for testing
@pytest.fixture
def sample_dataframe():
    data = {
        "RECENCY_OF_LAST_RECHARGE": [15, 40, 70, 100, 50],
        "BALANCE_M1": [100, 200, 150, 300, 250],
        "BALANCE_M2": [80, 180, 130, 280, 230],
        "BALANCE_M3": [70, 170, 120, 270, 220],
        "INC_DURATION_MINS_M1": [30, 20, 15, 0, 10],
        "INC_PROP_SMS_CALLS_M1": [10, 5, 8, 0, 6],
        "INC_DURATION_MINS_M2": [25, 15, 10, 0, 5],
        "INC_PROP_SMS_CALLS_M2": [8, 4, 6, 0, 3],
        "INC_DURATION_MINS_M3": [20, 10, 5, 0, 7],
        "INC_PROP_SMS_CALLS_M3": [5, 3, 4, 0, 2],
        "OUT_DURATION_MINS_M1": [40, 30, 25, 0, 20],
        "OUT_SMS_NO_M1": [15, 10, 12, 0, 8],
        "OUT_INT_DURATION_MINS_M1": [10, 5, 8, 0, 6],
        "OUT_888_DURATION_MINS_M1": [5, 3, 4, 0, 2],
        "OUT_VMACC_NO_CALLS_M1": [2, 1, 2, 0, 1],
        "OUT_DURATION_MINS_M2": [30, 20, 15, 0, 18],
        "OUT_SMS_NO_M2": [10, 8, 8, 0, 5],
        "OUT_INT_DURATION_MINS_M2": [8, 4, 6, 0, 4],
        "OUT_888_DURATION_MINS_M2": [3, 2, 2, 0, 1],
        "OUT_VMACC_NO_CALLS_M2": [1, 1, 1, 0, 1],
        "OUT_DURATION_MINS_M3": [20, 15, 10, 0, 12],
        "OUT_SMS_NO_M3": [8, 5, 6, 0, 4],
        "OUT_INT_DURATION_MINS_M3": [5, 3, 4, 0, 3],
        "OUT_888_DURATION_MINS_M3": [2, 1, 1, 0, 1],
        "OUT_VMACC_NO_CALLS_M3": [1, 1, 1, 0, 1],
        "CONTRACT_TENURE_DAYS": [600, 800, 100, 400, 1500]
    }
    df = pd.DataFrame(data)
    return df

# Test features_creation method
def test_features_creation(sample_dataframe):
    fe = FeaturesEngineering()
    df = fe.features_creation(sample_dataframe)
    assert "FLAG_RECHARGE_M1" in df.columns
    assert "FLAG_RECHARGE_M2" in df.columns
    assert "FLAG_RECHARGE_M3" in df.columns
    assert "FLAG_RECHARGE_PLUS_M3" in df.columns
    assert "AVERAGE_MULTIPLE_RECHARGE_M1_M2_M3" in df.columns
    assert "FLAG_IN_M1" in df.columns
    assert "FLAG_IN_M2" in df.columns
    assert "FLAG_IN_M3" in df.columns
    assert "FLAG_OUT_M1" in df.columns
    assert "FLAG_OUT_M2" in df.columns
    assert "FLAG_OUT_M3" in df.columns
    assert "OLD_CONTRACT" in df.columns

# Test filter_selection method
def test_filter_selection(sample_dataframe):
    fe = FeaturesEngineering()
    x_df = sample_dataframe.drop("TARGET", axis=1)
    y_df = sample_dataframe["TARGET"]
    target = "TARGET"
    x_filtered = fe.filter_selection(x_df, y_df, target, train=True)
    assert "FLAG_RECHARGE_M1" not in x_filtered.columns
    assert "INC_DURATION_MINS_M1" not in x_filtered.columns
    assert "OUT_SMS_NO_M1" not in x_filtered.columns
    assert "BALANCE_M1" in x_filtered.columns
    assert "INC_DURATION_MINS_M2" not in x_filtered.columns
    assert "OUT_SMS_NO_M2" not in x_filtered.columns
    assert "BALANCE_M2" in x_filtered.columns

# Test encoding_scaling method
def test_encoding_scaling(sample_dataframe):
    fe = FeaturesEngineering()
    x_df = sample_dataframe.drop("TARGET", axis=1)
    target = sample_dataframe["TARGET"]
    categorical_var_OHE = ["INC_DURATION_MINS_M1", "INC_DURATION_MINS_M2", "OUT_SMS_NO_M3"]
    categorical_var_OrdinalEncoding = {}
    categorical_var_TE = ["BALANCE_M1", "BALANCE_M2"]
    continious_var = ["INC_DURATION_MINS_M1", "INC_DURATION_MINS_M2"]
    encoding_type_cont = MinMaxScaler()
    encoded_scaled_df = fe.encoding_scaling(x_df, categorical_var_OHE, categorical_var_OrdinalEncoding, categorical_var_TE, target, continious_var, encoding_type_cont, train=True)
    assert "INC_DURATION_MINS_M1" not in encoded_scaled_df.columns
    assert "INC_DURATION_MINS_M2" not in encoded_scaled_df.columns
    assert "BALANCE_M1" not in encoded_scaled_df.columns
    assert "BALANCE_M2" not in encoded_scaled_df.columns



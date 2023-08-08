import pandas as pd
import numpy as np
import pytest
from src.data.data_cleaning.data_cleaning import Data_Cleaning

# Create a fixture to generate a sample DataFrame for testing
@pytest.fixture
def sample_dataframe() -> pd.DataFrame :
    data = {
        "CUSTOMER_AGE": [15, 25, 40, 60, -5],
        "CUSTOMER_GENDER": ["male", "female", "not ent", "unknown", "not specified"],
        "CONTRACT_TENURE_DAYS": [365, 730, 1825, 1095, -365],
        "NO_OF_RECHARGES_6M": [10, 15, -5, 20, 8],
        "FAILED_RECHARGE_6M": [1, 2, 3, 0, 1],
        "INC_OUT_PROP_DUR_MIN_M1": [100, -50, 75, 200, 30],
        "INC_OUT_PROP_DUR_MIN_M2": [50, 60, -10, 80, 20],
        "INC_OUT_PROP_DUR_MIN_M3": [80, 90, -15, 120, 25],
        "CURR_HANDSET_MODE": ["model1", "model2", "model3", "model4", "model5"]
    }
    df = pd.DataFrame(data)
    return df

# Test basic_treatment method
def test_basic_treatment(sample_dataframe):
    cleaner = Data_Cleaning()
    useless_columns = ["CONTRACT_TENURE_DAYS", "FAILED_RECHARGE_6M"]
    cleaned_df = cleaner.basic_treatment(sample_dataframe, useless_columns)
    
    assert len(cleaned_df) == 4
    assert "CONTRACT_TENURE_DAYS" not in cleaned_df.columns
    assert "FAILED_RECHARGE_6M" not in cleaned_df.columns
    assert cleaned_df["CUSTOMER_GENDER"].equals(pd.Series(["male", "female", "not ent", "unknown"], name="CUSTOMER_GENDER"))

# Test treat_anormal_variables method
def test_treat_anormal_variables(sample_dataframe):
    cleaner = Data_Cleaning()
    cleaned_df = cleaner.treat_anormal_variables(sample_dataframe)
    
    assert np.isnan(cleaned_df.loc[0, "CUSTOMER_AGE"])
    assert cleaned_df.loc[1, "CUSTOMER_GENDER"] == "female"
    assert cleaned_df.loc[2, "CUSTOMER_GENDER"] is None
    assert np.isnan(cleaned_df.loc[4, "age_first_contract"])
    assert np.isnan(cleaned_df.loc[2, "CONTRACT_TENURE_DAYS"])
    assert cleaned_df.loc[2, "NO_OF_RECHARGES_6M"] == 5
    assert np.all(cleaned_df[["INC_OUT_PROP_DUR_MIN_M1", "INC_OUT_PROP_DUR_MIN_M2", "INC_OUT_PROP_DUR_MIN_M3"]] >= 0)

# Test features_transformation method
def test_features_transformation(sample_dataframe):
    cleaner = Data_Cleaning()
    external_df = pd.DataFrame({
        "model": ["model1", "model2", "model3"],
        "marque_tel": ["brand1", "brand2", "brand3"]
    })
    transformed_df = cleaner.features_transformation(sample_dataframe, external_df)
    
    assert transformed_df["marque"].equals(pd.Series(["brand1", "brand2", "brand3", None, None], name="marque"))
    assert "CURR_HANDSET_MODE" not in transformed_df.columns


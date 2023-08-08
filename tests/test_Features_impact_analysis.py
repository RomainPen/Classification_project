import pandas as pd
import numpy as np
import pytest
from src.visualization.Features_impact_analysis import FeaturesImpactAnalysis
from xgboost import XGBClassifier
from sklearn.datasets import load_iris

# Create a fixture to generate a sample DataFrame for testing
@pytest.fixture
def sample_dataframe():
    data = {
        "Feature1": [1, 2, 3, 4, 5],
        "Feature2": [5, 4, 3, 2, 1],
        "Feature3": [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    return df

# Create a fixture for a sample XGBoost model
@pytest.fixture
def sample_model():
    X, y = load_iris(return_X_y=True)
    model = XGBClassifier()
    model.fit(X, y)
    return model

# Test Features_importance method
def test_features_importance(sample_model):
    x_df = pd.DataFrame(np.random.rand(10, 5), columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5'])
    fia = FeaturesImpactAnalysis(sample_model, x_df)
    importance_df = fia.Features_importance()
    assert "Features" in importance_df.columns
    assert "Features importance (in %)" in importance_df.columns

# Test SHAP_explainer method
def test_SHAP_explainer(sample_model, sample_dataframe):
    x_df = sample_dataframe.drop("TARGET", axis=1)
    fia = FeaturesImpactAnalysis(sample_model, x_df)
    file_saving = "test_dir/shap_summary.pdf" 
    fia.SHAP_explainer(file_saving)
    # Add assertions here to verify if the file was created correctly
    # For example, check if the file exists and its size



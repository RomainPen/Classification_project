import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytest
from src.visualization.EDA import EDA

# Create a fixture to generate a sample DataFrame for testing
@pytest.fixture
def sample_dataframe():
    data = {
        "CUSTOMER_AGE": [25, 30, 40, 60, 50],
        "CUSTOMER_GENDER": ["male", "female", "male", "female", "male"],
        "TARGET": [0, 1, 1, 0, 1],
        "NUMERIC_FEATURE1": [10, 15, 20, 25, 30],
        "NUMERIC_FEATURE2": [5, 8, 12, 18, 10],
        "OTHER_FEATURE": ["A", "B", "C", "D", "E"]
    }
    df = pd.DataFrame(data)
    return df

# Mocking the sns.catplot function
def mock_catplot(*args, **kwargs):
    pass
sns.catplot = mock_catplot

# Mocking the sns.histplot function
def mock_histplot(*args, **kwargs):
    pass
sns.histplot = mock_histplot

# Mocking the sns.heatmap function
def mock_heatmap(*args, **kwargs):
    pass
sns.heatmap = mock_heatmap

# Test distrib_for_cat_by_target method
def test_distrib_for_cat_by_target(sample_dataframe):
    eda = EDA(sample_dataframe, "TARGET")
    with pytest.raises(SystemExit):
        eda.distrib_for_cat_by_target("CUSTOMER_GENDER", sample_dataframe, "TARGET", "test_dir/")

# Test target_feature_distribution_groupby_categorical_features method
def test_target_feature_distribution_groupby_categorical_features(sample_dataframe):
    eda = EDA(sample_dataframe, "TARGET")
    with pytest.raises(SystemExit):
        eda.target_feature_distribution_groupby_categorical_features("test_dir/")

# Test distrib_for_num_by_target method
def test_distrib_for_num_by_target(sample_dataframe):
    eda = EDA(sample_dataframe, "TARGET")
    with pytest.raises(SystemExit):
        eda.distrib_for_num_by_target("NUMERIC_FEATURE1", sample_dataframe, "TARGET", "test_dir/")

# Test target_feature_distribution_groupby_numerical_features method
def test_target_feature_distribution_groupby_numerical_features(sample_dataframe):
    eda = EDA(sample_dataframe, "TARGET")
    with pytest.raises(SystemExit):
        eda.target_feature_distribution_groupby_numerical_features("test_dir/")

# Test correlation_matrix method
def test_correlation_matrix(sample_dataframe):
    eda = EDA(sample_dataframe, "TARGET")
    with pytest.raises(SystemExit):
        eda.correlation_matrix("test_dir/correlation_matrix.png")

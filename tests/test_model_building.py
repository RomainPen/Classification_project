import pytest
import numpy as np
from src.models.model_building.model_building import ModelBuilding
from sklearn.datasets import load_iris

# Create a fixture to generate a sample dataset for testing
@pytest.fixture
def sample_dataset():
    X, y = load_iris(return_X_y=True)
    return X, y

# Test train method
def test_train(sample_dataset):
    x_train, y_train = sample_dataset
    random_state = 42
    hyper_parameters = {
        "n_estimators": {"low": 50, "high": 200},
        "max_depth": {"low": 3, "high": 10},
        "min_child_weight": {"low": 1, "high": 10},
        "learning_rate": {"low": 0.01, "high": 0.1},
        "min_split_loss": {"low": 0, "high": 1},
        "colsample_bytree": {"low": 0.5, "high": 1},
        "subsample": {"low": 0.5, "high": 1}
    }
    scoring = "accuracy"
    k_fold = 5
    n_trials = 10
    
    mb = ModelBuilding(random_state)
    best_model = mb.train(x_train, y_train, hyper_parameters, scoring, k_fold, n_trials)
    
    # Add assertions here to check if the best_model is created correctly and the training process is successful

# Add more test cases if needed


import pytest
import numpy as np
from src.models.model_evaluation.model_evaluation import ModelEvaluation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Create a fixture to generate a sample dataset for testing
@pytest.fixture
def sample_dataset():
    X, y = load_iris(return_X_y=True)
    return X, y

# Test ACCURACY_score method
def test_accuracy_score(sample_dataset):
    x_data, y_data = sample_dataset
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42)
    model.fit(x_train, y_train)
    
    random_state = 42
    model_evaluator = ModelEvaluation(model, random_state)
    cross_val = False
    k_fold = 5
    
    accuracy = model_evaluator.ACCURACY_score(x_test, y_test, cross_val, k_fold)
    
    assert isinstance(accuracy, float)
    assert accuracy >= 0.0 and accuracy <= 1.0

# Test ROC_AUC_score method
def test_roc_auc_score(sample_dataset):
    x_data, y_data = sample_dataset
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42)
    model.fit(x_train, y_train)
    
    random_state = 42
    model_evaluator = ModelEvaluation(model, random_state)
    cross_val = False
    k_fold = 5
    
    roc_auc = model_evaluator.ROC_AUC_score(x_test, y_test, cross_val, k_fold)
    
    assert isinstance(roc_auc, float)
    assert roc_auc >= 0.0 and roc_auc <= 1.0

# Test F1_score method
def test_f1_score(sample_dataset):
    x_data, y_data = sample_dataset
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42)
    model.fit(x_train, y_train)
    
    random_state = 42
    model_evaluator = ModelEvaluation(model, random_state)
    cross_val = False
    k_fold = 5
    
    f1 = model_evaluator.F1_score(x_test, y_test, cross_val, k_fold)
    
    assert isinstance(f1, float)
    assert f1 >= 0.0 and f1 <= 1.0


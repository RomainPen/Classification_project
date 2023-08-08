import pytest
import numpy as np
from src.visualization.model_result_analysis  import ModelResultAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Create a fixture to generate a sample dataset for testing
@pytest.fixture
def sample_dataset():
    X, y = load_iris(return_X_y=True)
    return X, y

# Test CONFUSION_MATRIX method
def test_confusion_matrix(sample_dataset):
    x_data, y_data = sample_dataset
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42)
    model.fit(x_train, y_train)
    
    model_analyzer = ModelResultAnalysis(model, x_test, y_test)
    file_location = "tests_dir/confusion_matrix.png"
    model_analyzer.CONFUSION_MATRIX(file_location)
    
    # Add assertions to verify the generated file or visualize the plot if needed

# Test CLASSIFICATION_REPORT method
def test_classification_report(sample_dataset):
    x_data, y_data = sample_dataset
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42)
    model.fit(x_train, y_train)
    
    model_analyzer = ModelResultAnalysis(model, x_test, y_test)
    classification_report_str = model_analyzer.CLASSIFICATION_REPORT()
    
    assert isinstance(classification_report_str, str)

# Test ROC_AUC_curve method
def test_roc_auc_curve(sample_dataset):
    x_data, y_data = sample_dataset
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42)
    model.fit(x_train, y_train)
    
    model_analyzer = ModelResultAnalysis(model, x_test, y_test)
    file_location = "tests_dir/roc_auc_curve.png"
    model_analyzer.ROC_AUC_curve(file_location)
    
    # Add assertions to verify the generated file or visualize the plot if needed

# Test BEST_THRESHOLD method
def test_best_threshold(sample_dataset):
    x_data, y_data = sample_dataset
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42)
    model.fit(x_train, y_train)
    
    model_analyzer = ModelResultAnalysis(model, x_test, y_test)
    best_threshold_info = model_analyzer.BEST_THRESHOLD()
    
    assert isinstance(best_threshold_info, dict)
    
# Test LIFT_CURVE method
def test_lift_curve(sample_dataset):
    x_data, y_data = sample_dataset
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42)
    model.fit(x_train, y_train)
    
    model_analyzer = ModelResultAnalysis(model, x_test, y_test)
    file_location = "tests_dir/lift_curve.png"
    model_analyzer.LIFT_CURVE(file_location)
    
    # Add assertions to verify the generated file or visualize the plot if needed

# Test LEARNING_CURVE method
def test_learning_curve(sample_dataset):
    x_data, y_data = sample_dataset
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42)
    model.fit(x_train, y_train)
    
    model_analyzer = ModelResultAnalysis(model, x_test, y_test)
    cv = 5
    scoring = "accuracy"
    file_location = "tests_dir/learning_curve.png"
    model_analyzer.LEARNING_CURVE(cv, scoring, file_location)
    
    # Add assertions to verify the generated file or visualize the plot if needed



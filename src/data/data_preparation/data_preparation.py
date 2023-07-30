import pandas as pd
from src.data.feature_engineering import FeatureEngineering

class DataPreparation:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        return pd.read_csv(self.file_path)


    def preprocess_data(self, raw_data):
        # Data preprocessing steps...
        # Your existing data preprocessing logic
        # ...

        # Create an instance of the FeatureEngineering class
        feature_engineering = FeatureEngineering()

        # Apply feature engineering to create new features
        preprocessed_data = feature_engineering.engineer_features(raw_data)
        return preprocessed_data

    def save_processed_data(self, data, save_path):
        data.to_csv(save_path, index=False)
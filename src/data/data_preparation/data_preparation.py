import pandas as pd
from src.data.feature_engineering import FeatureEngineering
from src.data.data_cleaning.data_cleaning import Data_Cleaning

class DataPreparation :
    def __init__(self, file_path, external_data_path):
        self.file_path = file_path
        self.external_data_path = external_data_path

    def load_data(self):
        return pd.read_csv(self.file_path, index_col=False, sep=";")


    def preprocess_data(self, raw_data):
        # Data preprocessing steps...
        # Your existing data preprocessing logic
        # ...
        
        #drop 1st column :
        preprocessed_data = raw_data.iloc[:, 1:]
        
        #data cleaning :
        
        

        # Create an instance of the FeatureEngineering class
        feature_engineering = FeatureEngineering()

        # Apply feature engineering to create new features
        preprocessed_data = feature_engineering.engineer_features(raw_data)
        return preprocessed_data

    def save_processed_data(self, data, save_path):
        data.to_csv(save_path, index=False)
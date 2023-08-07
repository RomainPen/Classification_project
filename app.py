import streamlit as st 
import yaml
import os
import joblib
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer  
from src.data.data_cleaning.data_cleaning import Data_Cleaning
from src.data.features_engineering.features_engineering import FeaturesEngineering
from src.models.model_building.model_building import ModelBuilding
from src.models.model_evaluation.model_evaluation import ModelEvaluation
from src.visualization.model_result_analysis  import ModelResultAnalysis
from src.visualization.Features_impact_analysis import FeaturesImpactAnalysis



# Load settings.yaml :
settings_folder = os.path.join(os.path.dirname(__file__), '', 'config')
settings_path = os.path.join(settings_folder, 'settings.yaml')
with open(settings_path, "r") as settings_file:
        settings = yaml.safe_load(settings_file)
        
# Load the model :
model_folder = os.path.join(os.path.dirname(__file__), '', 'models')
model_path = os.path.join(model_folder, 'scoring_model.pkl')
with open(model_path, 'rb') as model_file:
    scoring_model = joblib.load(model_file) # or pickle.load(file)
 
# import df_train for pre-processing :
df_folder = os.path.join(os.path.dirname(__file__), '', 'data/processed')
df_path = os.path.join(df_folder, 'df_train.csv')
with open(df_path, 'r') as file:
    df_train = pd.read_csv(file, sep=";")
    


# Load the image : 
image_folder = os.path.join(os.path.dirname(__file__), '', 'reports\\figures')
image_path = os.path.join(image_folder, 'Customer_Churn.png')
image = Image.open(image_path)    



#download data as csv :
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')




# Interface building :
# Title :
st.title('Scoring modeling :')
# Image :
st.image(image, caption='Customer churn schema')


# We assume that df was already passed the first 4 step treatment :  
# basic treatment, anormale, feature transformation, handle_missing_value
def main():
    
    st.header("Dataset clusturing and theme analysis")
    # Import x_test CSV :
    uploaded_file = st.file_uploader("Choose a file (only accept .csv)")
    if uploaded_file is not None :
        x_test = pd.read_csv(uploaded_file, sep=';')

        # split df_train to x_train and y_train (idem for df_val)
        x_train = df_train.drop([settings["features_info"]["target"]], axis=1)
        y_train = df_train[settings["features_info"]["target"]]

        # 3/ Features engineering :
        features_engineering = FeaturesEngineering()
        
        # Feature creation on x_train
        x_train = features_engineering.features_creation(x_df=x_train)
        # Feature creation on x_val
        x_test = features_engineering.features_creation(x_df=x_test)
        
        # (1st) feature selection with filter methods (apply selection on x_train and y_train, then drop features on x_train and x_val)
        x_test = x_test.drop(settings["features_info"]["columns_to_drop"], axis=1)
        
        # encoding and scaling (apply on x_train and x_val), (don't take x_train_prepocessed and x_val_proprocessed)
        #numeric and > 2 :
        list_cont_col = x_train.select_dtypes(include=[np.number]).columns.tolist()
        list_cont_col = [col for col in list_cont_col if x_train[col].nunique() > 2]
        
        #numeric and <= 2 :
        list_binary_col = x_train.select_dtypes(include=[np.number]).columns.tolist()
        list_binary_col = [col for col in list_binary_col if x_train[col].nunique() <= 2]
        
        #categorical col :
        #list_cat_col = x_train.select_dtypes(include=['object']).columns.tolist()
        list_cat_col_OHE = settings["encoding_scaling"]["list_cat_col_OHE"] #['CUSTOMER_GENDER']
        list_cat_col_TE =  settings["encoding_scaling"]["list_cat_col_TE"] #['marque']
        
        # encoding and scaling
        x_train = features_engineering.encoding_scaling(x_df=x_train, categorical_var_OHE=list_cat_col_OHE, categorical_var_OrdinalEncoding={}, 
                                                        categorical_var_TE=list_cat_col_TE, target=y_train, 
                                                        continious_var=[], encoding_type_cont=StandardScaler(), train=True)
        
        x_test = features_engineering.encoding_scaling(x_df=x_test, categorical_var_OHE=list_cat_col_OHE, categorical_var_OrdinalEncoding={}, 
                                                        categorical_var_TE=list_cat_col_TE, target=y_train, 
                                                        continious_var=[], encoding_type_cont=StandardScaler(), train=False)
        
        # Predict chrun score :
        x_test_pred = pd.DataFrame(scoring_model.predict(x_test)).rename(columns={0: 'prediction'})
        x_test_proba = pd.DataFrame(scoring_model.predict_proba(x_test)).rename(columns={0: 'proba_0', 1:"proba_1"})
        x_test_prediction = pd.concat([x_test, x_test_pred, x_test_proba], axis=1) 

        # Show prediction :
        st.dataframe(x_test_prediction)

        #download prediction as csv :
        csv_all = convert_df(x_test_prediction)
        st.download_button(
            label="Download all prediction as CSV",
            data=csv_all,
            file_name='x_test_prediction.csv',
            mime='text/csv')
        
        csv_potential_churner = convert_df(x_test_prediction[x_test_prediction['proba_1'] >= settings["reports"]["Model_result_analysis"]["BEST_THRESHOLD"]])
        st.download_button(
            label= "Download only potential churner as CSV",
            data=csv_potential_churner,
            file_name='x_test_potiential_churner.csv',
            mime='text/csv')
        
    





# __name__ :
if __name__ == '__main__' :
    main() 
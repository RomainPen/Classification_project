import yaml
import pandas as pd
import numpy as np
from src.data.data_cleaning.data_cleaning import Data_Cleaning
from sklearn.model_selection import train_test_split
from src.visualization.EDA  import EDA
from src.data.features_engineering.features_engineering import FeaturesEngineering





def main(random_state) :
    # Load project setting from setting.yaml :
    with open("config/settings.yaml", "r") as settings_file:
        settings = yaml.safe_load(settings_file)
    
    # Open raw data :
    df = pd.read_csv(settings["data"]["raw_data_path"], index_col=False, sep=";")
    df = df.iloc[:, 1:]
    
    # 1/ Data cleaning : (class data_cleaning)
    data_cleaning = Data_Cleaning()
    
    # Basics treatments of df
    df = data_cleaning.basic_treatment(df=df, useless_columns=["CONTRACT_KEY"])
    
    # Treat anormal values
    df = data_cleaning.treat_anormal_variables(df=df)
    
    # open external data + feature trasnformation 
    external_df = pd.read_csv(settings["data"]["external_data_path"])
    df = data_cleaning.features_transformation(df=df, external_df=external_df)
    
    # Split df (df_train, df_val)
    #target = settings["features_info"]["target"]
    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df[settings["features_info"]["target"]], random_state=random_state)
    #reset index :
    df_train = df_train.reset_index(drop = True)
    df_val = df_val.reset_index(drop = True)
    
    # Handle missing values on df_train
    df_train = data_cleaning.impute_missing_values(df=df_train, target=settings["features_info"]["target"], train=True)
    
    # Handle missing values on df_val
    df_val = data_cleaning.impute_missing_values(df=df_val, target=settings["features_info"]["target"], train=False)

    #*************************************************to review***********************************************
    # 2/ EDA on df_train : (class EDA) #Find a way to save all of this summary
    Exp_data_analysis = EDA()
    
    # Skim(df)
    df_train_summary = Exp_data_analysis.general_summary_skimpy(df_train)
    
    # target feature distribution group by categorical features
    target_distrib_groupby_cat_features = Exp_data_analysis.target_feature_distribution_groupby_categorical_features(df_train)
    
    # target feature distribution group by numerical features
    target_distrib_groupby_num_features = Exp_data_analysis.target_feature_distribution_groupby_numerical_features(df_train)
    
    # correlation matrix
    corr_matrix_df_train = Exp_data_analysis.correlation_matrix(df_train)
    #***********************************************************************************************************
    
    # 2*/ Save and split dataframes :
    # save df_train and df_val to .csv
    df_train.to_csv(path_or_buf=settings["data"]["df_train"], sep=';', index=False)
    df_val.to_csv(path_or_buf=settings["data"]["df_val"], sep=';', index=False)
    
    # split df_train to x_train and y_train (idem for df_val)
    x_train = df_train.drop([settings["features_info"]["target"]], axis=1)
    y_train = df_train[settings["features_info"]["target"]]

    x_val = df_val.drop([settings["features_info"]["target"]], axis=1)
    y_val = df_val[settings["features_info"]["target"]]
    
    # save x_train and x_val (for testing data on app.py)
    x_train.to_csv(path_or_buf=settings["data"]["x_train"], sep=';', index=False)
    x_val.to_csv(path_or_buf=settings["data"]["x_val"], sep=';', index=False)
    
    
    #****************************************************NEXT STEP********************************************
    # 3/ Feature engineering : (class feature engineering)
    features_engineering = FeaturesEngineering()
    
    # Feature creation on x_train
    x_train = features_engineering.features_creation(x_df=x_train)
    
    # Feature creation on x_val
    x_val = features_engineering.features_creation(x_df=x_val)
    
    # add to setting.yaml, list of new col #to review
    #settings["features_info"]["new_features"] = list(x_train.loc[:,"marque":].columns) 
    
    # (1st) feature selection with filter methods (apply selection on df_train (or try on x_train and y_train), then drop features on x_train and x_val)
    # add to setting.yaml, list of col to drop
    # encoding and scaling (apply on x_train and x_val), (don't take x_train_prepocessed and x_val_proprocessed)
    
    # 4/ Modelisation (xgboost) : (class model_building)
    # Hyperparam optimization
    # save best hyper param in config/hyperparmeters.yaml
    # train model with best hyper param
    # save model
    
    # 5/ model evaluation : (class model_evaluation)
    # print roc_auc score (train, test, stratified cv)
    # print f1_score (train, test, stratified cv)
    
    # ****************Save all plot in reports folder*****************************
    # 6/ Analyse model result on val set : (class model_result_analysis)
    # Confusion matrix
    # Roc_auc curve
    # Find best threshold
    # lift curve
    # learning curve
    
    # 7/ model interpretation train set : (class Features_impact_analysis, global analysis) 
    # feature importance
    # plot beeswarm
    # plot summary_plot
    # plot bar
    


if __name__ == "__main__":
    # Load project setting from setting.yaml :
    with open("config/settings.yaml", "r") as settings_file:
        settings = yaml.safe_load(settings_file)
    main(random_state=settings["random_state"])
    
    
    
    
    
    
    
    
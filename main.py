import pandas as pd
import numpy as np


def main() :
    # Load project setting from setting.yaml :
    
    # Open raw data :
    
    # 1/ Data cleaning : (class data_cleaning)
    # Basics treatments of df
    # Treat anormal values
    # feature trasnformation + open external data
    
    # Split df (df_train, df_val)
    
    # Handle missing values on df_train
    # Handle missing values on df_val
    
    # 2/ EDA on df_train : (class EDA)
    # Skim(df)
    # target feature distribution group by categorical features
    # target feature distribution group by numerical features
    # correlation matrix
    
    # save df_train and df_val to .csv
    
    # split df_train to x_train and y_train (idem for df_val)
    # save x_train and x_val (for testing data on app.py)
    
    # 3/ Feature engineering : (class feature engineering)
    # Feature creation on x_train
    # Feature creation on x_val
    # add to setting.yaml, list of new col
    
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
    # 6/ Analyse model result on val set : (class model result analysis)
    # Confusion matrix
    # Roc_auc curve
    # Find best threshold
    # lift curve
    # learning curve
    # feature importance
    
    # 7/ model interpretation train set : (class SHAP, global analysis) 
    # plot beeswarm
    # plot summary_plot
    # plot bar
    


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
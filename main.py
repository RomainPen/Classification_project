import yaml
import pandas as pd
import numpy as np
from src.data.data_cleaning.data_cleaning import Data_Cleaning
from sklearn.model_selection import train_test_split
from src.visualization.EDA  import EDA
from src.data.features_engineering.features_engineering import FeaturesEngineering
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer  
from src.models.model_building.model_building import ModelBuilding
import joblib
from src.models.model_evaluation.model_evaluation import ModelEvaluation
from src.visualization.model_result_analysis  import ModelResultAnalysis
from src.visualization.Features_impact_analysis import FeaturesImpactAnalysis



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
    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df[settings["features_info"]["target"]], random_state=random_state)
    #reset index :
    df_train = df_train.reset_index(drop = True)
    df_val = df_val.reset_index(drop = True)
    
    # Handle missing values on df_train
    df_train = data_cleaning.impute_missing_values(df=df_train, target=settings["features_info"]["target"], train=True)
    
    # Handle missing values on df_val
    df_val = data_cleaning.impute_missing_values(df=df_val, target=settings["features_info"]["target"], train=False)




    # 2/ EDA on df_train : (class EDA) #Find a way to save all of this summary
    Exp_data_analysis = EDA(df=df_train, target=settings["features_info"]["target"])
    
    # Skim(df)
    #Exp_data_analysis.general_summary_skimpy()
    
    # target feature distribution group by categorical features
    Exp_data_analysis.target_feature_distribution_groupby_categorical_features(file_saving=settings["reports"]["EDA"]["target_distrib_groupby_cat_features"])
    
    # target feature distribution group by numerical features
    #Exp_data_analysis.target_feature_distribution_groupby_numerical_features(file_saving=settings["reports"]["EDA"]["target_distrib_groupby_num_features"])
    
    # correlation matrix
    Exp_data_analysis.correlation_matrix(file_saving=settings["reports"]["EDA"]["corr_matrix_df_train"])

    
    # 2*/ Save and split dataframes : basic treatment, anormale, feature transformation, handle_missing_value
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
    
    
    
    
    # 3/ Feature engineering : (class feature engineering)
    features_engineering = FeaturesEngineering()
    
    # Feature creation on x_train
    x_train = features_engineering.features_creation(x_df=x_train)
    
    # Feature creation on x_val
    x_val = features_engineering.features_creation(x_df=x_val)
    
    # add to setting.yaml, list of new col #to review
    #settings["features_info"]["new_features"] = list(x_train.loc[:,"marque":].columns) 
    
    # (1st) feature selection with filter methods (apply selection on x_train and y_train, then drop features on x_train and x_val)
    x_train = features_engineering.filter_selection(x_df=x_train, y_df=y_train, target=settings["features_info"]["target"], train=True)
    x_val = features_engineering.filter_selection(x_df=x_val, y_df=None, target=settings["features_info"]["target"], train=False)
    
    # add to setting.yaml, list of col to drop : to review
    #settings["features_info"]["columns_to_drop"] = list
    
    # encoding and scaling (apply on x_train and x_val), (don't take x_train_prepocessed and x_val_proprocessed)
    #numeric and > 2 :
    list_cont_col = [] #x_train.select_dtypes(include=[np.number]).columns.tolist()
    list_cont_col = [] #[col for col in list_cont_col if x_train[col].nunique() > 2]
    
    #numeric and <= 2 :
    list_binary_col = x_train.select_dtypes(include=[np.number]).columns.tolist()
    list_binary_col = [col for col in list_binary_col if x_train[col].nunique() <= 2]
    
    #categorical col :
    #list_cat_col = x_train.select_dtypes(include=['object']).columns.tolist()
    list_cat_col_OHE = ['CUSTOMER_GENDER']
    list_cat_col_TE =  ['marque']
    
    # encoding and scaling
    x_train = features_engineering.encoding_scaling(x_df=x_train, categorical_var_OHE=list_cat_col_OHE, categorical_var_OrdinalEncoding={}, 
                                                    categorical_var_TE=list_cat_col_TE, target=y_train, 
                                                    continious_var=list_cont_col, encoding_type_cont=MinMaxScaler(), train=True)
    
    x_val = features_engineering.encoding_scaling(x_df=x_val, categorical_var_OHE=list_cat_col_OHE, categorical_var_OrdinalEncoding={}, 
                                                    categorical_var_TE=list_cat_col_TE, target=y_train, 
                                                    continious_var=list_cont_col, encoding_type_cont=MinMaxScaler(), train=False)

    # feature selection with RFE : didn't used here because to slow
    
    
    
    
    # 4/ Modelisation (xgboost) : (class model_building)
    model_building = ModelBuilding(random_state=random_state)
    
    # # train model with best hyper param
    with open("config/hyperparameters.yaml", "r") as hyperparameters_file:
        hyperparameters = yaml.safe_load(hyperparameters_file)
    
    model = model_building.train(x_train=x_train, y_train=y_train, hyper_parameters = hyperparameters["hyper_param_optimisation"], 
                                 scoring=settings["optimisation_metric"], k_fold=5, n_trials=2) # or scoring = f1_weighted, accuracy, roc_auc
    
    # save best hyper param in config/hyperparmeters.yaml
    #hyperparmeters["best_hyparameters"] = dict
    
    # save best model
    joblib.dump(value = model, filename = settings["models"])
    
    
    
    
    
    # 5/ model evaluation : (class model_evaluation)
    model_evaluation = ModelEvaluation(model=model, random_state=random_state)
    # accuracy score :
    accuracy_train = model_evaluation.ACCURACY_score(x_df=x_train, y_df=y_train, cross_val=False, k_fold=None)
    accuracy_val = model_evaluation.ACCURACY_score(x_df=x_val, y_df=y_val, cross_val=False, k_fold=None)
    accuracy_cv_train = model_evaluation.ACCURACY_score(x_df=x_train, y_df=y_train, cross_val=True, k_fold=5)
    print(f"accuracy cv : {accuracy_cv_train}")
    
    # print roc_auc score (train, test, stratified cv)
    roc_auc_train = model_evaluation.ROC_AUC_score(x_df=x_train, y_df=y_train, cross_val=False, k_fold=None)
    roc_auc_val = model_evaluation.ROC_AUC_score(x_df=x_val, y_df=y_val, cross_val=False, k_fold=None)
    roc_auc_cv_train = model_evaluation.ROC_AUC_score(x_df=x_train, y_df=y_train, cross_val=True, k_fold=5)
    print(f"roc_auc cv : {roc_auc_cv_train}")
    
    # print f1_score (train, test, stratified cv)
    f1_score_train = model_evaluation.F1_score(x_df=x_train, y_df=y_train, cross_val=False, k_fold=None)
    f1_score_val = model_evaluation.F1_score(x_df=x_val, y_df=y_val, cross_val=False, k_fold=None)
    f1_score_cv_train = model_evaluation.F1_score(x_df=x_train, y_df=y_train, cross_val=True, k_fold=5)
    print(f"f1_score cv : {f1_score_cv_train}")
    
    # 5*/ creat the predict method in model_building (not sure, may be useless)
    



    # 6/ Analyse model result on val set : (class model_result_analysis)
    model_result_analysis = ModelResultAnalysis(model=model, x_df=x_val, y_df=y_val)
    
    # Confusion matrix 
    model_result_analysis.CONFUSION_MATRIX(file_location=settings["reports"]["Model_result_analysis"]["CONFUSION_MATRIX"])
    classification_report = model_result_analysis.CLASSIFICATION_REPORT()
    
    # Roc_auc curve
    model_result_analysis.ROC_AUC_curve(file_location=settings["reports"]["Model_result_analysis"]["ROC_AUC_curve"])
    
    # Find best threshold
    best_threshold = model_result_analysis.BEST_THRESHOLD(prix_1recharge=2, pourcentage_profit=0.6, remise_pour_1churner_predit_1mois=7)
    print(f"best threshold : {best_threshold}")
    
    # lift curve
    model_result_analysis.LIFT_CURVE(file_location=settings["reports"]["Model_result_analysis"]["LIFT_CURVE"])
    
    # learning curve
    model_result_analysis.LEARNING_CURVE(cv=10, scoring="f1", file_location=settings["reports"]["Model_result_analysis"]["LEARNING_CURVE"])
    
    
    
    
    # 7/ model interpretation train set : (class Features_impact_analysis, global analysis) 
    features_impact_analysis = FeaturesImpactAnalysis(model=model, x_df=x_train[:5000])
    
    # feature importance
    features_importance = features_impact_analysis.Features_importance()
    
    # plot beeswarm (later)
    
    # plot summary_plot
    features_impact_analysis.SHAP_explainer(file_saving=settings["reports"]["Features_impact_analysis"]["SHAP_explainer_summary"])
    
    # plot bar (later)
    


if __name__ == "__main__":
    # Load project setting from setting.yaml :
    with open("config/settings.yaml", "r") as settings_file:
        settings = yaml.safe_load(settings_file)
    main(random_state=settings["random_state"])
    
    
    
    
    
    
    
    
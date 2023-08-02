import pandas as pd 
import numpy as np 
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import yaml



class ModelBuilding:
    def __init__(self, random_state):
        self.random_state = random_state
        
    
    def objective(self, trial):
        n_estimators = trial.suggest_int(name='n_estimators', low=self.hyper_parameters['n_estimators']["low"],
                                         high=self.hyper_parameters['n_estimators']["high"]) #nb of tree
        max_depth = trial.suggest_int(name='max_depth', low=self.hyper_parameters['max_depth']["low"],
                                         high=self.hyper_parameters['max_depth']["high"]) #profondeur
        min_child_weight = trial.suggest_int(name='min_child_weight', low=self.hyper_parameters['min_child_weight']["low"],
                                         high=self.hyper_parameters['min_child_weight']["high"])
        learning_rate = trial.suggest_float(name='learning_rate', low=self.hyper_parameters['learning_rate']["low"],
                                         high=self.hyper_parameters['learning_rate']["high"])
        min_split_loss = trial.suggest_float(name='min_split_loss', low=self.hyper_parameters['min_split_loss']["low"],
                                         high=self.hyper_parameters['min_split_loss']["high"])
        colsample_bytree = trial.suggest_float(name='colsample_bytree', low=self.hyper_parameters['colsample_bytree']["low"],
                                         high=self.hyper_parameters['colsample_bytree']["high"]) #min leaf of each tree
        subsample = trial.suggest_float(name='subsample', low=self.hyper_parameters['subsample']["low"],
                                         high=self.hyper_parameters['subsample']["high"])

        model = XGBClassifier(random_state=self.random_state, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight,
                            learning_rate=learning_rate, min_split_loss=min_split_loss, colsample_bytree=colsample_bytree, subsample=subsample,
                            n_jobs=-1) #if gpu : , tree_method='gpu_hist', predictor="gpu_predictor"

        skf = StratifiedKFold(n_splits=self.k_fold, shuffle=True, random_state=self.random_state)
        skf.get_n_splits(self.x_train, self.y_train) 
        return cross_val_score(model, self.x_train, self.y_train, n_jobs=-1, cv=skf, scoring=self.scoring).mean() # or scoring = f1_weighted, accuracy, roc_auc

    
    
    def train(self, x_train, y_train, hyper_parameters:dict, scoring:str, k_fold:int, n_trials:int):
        self.x_train = x_train
        self.y_train = y_train
        self.hyper_parameters = hyper_parameters
        self.scoring = scoring
        self.k_fold = k_fold
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective, n_trials=n_trials)
        trial = self.study.best_trial
        self.best_param = trial.params
        
        self.best_model = XGBClassifier(random_state=self.random_state, n_estimators=(self.best_param)["n_estimators"], 
                                        max_depth=(self.best_param)["max_depth"], min_child_weight=(self.best_param)["min_child_weight"], 
                                        learning_rate=(self.best_param)["learning_rate"], min_split_loss=(self.best_param)["min_split_loss"],
                                        colsample_bytree=(self.best_param)["colsample_bytree"], subsample=(self.best_param)["subsample"], 
                                        n_jobs=-1) #if gpu : , tree_method='gpu_hist', predictor="gpu_predictor"
        self.best_model.fit(x_train, y_train)
        
        # save best hyperparameters in hyperparameters.yaml:
        
        # print result :
        print(f"{self.scoring} score cv : {trial.value}")
        print(f"Best hyperameters : {self.best_param}")
        
        return self.best_model
    
     

    
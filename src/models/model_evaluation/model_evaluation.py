import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


class ModelEvaluation:
    def __init__(self, model, random_state):
        self.model = model
        self.random_state = random_state
        pass
    
    def ACCURACY_score(self, x_df, y_df, cross_val:bool, k_fold:int):
        if cross_val:
            skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=self.random_state)
            skf.get_n_splits(x_df, y_df)
            return cross_val_score(self.model, x_df, y_df, cv=skf, scoring='accuracy').mean()
            # next step : print train and val score of each fold 
        else : 
            pred = self.model.predict(x_df)
            return accuracy_score(y_df, pred)
    
    
    def ROC_AUC_score(self, x_df, y_df, cross_val:bool, k_fold:int):
        if cross_val :
            skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=self.random_state)
            skf.get_n_splits(x_df, y_df)
            return cross_val_score(self.model, x_df, y_df, cv=skf, scoring='roc_auc').mean()
            # next step : print train and val score of each fold 
        else : 
            pred = self.model.predict_proba(x_df)[:, 1]
            return roc_auc_score(y_df, pred) #if multiclass, then : multi_class='ovr'
    
    
    def F1_score(self, x_df, y_df, cross_val:bool, k_fold:int):
        if cross_val :
            skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=self.random_state)
            skf.get_n_splits(x_df, y_df)
            return cross_val_score(self.model, x_df, y_df, cv=skf, scoring='f1_micro').mean()
            # next step : print train and val score of each fold 
        else : 
            pred = self.model.predict(x_df)
            return f1_score(y_df, pred, average='micro', pos_label=1)  #pos_label = 1 -> number of the target #or average=binary
    
    
    
    
    
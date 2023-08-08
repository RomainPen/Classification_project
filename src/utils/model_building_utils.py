import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def objective(trial, random_state, x_train, y_train, hyper_parameters, k_fold, scoring):
    n_estimators = trial.suggest_int(name='n_estimators', low=hyper_parameters['n_estimators']["low"],
                                      high=hyper_parameters['n_estimators']["high"])
    max_depth = trial.suggest_int(name='max_depth', low=hyper_parameters['max_depth']["low"],
                                  high=hyper_parameters['max_depth']["high"])
    min_child_weight = trial.suggest_int(name='min_child_weight', low=hyper_parameters['min_child_weight']["low"],
                                         high=hyper_parameters['min_child_weight']["high"])
    # ... (suggestions for other hyperparameters)
    
    model = XGBClassifier(random_state=random_state, n_estimators=n_estimators, max_depth=max_depth, min_child_weight=min_child_weight)
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_state)
    skf.get_n_splits(x_train, y_train) 
    return cross_val_score(model, x_train, y_train, n_jobs=-1, cv=skf, scoring=scoring).mean()

def train_model(x_train, y_train, hyper_parameters, scoring, k_fold, n_trials, random_state):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, random_state, x_train, y_train, hyper_parameters, k_fold, scoring), n_trials=n_trials)
    best_param = study.best_params
    
    best_model = XGBClassifier(random_state=random_state, n_estimators=best_param["n_estimators"],
                                max_depth=best_param["max_depth"], min_child_weight=best_param["min_child_weight"])
    best_model.fit(x_train, y_train)
    
    print(f"{scoring} score cv : {study.best_value}")
    print(f"Best hyperparameters : {best_param}")
    
    return best_model


#******************************************In model_building.py*************************************************
'''
class ModelBuilding:
    # ...

    def objective(self, trial):
        return objective(trial, self.random_state, self.x_train, self.y_train, self.hyper_parameters, self.k_fold, self.scoring)

    def train(self, x_train, y_train, hyper_parameters, scoring, k_fold, n_trials):
        # ...
        self.study.optimize(self.objective, n_trials=n_trials)
        trial = self.study.best_trial
        self.best_param = trial.params
        
        self.best_model = train_model(x_train, y_train, self.best_param, scoring, k_fold, n_trials, self.random_state)
        # ...

'''
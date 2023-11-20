from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_curve, recall_score, precision_score
import scikitplot as skplt
from sklearn.model_selection import learning_curve
import numpy as np


class ModelResultAnalysis:
    def __init__(self, model, x_df : pd.DataFrame, y_df : pd.Series):
        """
        Initialize the ModelResultAnalysis object.

        Args:
            model: The machine learning model to analyze.
            x_df (pd.DataFrame): The input features DataFrame.
            y_df (pd.Series): The target variable Series.

        Returns:
            None
        """
        self.model = model
        self.x_df = x_df
        self.y_df = y_df
        # prediction on x_df
        self.y_pred = self.model.predict(self.x_df)
        self.y_pred_proba = self.model.predict_proba(self.x_df)
        
    
    def CONFUSION_MATRIX(self, file_location):
        """
        Create and save a confusion matrix plot.

        Args:
            file_location: The file path for saving the confusion matrix plot in PNG format.

        Returns:
            None
        """
        # compute the confusion matrix
        cm = confusion_matrix(self.y_df, self.y_pred)
 
        f, ax=plt.subplots(figsize=(6,6))
        sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
        plt.ylabel('Actual',fontsize=13)
        plt.xlabel('Prediction',fontsize=13)
        plt.title('Confusion Matrix on y_val',fontsize=17)
        plt.savefig(file_location, format='png')
        plt.close()
        
    
    def CLASSIFICATION_REPORT(self):
        """
        Generate and return a classification report for the model's predictions.

        Returns:
            A classification report containing metrics such as precision, recall, and F1-score.
        """
        return classification_report(self.y_df, self.y_pred)
    
    
    def ROC_AUC_curve(self, file_location):
        """
        Create and save a Receiver Operating Characteristic (ROC) curve.

        Args:
            file_location : The file path for saving the ROC curve plot in PNG format.

        Returns:
            None
        """
        fpr_val_XGB, tpr_val_XGB, thresholds_val_XGB = roc_curve(self.y_df, self.model.predict_proba(self.x_df)[:,1])
        roc_auc_val_XGB = auc(fpr_val_XGB, tpr_val_XGB)
        
        plt.figure()
        plt.plot(fpr_val_XGB, tpr_val_XGB, color='darkgreen',
                 label='Val - ROC curve (area = %0.3f)' % roc_auc_val_XGB)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Comparaison courbes ROC Train/Val')
        plt.legend(loc="lower right")
        plt.savefig(file_location, format='png')
        plt.close()
    
    
    
    def BEST_THRESHOLD(self, prix_1recharge : float, pourcentage_profit : float, remise_pour_1churner_predit_1mois : float) -> dict :
        """
        Find the best threshold for model predictions based on maximizing profit and minimizing profit loss.
        # Objectif : limiter la perte de bénéfice sur les 12 prochains mois grace au modèle. Avoir un Gain_sur_perte_de_profit_grace_model_1mois le plus élevé possible.
        # Solution : proposer une remise de 5 euros par mois pendant 1an aux potentiel churners.

        Returns:
            dict: A dictionary containing information about the best threshold and its impact on profit.
        """
        # Buid table_choix_seuil :
        precision, recall, thresholds = precision_recall_curve(self.y_df, self.y_pred_proba[:, 1])
        table_choix_seuil = pd.DataFrame()
        table_choix_seuil["SEUIL"] = [0] + list(thresholds)
        table_choix_seuil["Precision_val"] = precision
        table_choix_seuil["Recall_val"] = recall
        table_choix_seuil = table_choix_seuil.sort_values(by = "SEUIL", axis=0, ascending=False)
        table_choix_seuil = pd.DataFrame(table_choix_seuil)
        
        
        best_seuil = {"seuil": 0,
                    "recall" : 0,
                    "precision": 0,
                    "Profit_total_mois_prochain_sans_model_1mois" : 0,
                    "Profit_total_mois_prochain_avec_model_1mois": 0,
                    "Gain_sur_perte_de_profit_grace_model_1mois" : 0}


        # Gain sur perte sur 1 mois :
        nb_recharge_moyenne_1mois_1client = self.x_df["AVERAGE_CHARGE_6M"].mean()/6
        CA_1client_1mois = nb_recharge_moyenne_1mois_1client*prix_1recharge
        profit_1client_1mois = pourcentage_profit*CA_1client_1mois
        Profit_total_du_mois_actuel = len(self.x_df)*profit_1client_1mois #profit au moment où les churners ne sont pas encore parties
        best_seuil["Profit_total_du_mois_actuel"] = Profit_total_du_mois_actuel

        for seuil in tqdm(table_choix_seuil['SEUIL']) :
            y_predict_seuil = (self.y_pred_proba[:, 1]>=seuil)
            Confusion_matrix = confusion_matrix(self.y_df, y_predict_seuil)
            Confusion_matrix = pd.DataFrame(Confusion_matrix)

            # Calcul Profit_total_mois_prochain_sans_model :
            nb_client_churner_obs = Confusion_matrix[1][1] + Confusion_matrix[0][1] #Tot_OBS_1 = observation total de tous les churner (True_positive + False_negative)
            Profit_total_mois_prochain_sans_model = Profit_total_du_mois_actuel - profit_1client_1mois*nb_client_churner_obs

            # Calcul Profit_total_mois_prochain_avec_model :
            TP = Confusion_matrix[1][1] #TP = true positive
            FN = Confusion_matrix[0][1] #FP = false negative
            Tot_PRED_1 = Confusion_matrix[1][1] + Confusion_matrix[1][0]  #Tot_PRED_1 = observation total de tous les churner prédit par le modèle (true_pos + false_pos)
            Profit_total_mois_prochain_avec_model = Profit_total_du_mois_actuel - ((Tot_PRED_1*remise_pour_1churner_predit_1mois) + (FN*profit_1client_1mois) - (TP*profit_1client_1mois))

            Gain_sur_perte_de_profit_grace_model_1mois = Profit_total_mois_prochain_avec_model - Profit_total_mois_prochain_sans_model  
            if Gain_sur_perte_de_profit_grace_model_1mois > best_seuil["Gain_sur_perte_de_profit_grace_model_1mois"] :
                best_seuil["seuil"] = seuil
                best_seuil["recall"] = str(recall_score(self.y_df, y_predict_seuil))
                best_seuil["precision"] = str(precision_score(self.y_df, y_predict_seuil))
                best_seuil["Profit_total_mois_prochain_sans_model_1mois"] = Profit_total_mois_prochain_sans_model
                best_seuil["Profit_total_mois_prochain_avec_model_1mois"] = Profit_total_mois_prochain_avec_model
                best_seuil["Gain_sur_perte_de_profit_grace_model_1mois"] = Gain_sur_perte_de_profit_grace_model_1mois

        return best_seuil
    
    
    
    def LIFT_CURVE(self, file_location:str):
        """
        Generate and save a lift curve plot.

        Args:
            file_location (str): The file location to save the plot.

        Returns:
            None
        """
        plt.figure(figsize=(7,7))
        skplt.metrics.plot_lift_curve(self.y_df,self.y_pred_proba)
        plt.savefig(file_location, format='png')
        plt.close()
        
    
    def LEARNING_CURVE(self, cv :int, scoring:str, file_location: str) :
        """
        Generate and save a learning curve plot.

        Args:
            cv (int): Number of cross-validation folds.
            scoring (str): The scoring metric to use for evaluation.
            file_location (str): The file location to save the plot.

        Returns:
            None
        """
        N, train_score, val_score = learning_curve(self.model, self.x_df, self.y_df, train_sizes= np.linspace(0.1,1,10) ,cv=cv, scoring=scoring) #or f1_weighted
        plt.plot(N, val_score.mean(axis=1), label='validation')
        plt.xlabel('train_sizes')
        plt.legend()
        plt.savefig(file_location, format='png')
        plt.close()
        
    
        
        
        
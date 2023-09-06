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
        
        
        
    def BEST_THRESHOLD(self) -> dict:
        """
        Find the best threshold for model predictions based on maximizing profit and minimizing profit loss.

        Returns:
            dict: A dictionary containing information about the best threshold and its impact on profit.
        """
        nb_client=self.x_df.shape[0]
        #objectif : limiter la perte de bénéfice sur les 12 prochains mois grace au modèle, pour ca on propose une offre de reduc de 3euros sur leur forfait pour 1 an
        #ex : Ici, le modele nous a permis de limiter la perte de benefice (ou profit) de 94953.59 euros sur 1 an (pour environs 9800 clients)
        #cela veut dire que si on garde les churner 1 an de plus et bien : au lieu d'avoir une perte de benefice de 302486.4 (pertes sans modele), on ne perd plus que 207532.8 euros (pertes avec modele)

        #forfait mensuel pour 1 client = 18 euros
        #profit mensuel par client (%)= 0.4
        #cout campagne d'offre de reduction pour 1 client pour 1 mois(campagne pub + offre de reduction) = 3 euros
        #coût de l'offre de reduction sur 1 an = 12*3
        precision, recall, thresholds = precision_recall_curve(self.y_df, self.y_pred_proba[:, 1])
        table_choix_seuil_val = pd.DataFrame()
        table_choix_seuil_val["SEUIL"] = [0] + list(thresholds)
        table_choix_seuil_val["Precision_val"] = precision
        table_choix_seuil_val["Recall_val"] = recall
        table_choix_seuil_val = table_choix_seuil_val.sort_values(by = "SEUIL", axis=0, ascending=False)
        table_choix_seuil_val = pd.DataFrame(table_choix_seuil_val)
        
        best_seuil = {"seuil":0,
                    "recall" : 0,
                    "precision": 0,
                    "tot_perte_sans_model" : 0,
                    "tot_perte_avec_model": 0,
                    "profit_net_sauve_grace_au_model_sur_1an" : 0}

        prix_forfait_mensuel_par_client = self.x_df["AVERAGE_CHARGE_6M"].mean()/6
        profit_mensuel_par_forfait_par_client_en_pourcent = 0.4
        profit_mensuel_par_forfait_par_client = profit_mensuel_par_forfait_par_client_en_pourcent*prix_forfait_mensuel_par_client
        cout_campagne_offre_par_client_par_mois = self.x_df["AVERAGE_CHARGE_6M"].mean()/6*0.2

        for i in tqdm(table_choix_seuil_val['SEUIL']) :
            seuil = i
            y_val_predict_seuil = (self.y_pred_proba[:, 1]>=seuil)*1

            Confusion_matrix = confusion_matrix(self.y_df, y_val_predict_seuil)
            Confusion_matrix = pd.DataFrame(Confusion_matrix)

            #calcul :
            #nb d'euros économisés pour 1 mois (faire *12 si on veut pour tous les ans)
            Tot_OBS_1 = Confusion_matrix[1][1] + Confusion_matrix[0][1] #Tot_OBS_1 = observation total de tous les churner
            TP = Confusion_matrix[1][1] #TP =true positive
            Tot_PRED_1 = Confusion_matrix[1][0] + Confusion_matrix[1][1] #Tot_PRED_1 = observation total de tous les churner prédit par le modèle

            tot_perte_avec_model = Tot_OBS_1*profit_mensuel_par_forfait_par_client - (TP*profit_mensuel_par_forfait_par_client - (Tot_PRED_1*cout_campagne_offre_par_client_par_mois))
            tot_perte_sans_model = Tot_OBS_1*profit_mensuel_par_forfait_par_client
            profit_net_sauve_grace_au_model_sur_1an = (tot_perte_sans_model - tot_perte_avec_model)*12

            if profit_net_sauve_grace_au_model_sur_1an > best_seuil["profit_net_sauve_grace_au_model_sur_1an"] :
                best_seuil["seuil"] = seuil
                best_seuil["recall"] = str(recall_score(self.y_df, y_val_predict_seuil))
                best_seuil["precision"] = str(precision_score(self.y_df, y_val_predict_seuil))
                best_seuil["tot_perte_sans_model"] = (tot_perte_sans_model*12)*nb_client
                best_seuil["tot_perte_avec_model"] = (tot_perte_avec_model*12)*nb_client
                best_seuil["profit_net_sauve_grace_au_model_sur_1an"] = profit_net_sauve_grace_au_model_sur_1an*nb_client

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
        
    
        
        
        
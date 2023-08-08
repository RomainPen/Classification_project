import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, recall_score, precision_score
import scikitplot as skplt
from sklearn.model_selection import learning_curve

def confusion_matrix_visualization(y_true, y_pred, file_location):
    cm = confusion_matrix(y_true, y_pred)

    f, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.savefig(file_location, format='png')
    plt.close()

# Define other utility functions for classification_report, ROC_AUC_curve, BEST_THRESHOLD, LIFT_CURVE, LEARNING_CURVE


#************************************In model_result_analysis.py********************************************
'''
class ModelResultAnalysis:
    # ...
    
    def CONFUSION_MATRIX(self, file_location):
        confusion_matrix_visualization(self.y_df, self.y_pred, file_location)
    
    # Define other methods using utility functions defined above
'''

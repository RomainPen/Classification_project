import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
from tqdm import tqdm
from sklearn.model_selection import learning_curve

#TEMPORARY

def plot_confusion_matrix(y_true, y_pred, file_location):
    cm = confusion_matrix(y_true, y_pred)
    f, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
    plt.ylabel('Prediction', fontsize=13)
    plt.xlabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17)
    plt.savefig(file_location, format='png')
    plt.close()

def plot_roc_auc_curve(y_true, y_pred_proba, file_location):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen',
             label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve')
    plt.legend(loc="lower right")
    plt.savefig(file_location, format='png')
    plt.close()

def calculate_best_threshold(y_true, y_pred_proba):
    # Implement the best threshold calculation logic here
    pass

def plot_lift_curve(y_true, y_pred_proba, file_location):
    plt.figure(figsize=(7, 7))
    skplt.metrics.plot_lift_curve(y_true, y_pred_proba)
    plt.savefig(file_location, format='png')
    plt.close()

def plot_learning_curve(estimator, X, y, cv, scoring, file_location):
    N, train_score, val_score = learning_curve(estimator, X, y, train_sizes=np.linspace(0.1, 1, 10), cv=cv, scoring=scoring)
    plt.plot(N, val_score.mean(axis=1), label='validation')
    plt.xlabel('Train Sizes')
    plt.ylabel(scoring.capitalize())
    plt.legend()
    plt.savefig(file_location, format='png')
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba, file_location):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(file_location, format='png')
    plt.close()

import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def get_confusion_matrix_results(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    # Ensure predicts are binary (0 or 1)
    
    TP = float(np.sum((labels == 1) & (predicts == 1)))
    TN = float(np.sum((labels == 0) & (predicts == 0)))
    FN = float(np.sum((labels == 1) & (predicts == 0)))
    FP = float(np.sum((labels == 0) & (predicts == 1)))
    
    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN else 0
    # Precision
    precision = TP / (TP + FP) if TP + FP else 0
    # Recall
    recall = TP / (TP + FN) if TP + FN else 0
    # F1 Score
    f1_score = 2 * precision * recall / (precision + recall) if precision * recall else 0
    
    return accuracy, precision, recall, f1_score

def get_curve_results(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    
    # AUC (Area Under the ROC Curve)
    try:
        auc_score = roc_auc_score(labels, predicts)
    except ValueError:
        auc_score = 0.5 # Default value when calculation fails

    # AUPR (Area Under the Precision-Recall Curve)
    precision_, recall_, _ = precision_recall_curve(labels, predicts)
    aupr_score = auc(recall_, precision_)
    
    return auc_score, aupr_score

# --- New function ---
def get_mean_and_std(data_list):
    """Calculate the mean and standard deviation of a list of data."""
    data = np.array(data_list)
    return np.mean(data), np.std(data)
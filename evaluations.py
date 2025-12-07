import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def get_confusion_matrix_results(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    
    TP = float(np.sum((labels == 1) & (predicts == 1)))
    TN = float(np.sum((labels == 0) & (predicts == 0)))
    FN = float(np.sum((labels == 1) & (predicts == 0)))
    FP = float(np.sum((labels == 0) & (predicts == 1)))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN else 0
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision * recall else 0
    
    return accuracy, precision, recall, f1_score

def get_curve_results(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    
    try:
        auc_score = roc_auc_score(labels, predicts)
    except ValueError:
        auc_score = 0.5 

    precision_, recall_, _ = precision_recall_curve(labels, predicts)
    aupr_score = auc(recall_, precision_)
    
    return auc_score, aupr_score

def get_mean_and_std(data_list):
    data = np.array(data_list)
    return np.mean(data), np.std(data)
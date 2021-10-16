
from .classification_metrics import *
from .regression_metrics import *
import numpy as np


from sklearn.metrics import accuracy_score , auc , balanced_accuracy_score , classification_report , confusion_matrix, \
    f1_score , jaccard_score, matthews_corrcoef, multilabel_confusion_matrix, precision_score , \
    recall_score ,roc_curve , top_k_accuracy_score, cohen_kappa_score

#https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

def get_accuracy_score(preds, gt):
    return( accuracy_score(preds, gt))

def get_auc(preds, gt):
    return( auc(preds, gt))

def balanced_accuracy_score(preds, gt):
    return( balanced_accuracy_score(preds, gt))

def get_classification_report(preds, gt):
    return( classification_report(preds, gt))

def get_confusion_matrix(preds, gt):
    return( confusion_matrix(preds, gt))

def get_f1_score(preds, gt):
    return( f1_score(preds, gt))

def get_jaccard_score(preds, gt):
    return( jaccard_score(preds, gt))

def get_matthews_corrcoef(preds, gt):
    return( matthews_corrcoef(preds, gt))

def get_multilabel_confusion_matrix(preds, gt):
    return( multilabel_confusion_matrix(preds, gt))

def get_precision_score(preds, gt):
    return( precision_score(preds, gt))

def get_recall_score(preds, gt):
    return( recall_score(preds, gt))

def get_roc_curve(preds, gt):
    return( roc_curve(preds, gt))

def get_top_k_accuracy_score(preds, gt):
    return( top_k_accuracy_score(preds, gt))

def get_cohen_kappa_score(preds, gt):
    return( cohen_kappa_score(preds, gt))

def get_classification_statistics(preds, gt):

    auc = 0 #get_auc(preds, gt)
    #get_classification_report(preds, gt)
    #cm = get_confusion_matrix(preds, gt)
    f1 = get_f1_score(preds, gt)
    jaccard = get_jaccard_score(preds, gt)
    mc = get_matthews_corrcoef(preds, gt)
    ml_cm = get_multilabel_confusion_matrix(preds, gt)
    precision = get_precision_score(preds, gt)
    recall = get_recall_score(preds, gt)
    roc = get_roc_curve(preds, gt)
    top_k_accuracy = get_top_k_accuracy_score(preds, gt)
    cohen = get_cohen_kappa_score(preds, gt)

    output = {
        'auc' : auc,
        'cm' :cm,
        'f1' : f1,
        'jaccard' : jaccard,
        'mc' : mc,
        'ml_cm' :ml_cm,
        'precision' : precision,
        'recall' : recall,
        'roc' : roc,
        'top_k_accuracy' : top_k_accuracy,
        'cohen' : cohen
    }

    return(output)

def get_regression_statistics(preds, gt):

    preds = np.array(preds)
    gt = np.array(gt)

    mae = get_mean_absolute_error(preds, gt)
    mse = get_mean_squared_error(preds, gt)
    msle = get_mean_squared_log_error(preds, gt)
    med_mae = get_median_absolute_error(preds, gt)
    #mape = get_mean_absolute_percentage_error(preds, gt)
    r2_score =  get_r2_score(preds, gt)

    ouput = {
        'mae' :mae,
        'mse' :mse,
        'msle' :msle,
        'med_mae' : med_mae,
        #"mape" : mape,
        'r2_score' : r2_score,
    }

    return(ouput)

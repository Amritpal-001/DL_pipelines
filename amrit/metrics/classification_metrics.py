
from sklearn.metrics import accuracy_score , auc , balanced_accuracy_score , classification_report , confusion_matrix, \
    f1_score , jaccard_score, matthews_corrcoef, multilabel_confusion_matrix, precision_score , \
    recall_score ,roc_curve , cohen_kappa_score

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

def get_cohen_kappa_score(preds, gt):
    return( cohen_kappa_score(preds, gt))

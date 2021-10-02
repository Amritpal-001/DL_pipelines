
from sklearn.metrics import mean_absolute_error , mean_squared_error, mean_squared_log_error , median_absolute_error , mean_absolute_percentage_error , r2_score


#https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

def get_mean_absolute_error(preds, gt):
    return(mean_absolute_error(preds, gt))

def get_mean_squared_error(preds, gt):
    return(mean_squared_error(preds, gt))

def get_mean_squared_log_error(preds, gt):
    return(mean_squared_log_error(preds, gt))

def get_median_absolute_error(preds, gt):
    return(median_absolute_error(preds, gt))

def get_mean_absolute_percentage_error(preds, gt):
    return(mean_absolute_percentage_error(preds, gt))

def get_r2_score(preds, gt):
    return(r2_score(preds, gt))


'''
explained_variance_score(y_true, …)
Explained variance regression score function.

metrics.max_error(y_true, y_pred)
max_error metric calculates the maximum residual error.

metrics.r2_score(y_true, y_pred, *[, …])
'''



from .classification_metrics import *
from .regression_metrics import *

def get_classification_statistics():

    print()

def get_regression_statistics(preds, gt):

    mae = get_mean_absolute_error(preds, gt)
    mse = get_mean_squared_error(preds, gt)
    msle = get_mean_squared_log_error(preds, gt)
    med_mae = get_median_absolute_error(preds, gt)
    mape = get_mean_absolute_percentage_error(preds, gt)
    r2_score =  get_r2_score(preds, gt)

    ouput = {
        'mae' :mae,
        'mse' :mse,
        'msle' :msle,
        'med_mae' : med_mae,
        "mape" : mape,
        'r2_score' : r2_score,
    }

    return(ouput)

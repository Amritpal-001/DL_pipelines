from sklearn.metrics import mean_absolute_error , mean_squared_error, mean_squared_log_error , median_absolute_error ,  r2_score
#mean_absolute_percentage_error
#https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

def get_mean_absolute_error(preds, gt):
    return(mean_absolute_error(preds, gt))

def get_mean_squared_error(preds, gt):
    return(mean_squared_error(preds, gt))

def get_mean_squared_log_error(preds, gt):
    return(mean_squared_log_error(preds, gt))

def get_median_absolute_error(preds, gt):
    return(median_absolute_error(preds, gt))

# def get_mean_absolute_percentage_error(preds, gt):
#     return(mean_absolute_percentage_error(preds, gt))

def get_r2_score(preds, gt):
    return(r2_score(preds, gt))

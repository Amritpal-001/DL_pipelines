import pandas

from ..metrics.meter import get_classification_statistics,get_regression_statistics
import numpy as np

from scipy import stats
import pandas as pd
from amrit.utils.dataModifier import analyze
import copy


def scatterplot(self):
        print()

def bland_altman_plot(self):
        print()

def compute_statistics(predictions , ground_truth):
        preds = np.array(predictions)
        gt = np.array(ground_truth)
        print(stats.describe(preds))
        stats = analyze(pd.DataFrame({'preds': preds, 'gt': gt} ))

        return stats


def compute_metrics(prediction: list = None, ground_truth=None, problem=None):
    '''
    input:
    prediction = single dataframes of model predictions
    ground_truth = pandas Data Series(single column)

    Output:
    Computes different metrics
    '''

    if problem == "classification":
        results = get_classification_statistics()
    else:  # problem == "regression":
        stats = compute_statistics()
        results = get_regression_statistics(prediction, ground_truth)
        print(results, stats)

    return results

''':cvar
https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.values.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
'''

## Binary classification

def get_help_index_functions():
    print("get_TP_index(preds , gt) - assumes probabilities, vinary class, and threshold=0.7(default)" )
    print(             "get_TP_index(preds , gt, threshold = None) - assume target outputs are provided for binary classification" )
    print(             "get_TP_index(preds , gt, true_class = 2 , threshold = None) - assumes multiclass output" )
    print(             "-------Invalid uses-------" )
    print(             "get_TP_index(preds , gt , true_class, threshold) ")

def threshold_values(dataframe , threshold):
    df = copy.deepcopy(dataframe)
    df[df >= threshold] = 1
    df[df <= threshold] = 0
    return df

def multiclass_to_binary(dataframe , true_class , presence = 1 , absence = 0):
    df = copy.deepcopy(dataframe)

    df[df == true_class] = presence
    df[df != true_class] = absence

    return df


def get_TP_index(preds , gt, true_class = None , threshold = 0.7):
    '''returns index of cases that are True positive(as per the pred pandas.Series)'''
    presence = 1
    absence = 0

    if true_class != None:
        gt = multiclass_to_binary(gt, true_class)
        preds = threshold_values(preds, threshold)
    else:
        preds = threshold_values(preds , threshold)

    TP_index = list(preds[(preds == presence) * (gt == presence)].index)
    return(TP_index)

def get_FP_index(preds , gt, true_class = None ,threshold = 0.7):
    '''returns index of cases that are False positive(as per the pred pandas.Series)'''
    presence = 1
    absence = 0

    assert true_class == None or threshold == None , "please donot provide both true class and threshold"

    if true_class != None:
        if threshold == None:
            preds = multiclass_to_binary(preds , true_class)
        gt = multiclass_to_binary(gt, true_class)
    if threshold != None:
        preds = threshold_values(preds, threshold)

    FP_index = list(preds[(preds == presence) * (gt == absence)].index)
    return (FP_index)

def get_TN_index(preds , gt, true_class = None ,threshold = 0.7):
    '''returns index of cases that are True negatives(as per the pred pandas.Series)'''
    presence = 1
    absence = 0

    assert true_class == None or threshold == None , "please donot provide both true class and threshold"

    if true_class != None:
        if threshold == None:
            preds = multiclass_to_binary(preds, true_class)
        gt = multiclass_to_binary(gt, true_class)

    if threshold != None:
        preds = threshold_values(preds, threshold)
    TN_index = list(preds[(preds == absence) * (gt == absence)].index)
    return (TN_index)


def get_FN_index(preds , gt , true_class = None , threshold = 0.7):
    '''returns index of cases that are False negatives(as per the pred pandas.Series)'''

    presence = 1
    absence = 0

    assert true_class == None or threshold == None , "please donot provide both true class and threshold"

    if true_class != None:
        if threshold == None:
            preds = multiclass_to_binary(preds , true_class)
        gt = multiclass_to_binary(gt, true_class)

    if threshold != None:
        preds = threshold_values(preds, threshold)
    FN_index = list(preds[(preds == absence) * (gt == presence)].index)
    return (FN_index)


def get_FarNorth_index(preds , gt , threshold = 0.7 ,far_north_threshold = 0.9):
    '''returns index of cases that are True positive(as per the pred pandas.Series)'''
    binary_pred = threshold_values(preds , threshold)

    farNorth_list = preds[get_FP_index(binary_pred, gt)]
    print(farNorth_list)
    print(len(farNorth_list))

    print( farNorth_list[farNorth_list> far_north_threshold].shape)
    #return (farNorth_list)

def get_FarSouth_index(preds , gt,  threshold = 0.7 ,far_south_threshold = 0.9):
    '''returns index of cases that are True positive(as per the pred pandas.Series)'''
    '''returns index of cases that are True positive(as per the pred pandas.Series)'''
    binary_pred = threshold_values(preds, threshold)

    farSouth_list = preds[get_FN_index(binary_pred, gt)]
    print(farSouth_list)
    print(len(farSouth_list))

    print(farSouth_list[farSouth_list > far_south_threshold].shape)
    # return (farNorth_list)


def filter_dfSeries_subset(series: pandas.Series,
                          search_series : pandas.Series ,
                         search_tag: str):
    return series[search_series == search_tag]

def compute_stratified_metrics(prediction: pandas.Series = None,
                               ground_truth: pandas.Series=None,
                               stratify_labels: pandas.Series = None,
                             problem: str=None):
    '''
    input:
    prediction = single dataframes of model predictions
    ground_truth = pandas Data Series(single column)

    Output:
    Computes different metrics
    '''

    assert prediction.isnull().values.any() == False , "Prediction series contains Null values"
    assert ground_truth.isnull().values.any() == False , "ground_truth series contains Null values"
    assert stratify_labels.isnull().values.any() == False , "stratify_labels series contains Null values"

    result_dict = {}

    all_labels = stratify_labels.unique()

    for label in all_labels:
        preds = filter_dfSeries_subset(prediction , stratify_labels , label)
        gt = filter_dfSeries_subset(ground_truth, stratify_labels, label)
        if problem == "classification":
            results = get_classification_statistics(preds , gt)
        else:  # problem == "regression":
            print(preds , gt)
            results = get_regression_statistics(preds, gt)

        result_dict[label] = results

    return result_dict

def compute_metrics_multimodel(prediction :list = None, ground_truth = None,  problem = None):

    '''
    input:
    prediction = list containing indiviual dataframes for different models
    ground_truth = pandas Data Series(single column)

    Output:
    Computes different metrics for all the predictions in the list.
    '''



    for preds in prediction:
        pass
    results = 0

    return results



def compute_stratified_stats(self , dataloader):
        """
        computer stratified stats for individual sub_class
        (or sub_category of a categorical columnn - if mentioned)
        """
        print()

def upload_weight_and_biases():
        print("weights and biases updated")

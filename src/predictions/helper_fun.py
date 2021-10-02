
from ..metrics.meter import get_classification_statistics,get_regression_statistics
import numpy as np
from scipy import stats
import pandas as pd
from src.utils.dataModifier import analyze



def scatterplot(self):
        print()

def bland_altman_plot(self):
        print()

def compute_statistics(self):
        preds = np.array(self.prediction)
        gt = np.array(self.ground_truth)
        print(stats.describe(preds))
        self.stats = analyze(pd.DataFrame({'preds': preds, 'gt': gt} ))

        return self.stats


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


def compute_metrics_in_batch(prediction :list = None, ground_truth = None,  problem = None):

    '''
    input:
    prediction = list containing indiviual dataframes for different models
    ground_truth = pandas Data Series(single column)

    Output:
    Computes different metrics for all the predictions in the list.
    '''



    for preds in prediction:

    return results



def compute_stratified_stats(self , dataloader):
        """
        computer stratified stats for individual sub_class
        (or sub_category of a categorical columnn - if mentioned)
        """
        print()

def upload_weight_and_biases():
        print("weights and biases updated")

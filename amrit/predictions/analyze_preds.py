
import numpy as np
import pandas as pd
from scipy import stats

from amrit.utils.dataModifier import analyze
from ..metrics.meter import get_classification_statistics, get_regression_statistics


class prediction_plotter():

    def __init__(self, prediction , dataloader = None , ground_truth = None , model = None):

        self.prediction = prediction
        if ground_truth == None:
            self.ground_truth = dataloader.Ytest
        else:
            self.ground_truth = ground_truth

        self.dataloader = dataloader

        if model != None:
            self.model = model
            self.problem = self.model.problem
        else:
            self.problem = 'regression'

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



    def compute_metrics(self, problem = None):

        if problem == None:
            problem = self.problem

        if problem == "classification":
            results = get_classification_statistics()
        elif problem == "regression":
            stats = self.compute_statistics()
            results = get_regression_statistics(self.prediction , self.ground_truth)
            print(results ,stats)
            self.result = results

        return results



    def compute_stratified_stats(self , dataloader):

        """
        computer stratified stats for individual sub_class
        (or sub_category of a categorical columnn - if mentioned)
        """
        print()

    def upload_weight_and_biases():

        print("weights and biases updated")

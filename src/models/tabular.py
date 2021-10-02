

# import lgbm
# import catboost
import xgboost
from src.models.model import Model

import matplotlib.pyplot as plt

#import logging
from ..constants import tabular_configs

#logging.basicConfig(filename='testrun.log' , level = logging.INFO , format = '%(asctime)s:%(levelname)s:%(messages)s:')


class tabularmodel(Model):

    def __init__(self, architecture = "xgboost" , problem = "classification" ,
                 output = 'probabilities' , device = 'cpu'):

        self.problem = problem #type of problem - regression/classification
        self.architecture = architecture  #model architecture e.g. - Xgboost, Resnet,
        self.device = device
        self.output = output

        self.config = tabular_configs #[self.architecture][self.device]

        self.check_inputs()

        if self.architecture == 'xgboost' and self.problem == 'regression':
            self.model = xgboost.XGBRegressor(**self.config[self.architecture][self.device])

        if self.architecture == 'xgboost' and self.problem == 'classification':
            self.model = xgboost.XGBClassifier( enable_categorical = True , **self.config[self.architecture][self.device])

        if self.architecture == 'lightgbm' and self.problem == 'regression':
            self.model = xgboost.XGBRegressor(**tabular_configs[self.architecture][self.device])


        """logging.info(f"Tabular model created "
                     f"architecture - {self.architecture},"
                     f"problem - {self.problem}"
                     f"device - {self.device}"
                     f"output - {self.output}")"""


    def get_configs(self):
        print(self.model_configs)
        return(self.model_configs)


    def fit(self, dataloader, split_data = True , split_mode = 'random' , test_size  = 0.1):

        print("starting training")

        self.dataloader = dataloader

        if split_data == True:
                dataloader.train_test_split(mode= split_mode , test_size  = test_size , split_random_seed   = 42)

        if split_mode != 'kfold' or 'stratified':
            self.model = self.model.fit( dataloader.Xtrain , dataloader.Ytrain , eval_metric = ['error', 'logloss'],
                                         eval_set = [ (dataloader.Xtrain , dataloader.Ytrain) , (dataloader.Xtest , dataloader.Ytest)])
        else:
            if split_mode == 'kfold':
                print("training kfold")
            if split_mode == 'stratified':
                print("training stratified")

        print("finished training")

        return(self.model)


    #def plot_
    def plot_train_metrics(self):
        results = self.model.evals_result()

        print(results)
        epochs = len(results['validation_0']['error'])
        X_axis = range(0,epochs)
        fig , ax = plt.subplots(figsize=(6,6))
        ax.plot(X_axis, results['validation_0']['error'] , label = 'Train error')
        ax.plot(X_axis, results['validation_1']['error'] , label = 'Test error')
        ax.plot(X_axis, results['validation_0']['logloss'], label='Train logloss')
        ax.plot(X_axis, results['validation_1']['logloss'], label='Test logloss')

        plt.show()


    def predict(self, Xtest= None):
        if Xtest == None:
            Xtest = self.dataloader.Xtest
        self.preds = self.model.predict(Xtest)
        return(self.preds)

    def save_model(self):
        print()

    def load_model(self):
        print()

    def featureImportance(self):
        print()

    def parameter_search(self):
        print()

    def optuna_search(self):
        print()

    def shapValues(self):
        print()





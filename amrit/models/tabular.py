

import matplotlib.pyplot as plt
# import lgbm
# import catboost
import xgboost

from amrit.models.model import Model
# import logging
from ..constants import tabular_configs


#logging.basicConfig(filename='test_tabular.log' , level = logging.INFO , format = '%(asctime)s:%(levelname)s:%(messages)s:')


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


    def fit_StratifiedKFold(self , dataloader, fold = 5 , stratified = True , split_data = True,   split_mode = 'random' , test_size  = 0.1):
        from sklearn.model_selection import StratifiedKFold, KFold
        import numpy as np
        import gc

        ### https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection

        # KFold - Split dataset into k consecutive folds (without shuffling by default).
        # RepeatedKFold - Repeats K-Fold n times.
        # StratifiedKFold - Takes group information into account to avoid building folds with imbalanced class distributions (for binary or multiclass classification tasks).
        #StratifiedShuffleSplit -

        #### Require addtional arguments - "groups"
        #GroupKFold - K-fold iterator variant with non-overlapping groups.
        #LeaveOneGroupOut - For splitting the data according to explicit domain-specific stratification of the dataset.
        #LeavePGroupsOut -



        if stratified == True:
            skf = StratifiedKFold(n_splits= fold, random_state=None, shuffle=False)
        else:
            skf = KFold(n_splits=fold , random_state=None, shuffle=False)


        if split_data == True:
                dataloader.train_test_split(mode= split_mode , test_size  = test_size , split_random_seed   = 42)

        train = dataloader.Xtrain
        y = dataloader.Ytrain
        test = dataloader.Xtest

        # Declaration Pred Datasets
        train_fold_pred = np.zeros((train.shape[0], 1))
        test_pred = np.zeros((test.shape[0], fold))

        for fold, (train_index, valid_index) in enumerate(skf.split( train , y )
):
            x_train, y_train = train.iloc[train_index], y[train_index]
            x_valid, y_valid = train.iloc[valid_index], y[valid_index]

            print('------------ Fold', fold + 1, 'Start! ------------')
            self.model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric='auc', verbose=50, early_stopping_rounds=200)

            train_fold_pred[valid_index, :] = self.model.predict(x_valid).reshape(-1, 1)
            test_pred[:, fold] = self.model.predict(test)
            del x_train, y_train, x_valid, y_valid
            gc.collect()

        test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)
        del test_pred
        gc.collect()
        print('Done!')

        return train_fold_pred, test_pred_mean



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





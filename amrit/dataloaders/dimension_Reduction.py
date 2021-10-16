

import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import glob

def set_seed(random_seed = 42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    #torch.manual_seed(random_seed)
    #torch.use_deterministic_algorithms(True)

    print(f"setting seed to {random_seed}")


def add_row_to_df(x, y, index_name=None):
        df1 = x.copy()

        df_length = len(df1)
        df1.loc[df_length] = y
        if index_name != None:
            df1.index = list(df1.index[:-1]) + [index_name]
        return (df1)

import scipy
from scipy import stats

def get_data_distribution(data):

    # Please write below the name of the statistical distributions that you would like to check.
    # Full list is here: https://docs.scipy.org/doc/scipy/reference/stats.html
    dist_names = ['weibull_min', 'norm', 'weibull_max', 'beta',
                  'invgauss', 'uniform', 'gamma', 'expon', "exponweib", "pareto", "genextreme",
                  'lognorm', 'pearson3', 'triang']

    # #Read your data and set y_std to the column that you want to fit.
    # y_std=pd.read_csv('my_df.csv')
    # y_std=y_std['column_A']

    # -------------------------------------------------
    chi_square_statistics = []
    size = len(data)

    # 20 equi-distant bins of observed Data
    percentile_bins = np.linspace(0, 100, 20)
    percentile_cutoffs = np.percentile(data, percentile_bins)
    observed_frequency, bins = (np.histogram(data, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(scipy.stats, distribution)
        param = dist.fit(data)
        #print("{}\n{}\n".format(dist, param))

        # Get expected counts in percentile bins  cdf of fitted sistrinution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins) - 1):
            expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum(((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square_statistics.append(ss)

    # Sort by minimum ch-square statistics
    results = pd.DataFrame()
    results['Distribution'] = dist_names
    results['chi_square'] = chi_square_statistics
    results.sort_values(['chi_square'], inplace=True)

    print('\nDistributions listed by goodness of fit:')
    print(results)


def analyze( data , force = False, percentiles = [0.05, 0.10, 0.25,0.50, 0.75, 0.90, 0.95] , include=None, exclude=None):
        '''
        runs basic statitics and analysis on the data
        :return:  None

        #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
        '''

        if force == False:
            if data.shape[0] > 100000:
                print(f"data shape is {data.shape[0]}, sharing statistics for sub sample, use force=True to disable" )
                data = data.sample(20000)
            else:
                data = data

        analysis = data.describe(percentiles=percentiles)
        analysis = add_row_to_df(analysis , data.skew(), index_name='skew')
        analysis = add_row_to_df(analysis , data.kurtosis(), index_name='kurtosis')
        analysis = add_row_to_df(analysis, data.mad(), index_name='mean absolute deviation')
        analysis = add_row_to_df(analysis, data.dtypes, index_name='dtypes')
        analysis = add_row_to_df(analysis, data.isna().sum(), index_name='Null counts')
        #print(analysis)
        return(analysis)


def trainTestSplit(df=None, target_column='target',
                   mode='random', test_size: float = 0.1, split_random_seed: int = 42):
    """
    modes supported
    1. 'random' - split into test train randomly as per test_size
    2. last_half - useful for chronological data - picks last (test_size*100) percent of data
    """

    if mode == 'random':
        test_size = test_size
        split_random_seed = split_random_seed
        train, test = train_test_split(df, test_size=test_size, random_state=split_random_seed)
        print(f"data randomly split into - {train.shape, test.shape} ")

    elif mode == 'last':
        test_size = test_size
        split_random_seed = None
        data_size = df.shape[0]

        split_index = data_size - int(data_size * test_size)
        train = df[:split_index]
        test = df[split_index:]
        print(f"data chronologically split into - {train.shape, test.shape} ")

    target_column = target_column

    if target_column != None:
        Xtrain = train.drop(columns=target_column)
        Xtest = test.drop(columns=target_column)
        Ytrain = train[target_column]
        Ytest = test[target_column]

        return (Xtrain, Xtest, Ytrain, Ytest)
    else:
        return (train, test)


def get_file_list(folder_path: str = None, extension='.dcm', recursive=False):
    assert folder_path != None, "please provide a folder path"
    if recursive == False:
        file_list = glob.glob(folder_path + './*' + extension, recursive=True)
    else:
        file_list = glob.glob(folder_path + '/**/*' + extension, recursive=True)
    file_list = sorted(file_list)

    return (file_list)


def parent_folder(file_name):
    return os.path.basename(os.path.dirname(file_name))


def get_labels_df_from_folder(folder_path: str = None, extension='.dcm', recursive=True):
    file_list = get_file_list(folder_path=folder_path, extension=extension, recursive=recursive)
    print(f"found {len(file_list)} in folder {folder_path}")
    labels = list(map(parent_folder, file_list))

    df = pd.DataFrame({'file_path': file_list, 'target': labels})
    return (df)


def get_df_statistics(df):
    shape = df.shape
    labels = df['target'].unique()
    class_count = df['target'].nunique()

    print(f"shape - {shape} ,"
          f"no. of classes - {class_count} , "
          f"classes - {labels}")

    return (shape, labels, class_count)

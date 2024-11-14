import multiprocessing
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
import numpy as np

import multiprocessing
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
pd.set_option('mode.chained_assignment', None)

import numpy as np

def fit_model(self):
    """
    User defined function for fitting a model.

    Inputs:
    test_size (float): size in % test in train / test split.
    target_column (int): single classifier target column in featureset.
    featureset_columns (list): list of featureset columns to be converted to float.
    classifier (scikit classifier): classifier object.
    dataset (pandas dataframe): pandas dataframe for splitting and fitting
    """
    dataset = self.dataset
    featureset_columns = self.featureset_columns
    target_column = self.target_column
    test_size = self.test_size
    classifier = self.classifier

    last_date = self.dataset.iloc[-1]['date']
    X_train, X_test, y_train, y_test = train_test_split(dataset[featureset_columns], dataset[target_column], test_size = test_size, random_state = 10)
    classifier.fit(X_train, y_train)
    y_pred = list(classifier.predict(X_test))
    y_pred_proba = list(classifier.predict_proba(X_test)[:,1])
    weights = classifier.coef_
    weights = pd.DataFrame(zip(X_train.columns, np.transpose(weights[0])), columns=['features', 'coef']) 
    weights['date'] = last_date
    singleton_dataset = dataset.tail(1)
    y_pred = y_pred[-1]
    y_pred_proba = y_pred_proba[-1]
    singleton_dataset['y_pred'] = y_pred
    singleton_dataset['y_pred_proba'] = y_pred_proba
    return singleton_dataset, weights


class WalkForward:
    """
    A class representing a recordset walkforward back test.

    Attributes:
        test_size (float): size in % test in train / test split.
        target_column (int): single classifier target column in featureset.
        featureset_columns (list): list of featureset columns to be converted to float.
        classifier (scikit classifier): classifier object.
        minimum_dataset_length (float): length in % of initial recordset from which to begin walk forward
    """

    def __init__(self, classifier = None, dataset = None, target_column='',featureset_columns = []):
        """
        Initializes WalkForward object
        """
        self.dataset = dataset
        self.test_size = .2
        self.target_column = target_column
        self.featureset_columns = featureset_columns
        self.target_column=target_column
        self.classifier = classifier
        self.minimum_dataset_length = .2
        return

    def fit(self):
        """
        Fit the classifier
        Initially set the minimum dataset length, then step over each record
        instantiating a WalkForwardSingleton for each record
        then, appending a rolling_df_list with the output of WalkForwardSingleton (a greedy approach for debug)
        concatenate the rolling df list into a result dataset property
        """
        len_df = int(len(self.dataset))
        len_end_x = int(round(len_df * self.minimum_dataset_length,0))
        incremental_end_x = len_end_x
        rolling_df_list = []
        rolling_weights_list = []
        for i in range(len_df - len_end_x):
            incremental_window_data = self.dataset.iloc[0:incremental_end_x]

            clsWalkForwardSingleton = Singleton()
            clsWalkForwardSingleton.classifier = self.classifier
            clsWalkForwardSingleton.target_column = self.target_column
            clsWalkForwardSingleton.featureset_columns = self.featureset_columns
            clsWalkForwardSingleton.test_size = .2
            clsWalkForwardSingleton.dataset = incremental_window_data
            clsWalkForwardSingleton.fit_model = fit_model.__get__(clsWalkForwardSingleton, Singleton)
            singleton_dataset, weights = clsWalkForwardSingleton.fit_model()
            
            incremental_end_x = incremental_end_x + 1

            rolling_df_list.append(singleton_dataset)
            rolling_weights_list.append(weights)

        self.result_dataset = pd.concat(rolling_df_list)
        self.weights_dataset = pd.concat(rolling_weights_list)

        return

class Singleton(multiprocessing.Process):
    """
    A class representing a singleton.
    Inherits from multi processing Process for threading
    """
    pass


if __name__ == "__main__":
    """
    Read sample dataset, do some target arranging / feature engineering
    """
    dataset = pd.read_csv('dataset.csv')
    symbol = "JPM US"
    featureset_columns = ['hyg_return', 'tlt_return', 'vb_return',
        'vtv_return', 'vug_return', 'rut_return', 'spx_return', 'DGS10_return',
        'DGS2_return', 'DTB3_return', 'DFF_return', 'T10Y2Y_return',
        'T5YIE_return', 'BAMLH0A0HYM2_return', 'DEXUSEU_return', 'KCFSI_return',
        'DRTSCILM_return', 'RSXFS_return', 'MARTSMPCSM44000USS_return',
        'H8B1058NCBCMG_return', 'DCOILWTICO_return', 'VXVCLS_return',
        'H8B1247NCBCMG_return', 'GASREGW_return',
        'CSUSHPINSA_return', 'UNEMPLOY_return']

    dataset = dataset.loc[dataset['symbol']==symbol]
    dataset['target'] = dataset['target'].shift(-1)
    prediction_record = dataset[-1:]
    dataset =  dataset.iloc[:-1]
    dataset[featureset_columns] = dataset[featureset_columns].diff(periods=1, axis=0)
    dataset.dropna(inplace=True)

    """
    Send dataset and simple binary classifier into WalkForward object and fit
    """
    classifier = LogisticRegression(random_state = 10)
    clsWalkForward = WalkForward(classifier,dataset,'target',featureset_columns)
    clsWalkForward.fit()
    print(clsWalkForward.result_dataset.tail())


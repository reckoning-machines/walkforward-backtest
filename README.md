# walkforward-backtest

A proposed python library to add recordset walk forward for model fitting

Purpose:

Python library scikit-learn classifiers have predict_proba and predict functions which apply a fitted (trained) model against the input / training dataset

However, for purposes of time series, running the final trained model back against the full dataset is not "as of" - the final trained model allows for leakage of future knowledge into past results

Rather, a true time series approach would fit models in series - that is, apply the model up to a certain day, yield a result, store the result, then fit for the next day, and so on

Proposed usage:

import walkforward as wf 

(your featureset prep here)

result = wf.walkforward_backtest(classifier, dataset, date_column, target_column, feature_columns, window, train_test_split, scoring_function)

result['score'] = rolling score

result['fitted_recordset'] = dataset with predictions and predicted probabilities

result['weights'] = rolling list of weights

result['models'] = rolling list of models

Current Status:

A class wrapper for scikit-learn predict and predict_proba for walk forward backtesting in time series analysis

In an effort to reduce the O, or runtime greediness, of the class, we inherit from multiprocess Process

And, this is a work in process... see "to do" list below

Example:

predict and predict proba back propogated, that is, the default

<img width="440" alt="image" src="https://github.com/user-attachments/assets/3532fbbd-8b29-4206-abcb-797c82d39ba4">

vs predict and predict proba "as of" in a rollforward approach

<img width="175" alt="image" src="https://github.com/user-attachments/assets/da96cf9b-4a50-47c3-af1e-ccbbe6cfb26e">

the differences are material.

Usage:

Modify the user defined function fit_model in the singleton, and apply your return values in the walkforward 'fit' method.  See comments in code.

<img width="604" alt="image" src="https://github.com/user-attachments/assets/2eb49644-6b91-4683-8914-93a6bb169b68">


to do:
- support for multi-class classifiers
- test Process inheritance for runtime improvement if any
- convert to a pip installable library module - will need ability for user defined feature engineering and fit_model function


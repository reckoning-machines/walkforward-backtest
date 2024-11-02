# walkforward-backtest
a class wrapper for scikit-learn predict and predict_proba for walk forward backtesting in time series analysis

scikit-learn classifiers have predict_proba and predict functions which can apply a fitted (trained) model against the full input dataset

however, for purposes of time series, running the final trained model back against the full dataset is not "as of" - the final trained model allows for leakage of future knowledge into past results

rather, a true time series approach would fit models in series - that is, apply the model up to a certain day, yield a result, store the result, then fit for the next day, and so on

in an effort to reduce the O, or runtime greediness, of the class, we inherit from multiprocess Process

and, this is a work in process

Example:
predict and predict proba back propogated

<img width="440" alt="image" src="https://github.com/user-attachments/assets/3532fbbd-8b29-4206-abcb-797c82d39ba4">

vs predict and predict proba "as of" in a rollforward approach

<img width="440" alt="image" src="https://github.com/user-attachments/assets/a8945f58-5e4e-4abd-8a75-cfeb7a0fc1e7">

to do:
- allow for user input for 'score'
- add docstring comments
- allow for feature engineering inside the class
- convert to a pip installable library module
- test Process inheritance for runtime


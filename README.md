# walkforward-backtest
a class wrapper for scikit-learn predict and predict_proba for walk forward backtest

scikit-learn classifiers have a predict_proba and predict functions which apply a trained model against the full input dataset

however, for purposes of time series, running the final trained model back against the full dataset is not "as of"

rather, a time series approach would apply a new model up to a certain day, then yield a result, then the next day, and so on

in an effort to reduce the O, or runtime greediness, of the class, we inherit from multiprocess Process

and, this is a work in process

Example:
predict and predict proba back propogated

<img width="440" alt="image" src="https://github.com/user-attachments/assets/3532fbbd-8b29-4206-abcb-797c82d39ba4">

vs predict and predict proba "as of" in a rollforward approach

<img width="440" alt="image" src="https://github.com/user-attachments/assets/a8945f58-5e4e-4abd-8a75-cfeb7a0fc1e7">


# Functions for train/test splits & cross validation
# We can output summary tables for key forecasting metrics (i.e., MAPE/MAE/...)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import *
import pandas as pd
from time_series.forecasting import normal_naive_models as nm
from time_series.forecasting import more_models as mm


def cross_val(df:pd.DataFrame,target_col:str, n_splits:int=5, test_size:int=None, model = None, eval_metric=mean_squared_error, **modelkwargs):
    """
    Cross validation (k-fold) for time series data.

    Inputs:
        df: pandas.DataFrame - Univariate time series dataset.
        target_col:str - Column with historical data
        n_splits: int - Number of folds.
        test_size: int -  Forecast horizon during each fold.
        model -  Valid model function
        eval_metric - metric from sklearn.metrics to evaluate the forecast
        **modelkwargs - keyword arguments for the model chosen

    Outputs:
        Cross validation summary (pandas.DataFrame)
    """

    if model is None:
        #cross validate all forecasting models

        cv = pd.DataFrame()

        #creatinf a column for each method to evaluate
        cv['naive'] = cross_val(df,target_col, n_splits, test_size, nm.naive_pi, **modelkwargs)['error']
        cv['drift'] = cross_val(df,target_col, n_splits, test_size, nm.drift_pi, **modelkwargs)['error']
        cv['mean'] = cross_val(df,target_col, n_splits, test_size, nm.mean_pi, **modelkwargs)['error']
        cv['ETS'] = cross_val(df,target_col, n_splits, test_size, mm.ETS_forecast, **modelkwargs)['error']
        cv['ARIMA'] = cross_val(df,target_col, n_splits, test_size, mm.ARIMA_forecast, **modelkwargs)['error']

        cv = cv.dropna()

        #evaluating each of the 5 methods against the observed data
        obs_data = df[target_col][cv.index]
        obs_data = pd.concat([obs_data] * 5, axis=1)

        eval_list = eval_metric(y_true = obs_data, y_pred = cv, multioutput = 'raw_values')
        keys = ['naive', 'drift', 'mean', 'ETS', 'ARIMA']

        eval = dict(zip(keys, eval_list))

        min_method = min(eval)

        return min_method

    else:

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        cross_val_idx = tscv.split(df[target_col])

        cv_summary = []
        for fold, (train, test) in enumerate(cross_val_idx):
            cv_output = df[target_col].copy().iloc[test].to_frame()

            # **Need to ensure model inputs are consistent across different models**
            forecast = model(df.copy().iloc[train],target_col, len(test), **modelkwargs)

            cv_output['forecast'] = forecast['forecast']
            cv_output['fold'] = fold
            cv_output['error'] = cv_output['forecast'] - cv_output[target_col] 
            cv_summary.append(cv_output)

        return pd.concat(cv_summary)

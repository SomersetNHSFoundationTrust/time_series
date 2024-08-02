# Functions for train/test splits & cross validation
# We can output summary tables for key forecasting metrics (i.e., MAPE/MAE/...)

from .models import prophet_forecast
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd


def cross_val(data, n_splits, test_size, model):
    """
    Cross validation (k-fold) for time series data.

    Inputs:
        data (pandas.DataFrame): Univariate time series dataset.
        n_splits (int): Number of folds.
        test_size (int): Forecast horizon during each fold.
        model: Valid model function

    Outputs:
        Cross validation summary (pandas.DataFrame)
    """

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    cross_val_idx = tscv.split(data)

    cv_summary = {}
    for fold, (train, test) in enumerate(cross_val_idx):
        cv_output = data.copy().iloc[test].reset_index(drop=True)

        # **Need to ensure model inputs are consistent across different models**
        forecast = model(data.copy().iloc[train].reset_index(drop=True), len(test))

        cv_output['forecast'] = forecast.yhat.values
        cv_output['fold'] = fold
        cv_summary[fold] = cv_output

    return pd.concat(cv_summary)

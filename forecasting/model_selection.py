# Functions for train/test splits & cross validation
# We can output summary tables for key forecasting metrics (i.e., MAPE/MAE/...)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import *
import pandas as pd
from .more_models import *

#metrics to cross validate different forecasting methods
eval_metrics = [mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, max_error]

    
def cross_val(df:pd.DataFrame, target_col:str, n_splits:int=5, test_size:int=None, model:dict = model_dict) -> dict[str,pd.DataFrame]:
    """
    Test forecasting method/s on observed data

    Inputs:
        :param df: pandas.DataFrame - Univariate time series dataset.
        :param target_col:str - Column with historical data
        :param n_splits: int - Number of folds.
        :param test_size: int -  Forecast horizon during each fold.
        :param model: dict - a dictionary with the models (str) as keys and their respective functions as values


    Outputs:
        Cross validation summary (pandas.DataFrame)
    """

    #defining a dictionary which will be the output of the cross validation
    output_dict = {}

    #iterating through the model dictionary
    for method in model:

        #splitting the time series to test
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        cross_val_idx = tscv.split(df[target_col])

        #creating a list to store each fold
        cv_summary = []

        #defining the model to test
        forecaster = model[method]

        for fold, (train, test) in enumerate(cross_val_idx):
            cv_output = df[target_col].copy().iloc[test].to_frame()

            #forecasting from training data
            forecast = forecaster(df.copy().iloc[train],target_col, len(test))

            cv_output['forecast'] = forecast['forecast'].values
            cv_output['fold'] = fold
            cv_output['error'] = cv_output['forecast'] - cv_output[target_col] 
            cv_summary.append(cv_output)

        output_dict[method] = pd.concat(cv_summary)

    return output_dict

def forecast_metrics(df:pd.DataFrame,target_col:str, n_splits:int=5, test_size:int=None, model:dict = model_dict, **modelkwargs) -> pd.DataFrame:
    """
    Cross validation (k-fold) for time series data.

    Inputs:
        :param df: pandas.DataFrame - Univariate time series dataset.
        :param target_col: str - Column with historical data
        :param n_splits: int - Number of folds.
        :param test_size: int -  Forecast horizon during each fold (less that len(df) / n_splits)
        :param model: dict - a dictionary with the models (str) as keys and their respective functions as values
        **modelkwargs - keyword arguments for the model chosen

    Outputs:
        Cross validation summary (pandas.DataFrame)
    """

    #cross validate all forecasting models
    train_test_dict = cross_val(df,target_col, n_splits, test_size, model)

    #creating a dataframe to store the test data from train_test
    cv = pd.DataFrame()

    #creating a dataframe to store the matric scores for each method
    
    output_frame = pd.DataFrame(index = ['mean_absolute_error', 'mean_absolute_error', 'mean_squared_error', 'max_error'])

    #evaluating each of the methods against 5 copies of the the observed data
    for method in model:
        cv[method] = train_test_dict[method]['forecast']

        obs_data = df[target_col][cv.index]

        eval_list = []

        for metric in eval_metrics:

            #creating a list of the metric score for each method
            eval = metric(y_true = obs_data, y_pred = cv[method])

            eval_list.append(eval)

        #adding this as a column to output_frame
        output_frame[method] = eval_list


    return output_frame.transpose()



def auto_forecast(df:pd.DataFrame, target_col:str, horizon:int, period:int=1, 
                  pred_width:list = [95,80], models:dict = model_dict,
                  n_splits:int = 5, test_size = None,
                  eval_metric:str = 'mean_squared_error', **kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - Seasonal period
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
        :param models: str - a dict of models and their respective functions to cross validate
        :param n_splits: int - Number of folds.
        :param test_size: int -  Forecast horizon during each fold (less that len(df) / n_splits)
        :param eval_metric: - one of {'mean_absolute_error', 'mean_absolute_error' 'mean_squared_error' 'max_error'} to cross validate the methods in models
        :param **kwargs - Keyword arguments for the sktime functions AutoETS or AutoARIMA
    Output:
        pandas.DataFrame: a cross validated forcast and prediction interval that minimises the evaluation metric specified
    """

    metric_frame = forecast_metrics(df, target_col,n_splits, test_size, models)

    min_method = metric_frame[eval_metric].idxmin()

    output_forecast = benchmark_forecast(df,target_col, min_method,horizon,period,pred_width=pred_width)

    output_forecast = output_forecast.rename(columns={'forecast':f'{min_method} forecast'})

    return output_forecast

    



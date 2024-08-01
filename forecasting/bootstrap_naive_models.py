import pandas as pd
import numpy as np
import random



# **************************************
# Naive Models - Standard implementation
# **************************************


def naive_method(df:pd.DataFrame, target_col:str, horizon:int, period:int=1) -> list:
    """
    Inputs:
        :param df: pandas.DataFrame -  Historical time series data with date-time index
        :param target_col: str - Column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :prarm period: int - Seasonal period
    Outputs:
        list: Forecasted time series with naive method
    """
    most_recent_value = df[target_col].iloc[-period:].tolist()

    mult_list = int(np.ceil(horizon / period))
    return (most_recent_value * mult_list)[:horizon]



def drift_method(df:pd.DataFrame, target_col:str, horizon:int) -> list:
    """
    Inputs:
        :param df: pandas.DataFrame -  Historical time series data with date-time index
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
    Outputs:
        list: Forecasted time series with drift method
    """
    latest_obs = df[target_col].iloc[-1]
    first_obs = df[target_col].iloc[0]

    slope = (latest_obs - first_obs) / (len(df) - 1)

    forecast_list = [latest_obs + slope * h for h in range(1, horizon + 1)]

    return forecast_list


def mean_method(df:pd.DataFrame,target_col:str,horizon:int) -> list:
    """
    Inputs:
        :param df: pandas.DataFrame -  Historical time series data with date-time index
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
    Outputs:
        list: Forecasted time series with mean method
    """

    mean = np.mean(df[target_col])

    return [mean] * horizon

# *****************
# Utility functions
# *****************


def forecast_dates(df:pd.DataFrame, horizon:int) -> pd.DataFrame :
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param horizon: int - Number of timesteps forecasted into the future
    Ouputs:
        pandas.DataFrame: A data frame with dates continued from df to the forecast horizon
    """

    ds = pd.to_datetime(df.index)
    forecast_ds = pd.date_range(start = ds[-1], periods = horizon+1, freq = ds.freq)
    
    return pd.DataFrame(index=forecast_ds[1:])



def naive_error(df:pd.DataFrame,target_col:str, period:int = 1) -> pd.DataFrame:

    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index
        :param target_col: str - column with historical data
        :param period: int - number of periods to shift the error by (seasonal period)
    Ouputs:
        pandas.DataFrame: dataframe with errors from a one step naive forecast and the fitted forecast
    """

    error = pd.DataFrame(index = df.index)

    error['fitted forecast'] = df[target_col].shift(period) 
    error['error'] = df[target_col] - error['fitted forecast']

    return error.dropna()



def drift_error(df:pd.DataFrame,target_col:str) -> pd.DataFrame:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index
        :param target_col: str - column with historical data
    Ouputs:
        pandas.DataFrame: dataframe with errors from drift one-step forecasts and the fitted forecast
    """

    fitted_values = [np.nan]

    for i in range(1,len(df)):
        forecast = drift_method(df.iloc[:i],target_col,horizon=1)
        fitted_values.append(forecast[0])
    
    error = pd.DataFrame(index = df.index)
    error['fitted forecast'] = fitted_values
    error['error'] = df[target_col] - error['fitted forecast']

    return error.dropna()



def mean_error(df:pd.DataFrame,target_col:str) -> pd.DataFrame:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index
        :param target_col: str - column with historical data
    Ouputs:
        pandas.DataFrame: dataframe with errors from mean one-step forecasts and the fitted forecast
    """

    fitted_values = [np.nan]

    for i in range(1,len(df)):
        forecast = mean_method(df.iloc[:i],target_col,horizon=1)
        fitted_values.append(forecast[0])

    error = pd.DataFrame(index = df.index)
    error['fitted forecast'] = fitted_values
    error['error'] = df[target_col] - error['fitted forecast']

    return error.dropna()


def bs_forecast_naive(df: pd.DataFrame, target_col:str, horizon: int, one_step_fcst_errors: pd.Series, period=1) -> list:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param one_step_fcsts: pd.Series - The errors of one-step forecasts to data to randomly sample from (without nans)
    Ouputs:
        list: a bootstrapped forecast by randomly sampling from the errors with naive_error()
    """

    # using the last entry in df to start the sampling
    forecast_list = [df[target_col].iloc[-x] + random.choice(one_step_fcst_errors) for x in range(period)]

    for _ in range(period, horizon):
        sample = forecast_list[-period] + random.choice(one_step_fcst_errors)
        forecast_list.append(sample)

    return forecast_list

def bs_forecast_drift(df: pd.DataFrame,target_col:str, horizon: int, one_step_fcst_errors: pd.Series) -> list:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param one_step_fcsts: pd.Series - The errors of one-step forecasts to data to randomly sample from (without nans)
    Ouputs:
        list: a bootstrapped forecast by randomly sampling from the errors
    """

    # using the last entry in df to start the sampling

    latest_obs = df[target_col].iloc[-1]
    first_obs = df[target_col].iloc[0]

    slope = (latest_obs - first_obs) / (len(df) - 1)

    initial_forecast = latest_obs + slope

    forecast_list = [initial_forecast + random.choice(one_step_fcst_errors)]

    for _ in range(1, horizon):
        latest_obs = forecast_list[-1]

        slope = (latest_obs - first_obs) / (len(df) + len(forecast_list) - 1)

        forecast = latest_obs + slope

        sample = forecast + random.choice(one_step_fcst_errors)

        forecast_list.append(sample)

    return forecast_list

def bs_forecast_mean(df: pd.DataFrame,target_col:str, horizon: int, one_step_fcst_errors: pd.Series) -> list:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param one_step_fcsts: pd.Series - The errors of one-step forecasts to data to randomly sample from (without nans)
    Ouputs:
        list: a bootstrapped forecast by randomly sampling from the errors
    """

    # using the mean of the observed data to start the forecast
    c = np.mean(df[target_col])
    forecast_list = [c + random.choice(one_step_fcst_errors)]

    for _ in range(1, horizon):

        c = np.mean(df[target_col].tolist() + forecast_list)

        sample = c + random.choice(one_step_fcst_errors)
        forecast_list.append(sample)

    return forecast_list

def bs_output(forecast_df:pd.DataFrame, pred_width:list = [95,80]) -> pd.DataFrame :
    """
    Inputs:
        :param forecast_df: pd.DataFrame - Data frame of simulated forecasts to calculate mean and prediction intervals from
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
    Outputs:
        pd.DataFrame: Data frame with forecast dates as the index and the mean, the lower and upper bounds for the prediction interval
    """

    #storing the mean and quantiles for each forecast point in columns
    output_forecast = pd.DataFrame(forecast_df.mean(axis=1), columns=['forecast'])

    for width in pred_width:
        new_pred_width = (100 - (100-width)/2) / 100 

        output_forecast[f'{width}% lower_pi'] = forecast_df.quantile(1 - new_pred_width, axis=1)
        output_forecast[f'{width}% upper_pi'] = forecast_df.quantile(new_pred_width, axis=1)

    return output_forecast




# ******************************
# Naive forecasts with bs p.i's
# ******************************

def bs_naive_pi(df: pd.DataFrame, target_col:str, horizon: int, period: int=1, repetitions:int=100,
             pred_width:list=[95,80], simulations:bool=False) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - Seasonal period
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: float - 0 <= pred_width < 100 list of widths of prediction intervals
        :param simulations: bool - Toggle whether to additionally return the simulations
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    forecast_df = forecast_dates(df,horizon)
    naive_errors = naive_error(df,target_col)['error']

    for run in range(repetitions):
        forecast_df[f'run_{run}'] = bs_forecast_naive(df,target_col, horizon, naive_errors, period)

    output_forecast = bs_output(forecast_df, pred_width)

    if simulations:

        return output_forecast, forecast_df
    
    else:

        return output_forecast


def bs_drift_pi(df: pd.DataFrame,target_col:str, horizon: int, repetitions: int = 100,
             pred_width=95.0,simulations:bool=False) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param bootstrap: bool - toggle bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: float - 0 <= pred_width < 100 width of prediction interval
        :param simulations: bool - Toggle whether to additionally return the simulations
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    forecast_df = forecast_dates(df, horizon)

    drift_errors = drift_error(df,target_col)['error']

    for run in range(repetitions):
        forecast_df[f'run_{run}'] = bs_forecast_drift(df,target_col, horizon, drift_errors)

    output_forecast = bs_output(forecast_df, pred_width)

    if simulations:

        return output_forecast, forecast_df
    
    else:
        return output_forecast


def bs_mean_pi(df: pd.DataFrame,target_col:str, horizon=int, repetitions: int = 100,
            pred_width=95.0,simulations:bool=False) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param bootstrap: bool - toggle bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: float - 0 <= pred_width < 100 width of prediction interval
        :param simulations: bool - Toggle whether to additionally return the simulations
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    forecast_df = forecast_dates(df, horizon)

    mean_errors = mean_error(df,target_col)['error']

    for run in range(repetitions):
        forecast_df[f'run_{run}'] = bs_forecast_mean(df,target_col, horizon, mean_errors)

    output_forecast = bs_output(forecast_df, pred_width)

    if simulations:

        return output_forecast, forecast_df
    
    else:
        return output_forecast

def bs_benchmark_forecast(df:pd.DataFrame, target_col:str, method:str, horizon:int, period:int=1,
                       repetitions:int=100, pred_width:list = [95,80], simulations:bool = False) -> pd.DataFrame:
    
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param method: str - one of {'naive','drift','mean'}, the method to simulate the forecast
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - Seasonal period
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
        :param simulations: bool - Toggle whether to additionally return the simulations
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    if method == 'naive':
        return bs_naive_pi(df, target_col, horizon, period, repetitions, pred_width,simulations)
    
    elif method == 'drift':
        return bs_drift_pi(df,target_col, horizon, repetitions, pred_width,simulations)
    
    elif method == 'mean':
        return bs_mean_pi(df,target_col, horizon, repetitions, pred_width,simulations)
    





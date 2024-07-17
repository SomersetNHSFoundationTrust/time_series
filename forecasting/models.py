# What forecasting models shall we include? What libraries will we need to use these models?

# Example
import pandas as pd
import numpy as np
from prophet import Prophet

def prophet_forecast(df, horizon, **kwargs):
    
    """
    :param df: Pandas.DataFrame - historical time series data.
    :param horizon: int - Number of time steps to forecast.
    :param kwargs: Facebook Prophet keyword arguments.
    :return: Pandas.DataFrame - Forecast.
    """

    model = Prophet(**kwargs)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon)

    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def naive(df, horizon):
    """
    Inputs:
        df (pandas.Series): Historical time series observations (in order)
        horizon (int): Number of timesteps forecasted into the future
    Outputs:
        list: Forecasted time series
    """
    most_recent_value = df.iloc[-1,0]
    return [most_recent_value] * horizon

def s_naive(df, period, horizon):
    """
    Inputs:
        df (pandas.Series): Historical time series observations (in order)
        period (int): Seasonal period (i.e., 12 for monthly data)
        horizon (int): Number of timesteps forecasted into the future
    Outputs:
        list: Forecasted time series
    """

    most_recent_seasonal = df.iloc[-period:,0].to_list()

    # We now need to make the forecast
    # Number of times to multiply the list to ensure we meet forecast horizon
    mult_list = int(np.ceil(horizon / period))

    return (most_recent_seasonal * mult_list)[:horizon]

def drift_method(df, horizon):
    """
    Inputs:
        df (pandas.Series): Historical time series observations (in order)
        horizon (int): Number of timesteps forecasted into the future
    Outputs:
        list: Forecasted time series
    """

    latest_obs = df.iloc[-1,0]
    first_obs = df.iloc[0,0]

    slope = (latest_obs - first_obs) / (len(df) - 1)

    forecast_list = [latest_obs + slope * h for h in range(1, horizon + 1)]

    return forecast_list

def mean_method(df,horizon):
    """
    Inputs:
        df (pandas.Series): Historical time series observations (in order)
        horizon (int): Number of timesteps forecasted into the future
    Outputs:
        list: Forecasted time series
    """
    lastest_obs = df.iloc[-1,0]
    first_obs = df.iloc[0,0]

    mean = (lastest_obs - first_obs) / len(df)

    return [mean] * horizon
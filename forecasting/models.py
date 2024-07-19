# What forecasting models shall we include? What libraries will we need to use these models?

# Example
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.api import ETSModel


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

def auto_params(df,periods=1):
    """
    Inputs:
        df: Historical time series data
    Ouputs:
        list: the automated selection of parameters for simple, double and triple exponential smoothing, alpha, beta and gamma
    """

    if periods == 1:
        ets_model = ETSModel(df[:,0]).fit()
        alpha = ets_model.smoothing_level()
        beta = ets_model.smoothing_trend()
        gamma = ets_model.smoothing_seasonal()
    
    else:
        ets_model = ETSModel(df[:,0],seasonal_periods=period)
        alpha = ets_model.smoothing_level()
        beta = ets_model.smoothing_trend()
        gamma = ets_model.smoothing_seasonal()

    return [alpha, beta, gamma]



def SES_iterator(df,alpha):
    """
    Inputs:
        df (pandas.DataFrame): Historical time series data
        alpha (float): 0<=alpha<=1 the smoothing parameter 
    Outputs:
        list : one step forecast for observed data
    """
    #using thr forst observed value for the first fitted value
    ses_fit = [df.iloc[0,0]]

    #iterating through the data
    for step in range(1,len(df)):
        forecast = alpha * df.iloc[step,0] + (1-alpha) * ses_fit[step-1]
        ses_fit.append(forecast)

    return ses_fit


def SES(df,alpha):
    """
    Inputs:
        df (pandas.DataFrame): Historical time series data
        alpha (float): 0<=alpha<=1 the smoothing parameter 
    Outputs:
        list : one step forecast for observed data
    """
    return SES_iterator(df,alpha)[-1]

def holt_model(df,alpha,beta):
    """
    Inputs:
        df (pandas.DataFrame): Historical time series data
        alpha (float): 0<=alpha<=1 the parameter for the level
        beta (float): 0<=beta<=1 the parameter for the trend
    Outputs:
        list : one step forecast for observed data
    """

    #estimating the first level and trend
    level_component = [df.index[0,0]]
    trend_component = [df.iloc[1,0] - df.iloc[0,0]]
    fitted_model = []

    for y in df.iloc[1:,0]:
        level = alpha * y + (1 - alpha) * (level_component[-1] + trend_component[-1])
        trend = (beta * (level - level_component[-1]) + (1 - beta) * trend_component[-1])

        # Using these values to create a one-step forecast
        forecast = level + trend

        level_component.append(level)
        trend_component.append(trend)
        fitted_model.append(forecast)

    return fitted_model

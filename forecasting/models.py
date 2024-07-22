# What forecasting models shall we include? What libraries will we need to use these models?

# Example
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.api import ETSModel


def prophet_forecast(df:pd.DataFrame, horizon:int, **kwargs) -> pd.DataFrame:
    
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index
        :param horizon: int - Number of time steps to forecast.
        :param kwargs - Facebook Prophet keyword arguments.
        :return: Pandas.DataFrame - Forecast.
    Outputs:
        pd.DataFrame: 
    """

    model = Prophet(**kwargs)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon)

    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def naive(df:pd.DataFrame, horizon:int) -> list:
    """
    Inputs:
        :param df: pandas.DataFrame -  Historical time series data with date-time index
        :param horizon: int - Number of timesteps forecasted into the future
    Outputs:
        list: Forecasted time series with naive method
    """
    most_recent_value = df.iloc[-1,0]
    return [most_recent_value] * horizon

def s_naive(df:pd.DataFrame, period:int, horizon:int) -> list:
    """
    Inputs:
        :param df: pandas.DataFrame -  Historical time series data with date-time index
        :param period: int - Seasonal period
        :param horizon: int - Number of timesteps forecasted into the future
    Outputs:
        list: Forecasted time series with seasonal naive method
    """

    most_recent_seasonal = df.iloc[-period:,0].to_list()

    # We now need to make the forecast
    # Number of times to multiply the list to ensure we meet forecast horizon
    mult_list = int(np.ceil(horizon / period))

    return (most_recent_seasonal * mult_list)[:horizon]

def drift_method(df:pd.DataFrame, horizon:int) -> list:
    """
    Inputs:
        :param df: pandas.DataFrame -  Historical time series data with date-time index
        :param horizon: int - Number of timesteps forecasted into the future
    Outputs:
        list: Forecasted time series with drift method
    """

    latest_obs = df.iloc[-1,0]
    first_obs = df.iloc[0,0]

    slope = (latest_obs - first_obs) / (len(df) - 1)

    forecast_list = [latest_obs + slope * h for h in range(1, horizon + 1)]

    return forecast_list

def mean_method(df:pd.DataFrame,horizon:int) -> list:
    """
    Inputs:
        :param df: pandas.DataFrame -  Historical time series data with date-time index
        :param horizon: int - Number of timesteps forecasted into the future
    Outputs:
        list: Forecasted time series with mean method
    """
   

    mean = sum(df.iloc[:,0]) / len(df)

    return [mean] * horizon

def ETS_forecast(df:pd.DataFrame,target_col = None, period:int=1):

    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: Any - column to forecast
        :param period: int - seasonal period
    Ouputs:
        list: the automated selection of parameters for simple, double and triple exponential smoothing, alpha, beta and gamma
    """

    if target_col is None:

        ets_model = ETSModel(df[:,0],seasonal_periods=period).fit()
        alpha = ets_model.smoothing_level()
        beta = ets_model.smoothing_trend()
        gamma = ets_model.smoothing_seasonal()
    
    else:
    
        ets_model = ETSModel(df[target_col],seasonal_periods=period).fit()
        alpha = ets_model.smoothing_level()
        beta = ets_model.smoothing_trend()
        gamma = ets_model.smoothing_seasonal()

    return [alpha, beta, gamma]



def SES_iterator(df,alpha):
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data
        :param alpha: float - 0<= alpha <=1 the smoothing parameter 
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
        :param df: pandas.DataFrame - Historical time series data
       :param  alpha: float - 0<=alpha<=1 the smoothing parameter 
    Outputs:
        list: one step forecast for observed data
    """
    return SES_iterator(df,alpha)[-1]

def holt_model(df,alpha,beta):
    """
    Inputs:
        :param df pandas.DataFrame - Historical time series data
        :param alpha float - 0<=alpha<=1 the parameter for the level
        :param beta: float - 0<=beta<=1 the parameter for the trend
    Outputs:
        list: one step forecast for observed data
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

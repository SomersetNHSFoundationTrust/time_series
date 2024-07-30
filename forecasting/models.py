# What forecasting models shall we include? What libraries will we need to use these models?

# Example

import pandas as pd
import numpy as np
from prophet import Prophet
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS


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



def naive(df:pd.DataFrame, target_col:str, horizon:int, period:int=1) -> list:
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


def ETS_forecast(df:pd.DataFrame,target_col:str, horizon:int, period:int,pred_width:float=95,**AutoETS_kwargs) -> pd.DataFrame:

    #add pred_width
    #fix dates

    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column to forecast
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - seasonal period
        :param pred_width: float - 0 <= pred_width < 100 width of prediction interval
        :param **AutoETS_kwargs - keyword arguments for the sktime AutoETS() function
    Ouputs:
        pd.DataFrame: forecast with exponential smmothing with auto-selected parameters
    """

    df.index=pd.to_datetime(df.index)
    df = df.asfreq(df.index.freq)

    fh = [i for i in range(1,horizon+1)]

    forecaster = AutoETS(sp=period,auto=True,**AutoETS_kwargs).fit(df[target_col])

    forecast = forecaster.predict(fh=fh)
    pred_int= forecaster.predict_interval(fh=fh,coverage=[pred_width/100])

    forecast_ds = pd.date_range(start = df.index[-1], periods = horizon+1, freq = df.index.freq)

    output_forecast = pd.DataFrame(index = forecast_ds[1:])
    output_forecast['forecast'] = forecast
    output_forecast[['lower_pi', 'upper_pi']] = pred_int[target_col,pred_width / 100]

    return output_forecast


   

def ARIMA_forecast(df:pd.DataFrame, target_col:str,horizon:int,period:int = 1,pred_width:float = 95, **AutoARIMA_kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column to forecast
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - seasonal period
        :param pred_width: float - 0 <= pred_width < 100 width of prediction interval
        :param **AutoARIMA_kwargs - keyword arguments for the sktime AutoARIMA() function
    Ouputs:
        pd.DataFrame: forecast with ARIMA with auto-selected parameters
    """

    df.index = pd.to_datetime(df.index)
    df = df.asfreq(df.index.freq)

    fh = [i for i in range(1,horizon+1)]

    forecaster = AutoARIMA(sp = period,**AutoARIMA_kwargs,suppress_warnings=True).fit(df[target_col])


    forecast = forecaster.predict(fh=fh)
    pred_int = forecaster.predict_interval(fh=fh,coverage=pred_width / 100)

    forecast_ds = pd.date_range(start = df.index[-1], periods = horizon+1, freq = df.index.freq)

    output_forecast = pd.DataFrame(index = forecast_ds[1:])
    output_forecast['forecast'] = forecast
    output_forecast[['lower_pi', 'upper_pi']] = pred_int[target_col, pred_width / 100]

    return output_forecast

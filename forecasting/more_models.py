import pandas as pd
from prophet import Prophet
from statsforecast.models import AutoETS, AutoARIMA
from .normal_naive_models import naive_pi, drift_pi, mean_pi, normal_benchmark_forecast
from .bootstrap_naive_models import forecast_dates, bs_benchmark_forecast, naive_error, drift_error, mean_error
from sklearn.metrics import *
from statsmodels.tsa.seasonal import MSTL
#from .model_selection import auto_forecast


def prophet_forecast(df:pd.DataFrame, target_col:str, horizon:int, pred_width:list = [95,80], **kwargs) -> pd.DataFrame:
    
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index
        :param target_col:str - column with historical data
        :param horizon: int - Number of time steps to forecast.
        :param kwargs - Facebook Prophet keyword arguments.
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
    Outputs:
        pd.DataFrame: Data frame with prophet forecast and prediction intervals
    """
    output_forecast = pd.DataFrame()
    #storing the forecast in output_forecast
    data = {'ds':df.index, 'y':df[target_col].values}
    input_df = pd.DataFrame(data)
 
    model = Prophet(**kwargs)
    model.fit(input_df)
    future = model.make_future_dataframe(periods=horizon)

    forecast = model.predict(future)
    output_forecast.index = future['ds'].iloc[-horizon:]
    output_forecast['forecast'] = forecast['yhat'].iloc[-horizon:].values

    #storing each prediction interval
    for width in pred_width:
        model = Prophet(**kwargs,interval_width=width/100)
        model.fit(input_df)
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

        output_forecast[[f'{width}% lower_pi', f'{width}% upper_pi']] = forecast[['yhat_lower','yhat_upper']].iloc[-horizon:].values

    return output_forecast


def ETS_fit(df:pd.DataFrame, target_col:str, period:int=1, **AutoETS_kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column to forecast
        :param period: int - seasonal period
        :param **AutoETS_kwargs - keyword arguments for the statsforecast AutoETS() function
    Ouputs:
        pd.DataFrame: fitted exoonential smoothing forecast
    """

    #using AutoETS to forecast
    forecaster = AutoETS(season_length=period,**AutoETS_kwargs)
    forecaster.fit(df[target_col].values)

    #storing the forecast and poutputing the result

    forecast = pd.DataFrame(forecaster.predict_in_sample())
    output_forecast = pd.DataFrame(index = df.index)
    output_forecast['fitted forecast'] = forecast['fitted'].values

    return output_forecast

     



def ETS_forecast(df:pd.DataFrame,target_col:str, horizon:int, period:int=1,pred_width:list=[95,80],**AutoETS_kwargs) -> pd.DataFrame:

    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column to forecast
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - seasonal period
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
        :param **AutoETS_kwargs - keyword arguments for the statsforecast AutoETS() function
    Ouputs:
        pd.DataFrame: forecast with exponential smmothing with auto-selected parameters
    """

    #using AutoETS to forecast
    forecaster = AutoETS(season_length=period,**AutoETS_kwargs)
    forecaster.fit(df[target_col].values)

    #storing the forecast and prediction intervals
    forecast = pd.DataFrame(forecaster.predict(h=horizon,level=pred_width))

    #creating the output dataframe, renaming the columns and setting the index to the extended dates
    output_forecast = forecast_dates(df,horizon)
    output_forecast['forecast'] = forecast['mean'].values

    for width in pred_width:

        output_forecast[[f'{width}% lower_pi', f'{width}% upper_pi']] = forecast[[f'lo-{width}',f'hi-{width}']].values

    return output_forecast



def ARIMA_fit(df:pd.DataFrame, target_col:str, period:int=1, **AutoARIMA_kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column to forecast
        :param period: int - seasonal period
        :param **AutoETS_kwargs - keyword arguments for the statsforecast AutoETS() function
    Ouputs:
        pd.DataFrame: fitted exoonential smoothing forecast
    """

    #using AutoARIMA to forecast
    forecaster = AutoARIMA(season_length=period,**AutoARIMA_kwargs)
    forecaster.fit(df[target_col].values)

    #storing the forecast and poutputing the result

    forecast = pd.DataFrame(forecaster.predict_in_sample())
    output_forecast = pd.DataFrame(index = df.index)
    output_forecast['fitted forecast'] = forecast['fitted'].values

    return output_forecast



   

def ARIMA_forecast(df:pd.DataFrame, target_col:str,horizon:int,period:int = 1,pred_width:list = [95,80], **AutoARIMA_kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column to forecast
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - seasonal period
        :param pred_width: float - 0 <= pred_width < 100  list of widths of prediction intervals
        :param **AutoARIMA_kwargs - keyword arguments for the statsforecast AutoARIMA() function
    Ouputs:
        pd.DataFrame: ARIMA forecast with auto-selected parameters
    """

   #using AutoARIMA to forecast
    forecaster = AutoARIMA(season_length=period,**AutoARIMA_kwargs)
    forecaster.fit(df[target_col].values)

    #storing the forecast and prediction intervals
    forecast = pd.DataFrame(forecaster.predict(h=horizon,level=pred_width))

    #creating the output dataframe, renaming the columns and setting the index to the extended dates
    output_forecast = forecast_dates(df,horizon)
    output_forecast['forecast'] = forecast['mean'].values

    for width in pred_width:

        output_forecast[[f'{width}% lower_pi', f'{width}% upper_pi']] = forecast[[f'lo-{width}',f'hi-{width}']].values

    return output_forecast

def MSTL_forecast(df:pd.DataFrame, target_col:str,horizon:int,
                  period:list[int],pred_width:list[float] = [95,80],
                  trend_forecaster = ETS_forecast ,**model_kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column to forecast
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - seasonal periods in a list
        :param pred_width: float - 0 <= pred_width < 100  list of widths of prediction intervals
        :param trend_forecaster: the model (key from model_dict) used to forecast the trend,
                                 if left none it will use autoforecast to minimise the chosen evaluation metric
        :param **model_kwargs - keyword arguments for trend_forecaster
    Ouputs:
        pd.DataFrame: MSTL forecast for the trend
    """
    #decomposing df and finding the trend
    mstl = MSTL(df[target_col],periods=period).fit()
    trend = pd.DataFrame(mstl.trend.values,index=df.index,columns=['trend'])

    output_forecast = trend_forecaster(trend, 'trend', horizon,pred_width = pred_width,**model_kwargs)
        
    return output_forecast
        


model_dict = {'naive':naive_pi, 'drift':drift_pi, 'mean':mean_pi,
              'ETS':ETS_forecast, 'ARIMA':ARIMA_forecast,
              'prophet':prophet_forecast, 'MSTL':MSTL_forecast}



def benchmark_forecast(df:pd.DataFrame, target_col:str, method:str, horizon:int, period:int=1, 
                       bootstrap=False, repetitions:int=100, pred_width:list = [95,80], **kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param method: str - one of the keys from model_dict
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - Seasonal period
        :param bootstrap: bool - Toggle whether to simulate forecast and prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
        :param **kwargs - Keyword arguments for the sktime functions AutoETS or AutoARIMA
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """


    if bootstrap:

        forecast = bs_benchmark_forecast(df,target_col, method, horizon,period,repetitions,pred_width)
    
    else:

        if method == 'naive' or method == 'drift' or method == 'mean':

            forecast = normal_benchmark_forecast(df,target_col, method, horizon, period, pred_width)

        elif method == 'ETS':

            forecast = ETS_forecast(df, target_col, horizon, period, pred_width, **kwargs)

        elif method == 'ARIMA':

            forecast = ARIMA_forecast(df,target_col, horizon, period, pred_width, **kwargs)


    return forecast

def benchmark_fit(df:pd.DataFrame, target_col:str, method:str, period:int=1, **kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param method: str - one of {'naive','drift','mean','ETS','ARIMA'}, the method to simulate the forecast
        :param period: int - Seasonal period
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
        :param **kwargs - Keyword arguments for the sktime functions AutoETS or AutoARIMA
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    fitted_forecast = pd.DataFrame()

    if method == 'naive':

        fitted_forecast = pd.DataFrame(index = df.index)
        fitted_forecast['fitted forecast'] = naive_error(df, target_col, period)['fitted forecast']

    elif method == 'drift':

        fitted_forecast = pd.DataFrame(index = df.index)
        fitted_forecast['fitted forecast'] = drift_error(df,target_col)['fitted forecast']

    elif method == 'mean':

        fitted_forecast = pd.DataFrame(index = df.index)
        fitted_forecast['fitted forecast'] = mean_error(df,target_col)['fitted forecast']

    elif method == 'ETS':

        fitted_forecast = ETS_fit(df, target_col, period, **kwargs)

    elif method == 'ARIMA':

        fitted_forecast = ARIMA_fit(df,target_col, period, **kwargs)

    return fitted_forecast




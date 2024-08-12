import pandas as pd
from prophet import Prophet
from statsforecast.models import AutoETS, AutoARIMA
from .normal_naive_models import naive_pi, drift_pi, mean_pi
from .bootstrap_naive_models import forecast_dates, naive_error, drift_error, mean_error
from sklearn.metrics import *
from statsmodels.tsa.seasonal import MSTL
#from .model_selection import auto_forecast


def prophet_fit(df:pd.DataFrame, target_col:str,**kwargs) -> pd.DataFrame:
    
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with historical data.
        :param kwargs - Facebook Prophet keyword arguments.
    Outputs:
        pandas.DataFrame: Facebook Prophet fitted forecast for df.
    """

    data = {'ds':df.index, 'y':df[target_col].values}
    input_df = pd.DataFrame(data)

    model = Prophet(**kwargs)
    model.fit(input_df)
    forecast = model.predict()

    fitted_forecast = pd.DataFrame(forecast['yhat'].values, index = df.index, columns = ['fitted forecast'])

    return fitted_forecast

    
def prophet_forecast(df:pd.DataFrame, target_col:str, horizon:int, period:int = 1, pred_width:list = [95,80], **kwargs) -> pd.DataFrame:
    
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of time steps to forecast.
        :param period: int - Seasonal period.
        :param kwargs - Facebook Prophet keyword arguments.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
    Outputs:
        pandas.DataFrame: Facebook Prophet forecast and prediction intervals for df.
    """

    output_forecast = pd.DataFrame()

    #setting up the input dataframe for Prophet and storing this in output_forecast
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
        model = Prophet(**kwargs, interval_width=width/100)
        model.fit(input_df)
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

        output_forecast[[f'{width}% lower_pi', f'{width}% upper_pi']] = forecast[['yhat_lower','yhat_upper']].iloc[-horizon:].values

    return output_forecast


def ETS_fit(df:pd.DataFrame, target_col:str, period:int=1, **AutoETS_kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column to forecast.
        :param period: int - Seasonal period.
        :param **AutoETS_kwargs - Keyword arguments for the statsforecast AutoETS() class.
    Ouputs:
        pandas.DataFrame: Fitted exponential smoothing forecast for df.
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
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column to forecast.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
        :param **AutoETS_kwargs - Keyword arguments for the statsforecast AutoETS() class.
    Ouputs:
        pandas.DataFrame: ETS forecast with auto-selected parameters and prediction intervals for df.
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
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column to forecast.
        :param period: int - Seasonal period.
        :param **AutoARIMA_kwargs - Keyword arguments for the statsforecast AutoARIMA() class.
    Ouputs:
        pandas.DataFrame: Fitted ARIMA forecast for df.
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
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column to forecast.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
        :param **AutoARIMA_kwargs - Keyword arguments for the statsforecast AutoARIMA() class.
    Ouputs:
        pandas.DataFrame: ARIMA forecast with auto-selected parameters and prediction intervals for df.
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


def MSTL_forecast(df:pd.DataFrame, target_col:str,horizon:int, period=1, pred_width:list = [95,80], trend_forecaster:str = 'ETS' ,**kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column to forecast.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int or list - Seasonal periods in a list.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
        :param trend_forecaster: str - The model (a key from model_dict) used to forecast the trend.
        :param **kwargs - Keyword arguments for trend_forecaster.
    Ouputs:
        pandas.DataFrame: MSTL forecast and prediction intervals for the trend of df.
    """

    #decomposing df and finding the trend
    mstl = MSTL(df[target_col],periods=period).fit()
    trend = pd.DataFrame(mstl.trend.values, index=df.index, columns=['trend'])

    forecaster = model_dict[trend_forecaster]
    output_forecast = forecaster(df = trend,
                                 target_col = 'trend',
                                 horizon = horizon,
                                 pred_width = pred_width,
                                 **kwargs)
        
    return output_forecast
        


model_dict = {'naive':naive_pi, 'drift':drift_pi, 'mean':mean_pi,
              'ETS':ETS_forecast, 'ARIMA':ARIMA_forecast,
              'prophet':prophet_forecast}



def benchmark_forecast(df:pd.DataFrame, target_col:str, horizon:int, model:str, period:int=1, pred_width:list=[], **kwargs) -> pd.DataFrame:
    
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param model: str - The model (a key from model_dict) to forecast the trend.
        :param period: int - Seasonal period.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals, defaults to none for faster forecasting.
        :param **kwargs - Keyword arguments for the chosen model.
    Output:
        pandas.DataFrame: A forecast for df of the chosen model.
    """

    forecaster = model_dict[model]
    
    output_forecast = forecaster(df=df,
                                 target_col=target_col,
                                 horizon = horizon,
                                 period=period,
                                 pred_width = pred_width,
                                 **kwargs)
    
    return output_forecast


def benchmark_fit(df:pd.DataFrame, target_col:str, model:str, period:int=1, **kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column with historical data.
        :param model: str - Model to calculate the fitted forecast, one of the keys from model_dict.
        :param period: int - Seasonal period.
        :param **kwargs - Keyword arguments for the sktime functions AutoETS or AutoARIMA.
    Output:
        pandas.DataFrame: A fitted forecast for df of the specified model.
    """

    fitted_forecast = pd.DataFrame()

    if model == 'naive':

        fitted_forecast = pd.DataFrame(index = df.index)
        fitted_forecast['fitted forecast'] = naive_error(df, target_col, period)['fitted forecast']

    elif model == 'drift':

        fitted_forecast = pd.DataFrame(index = df.index)
        fitted_forecast['fitted forecast'] = drift_error(df,target_col)['fitted forecast']

    elif model == 'mean':

        fitted_forecast = pd.DataFrame(index = df.index)
        fitted_forecast['fitted forecast'] = mean_error(df,target_col)['fitted forecast']

    elif model == 'ETS':

        fitted_forecast = ETS_fit(df, target_col, period, **kwargs)

    elif model == 'ARIMA':

        fitted_forecast = ARIMA_fit(df,target_col, period, **kwargs)

    elif model == 'prophet':
        fitted_forecast = prophet_fit(df,target_col, **kwargs)

    return fitted_forecast




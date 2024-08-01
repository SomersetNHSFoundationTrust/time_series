import pandas as pd
from prophet import Prophet
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from statsmodels.tsa.seasonal import STL
from .normal_naive_models import normal_benchmark_forecast
from .bootstrap_naive_models import bs_benchmark_forecast, naive_error, drift_error, mean_error


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


def ETS_fit(df:pd.DataFrame, target_col:str, period:int=1, **AutoETS_kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column to forecast
        :param period: int - seasonal period
        :param **AutoETS_kwargs - keyword arguments for the sktime AutoETS() function
    Ouputs:
        pd.DataFrame: fitted exoonential smoothing forecast
    """

    #setting up the dates and horizon to use in AutoETS
    df.index=pd.to_datetime(df.index)
    df = df.asfreq(df.index.freq)

    #Using AutoETS to find the fitted forecast and storing the forecast

    forecaster = AutoETS(sp=period, auto=True, **AutoETS_kwargs).fit(df[target_col])

    fh = [i for i in range(0,-len(df),-1)]
    forecast = forecaster.predict(fh=fh)

    #outputting a data frame with a date-time index and the fitted forecast
    output_forecast = pd.DataFrame(index = df.index)
    output_forecast['fitted forecast'] = forecast

    return output_forecast

     



def ETS_forecast(df:pd.DataFrame,target_col:str, horizon:int, period:int=1,pred_width:list=[95,80],**AutoETS_kwargs) -> pd.DataFrame:

    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column to forecast
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - seasonal period
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
        :param **AutoETS_kwargs - keyword arguments for the sktime AutoETS() function
    Ouputs:
        pd.DataFrame: forecast with exponential smmothing with auto-selected parameters
    """

    #setting up the dates and horizon to use in AutoETS
    df.index=pd.to_datetime(df.index)
    df = df.asfreq(df.index.freq)

    #Using AutoETS to forecast and storing the forecast
    forecaster = AutoETS(sp=period, auto=True, **AutoETS_kwargs).fit(df[target_col])

    fh = [i for i in range(1,horizon+1)]
    forecast = forecaster.predict(fh=fh)

    #setting up the width values and storing the prediction interval
    new_pred_width = [width/100 for width in pred_width]

    pred_int= forecaster.predict_interval(fh=fh, coverage=new_pred_width)

    #outputting the results in the same form as prediction_intervals
    forecast_ds = pd.date_range(start = df.index[-1], periods = horizon+1, freq = df.index.freq)

    output_forecast = pd.DataFrame(index = forecast_ds[1:])
    output_forecast['forecast'] = forecast

    for width in pred_width:

        output_forecast[[f'{width}% lower_pi', f'{width}% upper_pi']] = pred_int[target_col, width / 100]

    return output_forecast



def ARIMA_fit(df:pd.DataFrame, target_col:str, period:int=1, **AutoARIMA_kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column to forecast
        :param period: int - seasonal period
        :param **AutoETS_kwargs - keyword arguments for the sktime AutoETS() function
    Ouputs:
        pd.DataFrame: fitted exoonential smoothing forecast
    """

    df.index=pd.to_datetime(df.index)
    df = df.asfreq(df.index.freq)

    #Using AutoARIMA to find the fitted forecast and storing the forecast

    forecaster = AutoARIMA(sp=period, **AutoARIMA_kwargs).fit(df[target_col])

    fh = [i for i in range(0,-len(df),-1)]
    forecast = forecaster.predict(fh=fh)

    #outputting a data frame with a date-time index and the fitted forecast
    output_forecast = pd.DataFrame(index = df.index)
    output_forecast['fitted forecast'] = forecast

    return output_forecast


   

def ARIMA_forecast(df:pd.DataFrame, target_col:str,horizon:int,period:int = 1,pred_width:list = [95,80], **AutoARIMA_kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column to forecast
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - seasonal period
        :param pred_width: float - 0 <= pred_width < 100  list of widths of prediction intervals
        :param **AutoARIMA_kwargs - keyword arguments for the sktime AutoARIMA() function
    Ouputs:
        pd.DataFrame: forecast with ARIMA with auto-selected parameters
    """

    #setting up the dates and horizon to use in AutoETS
    df.index = pd.to_datetime(df.index)
    df = df.asfreq(df.index.freq)

    fh = [i for i in range(1,horizon+1)]

    #Using AutoETS to forecast and storing the forecast
    forecaster = AutoARIMA(sp = period,**AutoARIMA_kwargs,suppress_warnings=True).fit(df[target_col])

    forecast = forecaster.predict(fh=fh)

    #setting up the width values and storing the prediction interval
    new_pred_width = [width / 100 for width in pred_width]

    pred_int = forecaster.predict_interval(fh=fh,coverage=new_pred_width)

    #outputting the results in the same form as prediction_intervals
    forecast_ds = pd.date_range(start = df.index[-1], periods = horizon+1, freq = df.index.freq)

    output_forecast = pd.DataFrame(index = forecast_ds[1:])
    output_forecast['forecast'] = forecast

    for width in pred_width:

        output_forecast[[f'{width}% lower_pi', f'{width}% upper_pi']] = pred_int[target_col, width / 100]

    return output_forecast



def STL_forecast(df:pd.DataFrame,target_col:str, method:str, horizon:int,
                 period:int=1, bootstrap:bool = False, repetitions:int=100,
                 pred_width:list = [95,80], **STLkwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param method: str - one of {'naive','drift','mean'}, the method to simulate the forecast
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - seasonal period
        :param bootstrap: bool - Toggle whether to simulate forecast and prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
        :param **STLkwargs: keyword arguments for the statsmodels function STL()
    Output:
        pandas.DataFrame: a forecast for the trend using the method specified
    """

    stl = STL(df[target_col], period=period, **STLkwargs).fit()
    trend = stl.trend

    trend_df = pd.DataFrame(index = trend.index)
    trend_df['data'] = trend.values

    forecast = pd.DataFrame()

    if bootstrap:

        forecast = bs_benchmark_forecast(trend,'data', method, horizon, period, repetitions, pred_width)

    else:

        forecast = normal_benchmark_forecast(trend,'data', method, horizon, period, pred_width)
    
    return forecast


def benchmark_forecast(df:pd.DataFrame, target_col:str, method:str, horizon:int, period:int=1, 
                       bootstrap=False, repetitions:int=100, pred_width:list = [95,80], **kwargs) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param method: str - one of {'naive','drift','mean','ETS','ARIMA'}, the method to simulate the forecast
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

        forecast = bs_benchmark_forecast(df,target_col, method,horizon,period,repetitions,pred_width)
    
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




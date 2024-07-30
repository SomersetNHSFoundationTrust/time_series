import pandas as pd
import numpy as np
from .models import naive, drift_method, mean_method
import random
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL




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



def naive_error(df:pd.DataFrame,target_col:str, period:int = 1) -> pd.Series:

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

    return error



def drift_error(df:pd.DataFrame,target_col:str) -> pd.Series:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index
        :param target_col: str - column with historical data
    Ouputs:
        pandas.DataFrame: dataframe with errors from drift one-step forecasts and the fitted forecast
    """

    fitted_values = [np.nan, np.nan]

    for i in range(2,len(df)):
        forecast = drift_method(df.iloc[:i],'y',horizon=1)
        fitted_values.append(forecast[0])
    
    error = pd.DataFrame(index = df.index)
    error['fitted forecast'] = fitted_values
    error['error'] = error['fitted forecast'] - df[target_col]

    return error



def mean_error(df:pd.DataFrame,target_col:str) -> pd.Series:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index
        :param target_col: str - column with historical data
    Ouputs:
        pandas.DataFrame: dataframe with errors from mean one-step forecasts and the fitted forecast
    """

    fitted_values = [np.nan]

    for i in range(1,len(df)):
        forecast = mean_method(df.iloc[:i],'y',horizon=1)
        fitted_values.append(forecast[0])

    error = pd.DataFrame(index = df.index)
    error['fitted forecast'] = fitted_values
    error['error'] = error['fitted forecast'] - df[target_col]

    return error





def bs_forecast(df:pd.DataFrame,target_col:str,horizon:int,one_step_fcst_errors:pd.Series) -> list:

    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param one_step_fcsts: pd.Series - The errors of one-step forecasts to data to randomly sample from
    Ouputs:
        list: a bootstrapped forecast by randomly sampling from the errors
    """

    #using the last entry in df to start the sampling
    forecast_list = [ df[target_col].iloc[-1] + random.choice(one_step_fcst_errors) ]

    for _ in range(1,horizon):

        sample = forecast_list[-1] + random.choice(one_step_fcst_errors)
        forecast_list.append(sample)

    return forecast_list



def bs_output(forecast_df:pd.DataFrame, pred_width:list = [95.0]) -> pd.DataFrame :
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



def pi_output(forecast_df:pd.DataFrame, horizon:int, forecast_sd:list, pred_width:list = [95.0]) -> pd.DataFrame:
    """
    Inputs:
        :param forecast_df: pd.DataFrame - Data frame with extended dates and forecasted points
        :param horizon: int - Number of timesteps forecasted into the future
        :param forecast_sd: list - Multi-step standard deviation for each forecasted point
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
    Outputs:
        pd.DataFrame: Data frame with forecast dates as the index and the mean, the lower and upper bounds for the prediction interval

    """

    #putting the widths in reverse order for graphing
    pred_width = np.sort(pred_width)
    pred_width = reversed(pred_width)

    output_forecast = forecast_df

    for width in pred_width:
        new_pred_width = (100 - (100 - width) / 2) / 100 

        #calculating the multiplier for the forecast standard deviation depending on the prediction width
        pi_mult = norm.ppf(new_pred_width)

        output_forecast[f'{width}% lower_pi'] = [forecast_df['forecast'].iloc[i] - pi_mult * forecast_sd[i] for i in range(horizon)]
        output_forecast[f'{width}% upper_pi'] = [forecast_df['forecast'].iloc[i] + pi_mult * forecast_sd[i] for i in range(horizon)]

    return output_forecast





def naive_pi(df:pd.DataFrame, target_col:str, horizon:int, bootstrap=False, repetitions=100, pred_width:list = [95.0]) -> pd.DataFrame:
   
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with dates as index
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param bootstrap: bool - bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    #extending the dates from df and calculating the errors from the fitted forecast
    forecast_df = forecast_dates(df,horizon)
    errors = naive_error(df,target_col)['error']

    if bootstrap:

        ###Regular bootstrap###
        
        #creating repetitions of the bootstrapped forecasts and storing them
        for run in range(repetitions):
            forecast_df[f'run_{run}'] = bs_forecast(df, target_col, horizon, errors) 

        #calculating the mean and quantiles
        output_forecast = bs_output(forecast_df,pred_width)

        return output_forecast
        
    else:

        ###Naive###

        #storing the extended dates and forecasted values in a dataframe
        forecast_df['forecast'] = naive(df,target_col, horizon)

        #calculating the residuals using the forecast (the last observed value)
        sd_resiuals = np.std(errors)
        forecast_sd = [sd_resiuals * np.sqrt(h) for h in range(1,horizon+1)]
        
        #creating an output forecast with the naive forecast and prediction interval
        output_forecast = pi_output(forecast_df,horizon,forecast_sd,pred_width)
        
        return output_forecast
    



def s_naive_pi(df:pd.DataFrame,target_col:str, horizon:int,  period:int, bootstrap=False, repetitions=100, pred_width:list = [95.0]) -> pd.DataFrame:
   
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - Seasonal period
        :param bootstrap: bool - toggle bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """
    #extending the dates from df and calculating the errors from the fitted forecast   
    forecast_df = forecast_dates(df,horizon)
    errors = naive_error(df,target_col,period)['error']


    if bootstrap:
        
        for run in range(repetitions):
            forecast_df[f'run_{run}'] = bs_forecast(df,target_col,horizon,errors)

        output_forecast = bs_output(forecast_df,pred_width)

        return output_forecast
        
    else:

        ###Seasonal naive###

        #calculating the seasonal naive forecast for df
        forecast_df['forecast'] = naive(df,target_col,horizon,period)

        sd_residuals = np.std(errors)

        #using the number of seasons prior to each point to calculate the standard deviation of forecasted points
        seasons_in_forecast = [np.floor((h-1) / period) for h in range(1,horizon+1)] 
        forecast_sd = [sd_residuals * np.sqrt(seasons_in_forecast[h-1]+1) for h in range(1,horizon+1)]

        output_forecast = pi_output(forecast_df,horizon,forecast_sd, pred_width)
        
        return output_forecast



def drift_pi(df:pd.DataFrame,target_col:str,horizon:int,bootstrap:bool = False, repetitions:int = 100, pred_width:list = [95.0]) -> pd.DataFrame:

    
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param bootstrap: bool - toggle bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """
    #extending the dates from df and calculating the errors from the fitted forecast
    forecast_df = forecast_dates(df,horizon)
    errors = drift_error(df,target_col)['error']


    if bootstrap:
        
        for run in range(repetitions):
            forecast_df[f'run_{run}'] = bs_forecast(df,target_col,horizon,errors)
        
        output_forecast = bs_output(forecast_df, pred_width)

        return output_forecast
    else:
        #setting up the forecast dataframe
        forecast_df['forecast'] = drift_method(df,target_col,horizon)


        #calculating the residuals
        sd_residuals = np.std(errors)

        #calculating the standard deviation for the residuals and the forecasted points
        forecast_sd = [sd_residuals * np.sqrt(i * (1 + i/(len(df)-1))) for i in range(1,horizon+1)]

        #outputting the result
        output_forecast = pi_output(forecast_df,horizon,forecast_sd,pred_width)
        
        return output_forecast



def mean_pi(df:pd.DataFrame,target_col:str, horizon=int, bootstrap:bool = False, repetitions:int = 100, pred_width:list = [95.0]) -> pd.DataFrame:

    
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param bootstrap: bool - toggle bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    forecast_df = forecast_dates(df,horizon)
    errors = mean_error(df,target_col)['error']
    
    if bootstrap:

        for run in range(repetitions):
            forecast_df[f'run_{run}'] = bs_forecast(df,target_col,horizon,errors)

        output_forecast = bs_output(forecast_df, pred_width)

        return output_forecast
    
    else:

        #setting up the forecast dataframe
        forecast_df['forecast'] = mean_method(df,target_col,horizon)

        #calculating the standard deviation of the residuals and the forecasted points
        sd_residuls = np.std(errors)
        forecast_sd = [sd_residuls * np.sqrt(1 + 1/len(df))] * horizon

        #outputting the result
        output_forecast = pi_output(forecast_df,horizon,forecast_sd, pred_width)

        return output_forecast



def STL_pi(df:pd.DataFrame,target_col:str, horizon:int, method:str,
           period:int=1, bootstrap:bool = False, repetitions:int = 100, pred_width:list = [95.0]) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param method: str - one of {'naive','seasonal naive','drift','mean'}, the method to simulate the forecast
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - seasonal period
        :param bootstrap: bool - toggle bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
    Output:
        pandas.DataFrame: a forecast for the trend using the method specified
    """

    stl = STL(df[target_col], period=period).fit()
    trend = stl.trend

    if method == 'naive':
        output_forecast = naive_pi(trend,target_col,horizon,bootstrap,repetitions,pred_width)
    
    elif method == 'seasonal_naive':
        output_forecast = s_naive_pi(trend,target_col,horizon,period,bootstrap,repetitions,pred_width)

    elif method == 'drift':
        output_forecast = drift_pi(trend,target_col, horizon,bootstrap,repetitions,pred_width)

    elif method == 'mean':
        output_forecast = mean_pi(trend,target_col,horizon,bootstrap,repetitions,pred_width)

    return output_forecast



def benchmark_forecast(df:pd.DataFrame, target_col:str, method:str, horizon:int, period:int=1,
                       bootstrap:bool=False, repetitions:int=100, pred_width:list = [95.0]) -> pd.DataFrame:
    
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param method: str - one of {'naive','seasonal naive','drift','mean'}, the method to simulate the forecast
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - Seasonal period
        :param bootstrap: bool - toggle bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    if method == 'naive':
        forecast = naive_pi(df, target_col, horizon, bootstrap, repetitions, pred_width)

    elif method == 'seasonal naive':   
        forecast = s_naive_pi(df,target_col, horizon, period, bootstrap, repetitions, pred_width)
    
    elif method == 'drift':
        forecast = drift_pi(df,target_col, horizon, bootstrap, repetitions, pred_width)
    
    elif method == 'mean':
        forecast = mean_pi(df,target_col, horizon, bootstrap, repetitions, pred_width)

    return forecast
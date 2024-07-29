import pandas as pd
import numpy as np
from models import naive, drift_method, mean_method
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



def bs_naive_error(df:pd.DataFrame,target_col:str, period:int = 1) -> pd.Series:

    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index
        :param target_col: str - column with historical data
        :param period: int - number of periods to shift the error by (seasonal period)
    Ouputs:
        pandas.DataFrame: dataframe with errors from shifted observations to sample from
    """

    bs = pd.DataFrame(index = df.index)
    bs['naive'] = df[target_col].shift(period)  
    bs['error'] = df[target_col] - bs['naive']

    return bs['error'].dropna()



def bs_drift_error(df:pd.DataFrame,target_col:str) -> pd.Series:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index
        :param target_col: str - column with historical data
    Ouputs:
        pandas.DataFrame: dataframe with errors from drift one-step forecasts
    """

    fitted_values = [np.nan, np.nan]

    for i in range(2,len(df)):
        forecast = drift_method(df[target_col].iloc[:i],horizon=1)
        fitted_values.append(forecast[0])
    
    bs = pd.DataFrame(index = df.index)
    bs['slope'] = fitted_values
    bs['error'] = bs['slope'] - df[target_col]

    return bs['error'].dropna()



def bs_mean_error(df:pd.DataFrame,target_col:str) -> pd.Series:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index
        :param target_col: str - column with historical data
    Ouputs:
        pandas.DataFrame: dataframe with errors from mean one-step forecasts
    """

    fitted_values = [np.nan]

    for i in range(1,len(df)):
        forecast = mean_method(df[target_col].iloc[:i],horizon=1)
        fitted_values.append(forecast[0])

    bs = pd.DataFrame(index = df.index)
    bs['slope'] = fitted_values
    bs['error'] = bs['slope'] - df[target_col]

    return bs['error'].dropna()





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



def bs_output(forecast_df:pd.DataFrame, pred_width:float = 95) -> pd.DataFrame :
    """
    Inputs:
        :param forecast_df: pd.DataFrame - Data frame of simulated forecasts to calculate mean and prediction intervals from
        :param pred_width: int - Width of prediction interval (an integer between 0 and 100)
    Outputs:
        pd.DataFrame: Data frame with forecast dates as the index and the mean, the lower and upper bounds for the prediction interval
    """

    new_pred_width = (100 - (100-pred_width)/2) / 100

    #storing the mean and quantiles for each forecast point in columns
    #the dates in forecast_df are 'ds' from the forecast_dates function
    output_forecast = pd.DataFrame(forecast_df.mean(axis=1), columns=['forecast'])
    output_forecast['lower_pi'] = forecast_df.quantile(1 - new_pred_width, axis=1)
    output_forecast['upper_pi'] = forecast_df.quantile(new_pred_width, axis=1)

    return output_forecast

def fitted_forecast(df:pd.DataFrame, target_col:str, method:str, period:int=1) -> list:
    
    """
    Inputs: 
        :param df: pd.DataFrame - Time series to calculate residuals from
        :param target_col: str - column with historical data
        :param method: str - one of {'naive','seasonal naive','drift','mean'}, the method to simulate the forecast
        :param period: int - Seasonal period
    Outputs:
        float: the standard deviation of the residuals
    """

    if method == 'naive':
        fitted_fcst = [df[target_col].iloc[-1]] * len(df)

    elif method == 'seasonal naive':

        season = df[target_col].iloc[-period:].to_list()
        mult_season = int(len(df) / period) + 1
        fitted_fcst = (season * mult_season)[-len(df) % period:]
    
    elif method == 'drift':

        slope = (df[target_col].iloc[-1] - df[target_col].iloc[0]) / len(df)

        fitted_fcst = [df[target_col].iloc[0] + slope * i for i in range(1, len(df) + 1)]

    elif method == 'mean':
        mean = sum(df[target_col]) / len(df)
        fitted_fcst = [mean] * len(df)

    return fitted_fcst
    




def residual_sd(df:pd.DataFrame,target_col:str, fitted_fcst:pd.DataFrame, no_missing_values:int, no_parameters:int=0) -> float :
    """
    Inputs: 
        :param df: pd.DataFrame - Time series to calculate residuals from
        :param target_col: str - column with historical data
        :param fitted_fcst: list - fitted forecast from fitted_forecast function
        :param no_ missing_values: int - number of values that are needed to calculate the forecast
        :param no_parameters: int - number of parameters used to calculate the forecast
    Outputs:
        float: the standard deviation of the residuals
    """
    square_residuals = [(df[target_col].iloc[i]-fitted_fcst[i]) ** 2 for i in range(len(df))]
    residual_sd = np.sqrt(1 / (len(df)-no_missing_values-no_parameters) * sum(square_residuals))

    return residual_sd



def pi_output(forecast_df:pd.DataFrame, horizon:int, forecast_sd:list, pred_width:float = 95) -> pd.DataFrame:
    """
    Inputs:
        :param forecast_df: pd.DataFrame - Data frame with extended dates and forecasted points
        :param horizon: int - Number of timesteps forecasted into the future
        :param forecast_sd: list - Multi-step standard deviation for each forecasted point
        :param pred_width: float - 0 <= pred_width < 100 width of prediction interval
        :param period: int - Seasonal period
    Outputs:
        pd.DataFrame: Data frame with forecast dates as the index and the mean, the lower and upper bounds for the prediction interval

    """

    new_pred_width = (100 - (100 - pred_width) / 2) / 100 

    #calculating the multiplier for the forecast standard deviation depending on the prediction width
    pi_mult = norm.ppf(new_pred_width)
    
    forecast_df['lower_pi'] = [forecast_df['forecast'][i] - pi_mult * forecast_sd[i] for i in range(horizon)]
    forecast_df['upper_pi'] = [forecast_df['forecast'][i] + pi_mult * forecast_sd[i] for i in range(horizon)]

    return forecast_df





######################################################################################################################################





def naive_pi(df:pd.DataFrame, target_col:str, horizon:int, bootstrap=False, repetitions=100, pred_width = 95.0) -> pd.DataFrame:
   
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with dates as index
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param bootstrap: bool - bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: float - 0 <= pred_width < 100 width of prediction interval
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    if bootstrap:

        ###Regular bootstrap###

        #continuing the dates from df
        forecast_df = forecast_dates(df,horizon)
        naive_error = bs_naive_error(df,target_col)
        
        #creating repetitions of the bootstrapped forecasts and storing them
        for run in range(repetitions):
            forecast_df[f'run_{run}'] = bs_forecast(df, target_col, horizon, naive_error) 

        #calculating the mean and quantiles
        output_forecast = bs_output(forecast_df,pred_width)

        return output_forecast
        
    else:

        ###Naive###

        #storing the extended dates and forecasted values in a dataframe
        forecast_df = forecast_dates(df,horizon)
        forecast_df['forecast'] = naive(df,target_col, horizon)

        #using a fitted forecast to compare to the data to calculate the residuals
        fitted_fcst = fitted_forecast(df,target_col, method = 'naive')

        #calculating the residuals using the forecast (the last observed value)
        sd_resiuals = residual_sd(df,target_col,fitted_fcst,no_missing_values=1)
        forecast_sd = [sd_resiuals * np.sqrt(h) for h in range(1,horizon+1)]
        print(sd_resiuals, forecast_sd)
        
        #creating an output forecast with the naive forecast and prediction interval
        output_forecast = pi_output(forecast_df,horizon,forecast_sd,pred_width)
        
        return output_forecast
    



def s_naive_pi(df:pd.DataFrame,target_col:str, horizon:int,  period:int, bootstrap=False, repetitions=100, pred_width = 95.0) -> pd.DataFrame:
   
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - Seasonal period
        :param bootstrap: bool - toggle bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: float - 0 <= pred_width < 100 width of prediction interval
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    if bootstrap:

        forecast_df = forecast_dates(df,horizon)

        naive_errors = bs_naive_error(df,target_col,period)

        for run in range(repetitions):
            forecast_df[f'run_{run}'] = bs_forecast(df,target_col,horizon,naive_errors)

        output_forecast = bs_output(forecast_df,pred_width)

        return output_forecast
        
    else:

        ###Seasonal naive###

        #calculating the seasonal naive forecast for df
        forecast_df = forecast_dates(df,horizon)
        forecast_df['forecast'] = naive(df,target_col,horizon,period)

        #using a fitted forecast to compare to the data to calculate the residuals

        fitted_fcst = fitted_forecast(df,target_col, method = 'seasonal naive', period=period)
        sd_residuals = residual_sd(df,target_col,fitted_fcst, no_missing_values=period)

        #using the number of seasons prior to each point to calculate the standard deviation of forecasted points
        seasons_in_forecast = [np.floor((h-1) / period) for h in range(1,horizon+1)] 
        forecast_sd = [sd_residuals * np.sqrt(seasons_in_forecast[h-1]+1) for h in range(1,horizon+1)]

        #creating an output forecast with the seasonal naive forecast and prediction interval
        output_forecast = pi_output(forecast_df,horizon,forecast_sd, pred_width)
        
        return output_forecast



def drift_pi(df:pd.DataFrame,target_col:str,horizon:int,bootstrap:bool = False, repetitions:int = 100, pred_width=95.0) -> pd.DataFrame:

    
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param bootstrap: bool - toggle bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: float - 0 <= pred_width < 100 width of prediction interval
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    if bootstrap:
        forecast_df = forecast_dates(df,horizon)

        drift_errors = bs_drift_error(df,target_col)

        for run in range(repetitions):
            forecast_df[f'run_{run}'] = bs_forecast(df,target_col,horizon,drift_errors)
        
        output_forecast = bs_output(forecast_df, pred_width)

        return output_forecast
    else:
        #setting up the forecast dataframe
        forecast_df = forecast_dates(df,horizon)
        forecast_df['forecast'] = drift_method(df,horizon)

        #using a fitted forecast to compare to the data to calculate the residuals

        fitted_fcst = fitted_forecast(df,target_col, method = 'drift')

        #calculating the residuals
        sd_residuals = residual_sd(df,target_col,fitted_fcst,no_missing_values=2, no_parameters=1)

        #calculating the standard deviation for the residuals and the forecasted points
        forecast_sd = [sd_residuals * np.sqrt(i * (1 + i/(len(df)-1))) for i in range(1,horizon+1)]

        #outputting the result
        output_forecast = pi_output(forecast_df,horizon,forecast_sd,pred_width)
        
        return output_forecast



def mean_pi(df:pd.DataFrame,target_col:str, horizon=int, bootstrap:bool = False, repetitions:int = 100, pred_width=95.0) -> pd.DataFrame:

    
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param horizon: int - Number of timesteps forecasted into the future
        :param bootstrap: bool - toggle bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: float - 0 <= pred_width < 100 width of prediction interval
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """
    
    if bootstrap:

        forecast_df = forecast_dates(df,horizon)

        mean_errors = bs_mean_error(df,target_col)

        for run in range(repetitions):
            forecast_df[f'run_{run}'] = bs_forecast(df,target_col,horizon,mean_errors)

        output_forecast = bs_output(forecast_df, pred_width)

        return output_forecast
    
    else:

        #setting up the forecast dataframe
        forecast_df = forecast_dates(df,horizon)
        forecast_df['forecast'] = mean_method(df,horizon)

        #using a fitted forecast to compare to the data to calculate the residuals
        fitted_fcst = fitted_forecast(df,target_col, method = 'mean')

        #calculating the standard deviation of the residuals and the forecasted points
        sd_residuls = residual_sd(df,target_col,fitted_fcst, no_missing_values=2, no_parameters=1)
        forecast_sd = [sd_residuls * np.sqrt(1 + 1/len(df))] * horizon

        #outputting the result
        output_forecast = pi_output(forecast_df,horizon,forecast_sd, pred_width)

        return output_forecast



def STL_pi(df:pd.DataFrame,target_col:str, horizon:int, method:str,
           period:int=1, bootstrap:bool = False, repetitions:int = 100, pred_width=95.0) -> pd.DataFrame:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param method: str - one of {'naive','seasonal naive','drift','mean'}, the method to simulate the forecast
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - seasonal period
        :param bootstrap: bool - toggle bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: float - 0 <= pred_width < 100 width of prediction interval
    Output:
        pandas.DataFrame: a forecast for the trend using the method specified
    """

    stl = STL(df[target_col], period=period).fit()
    trend = stl.trend()

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
                       bootstrap:bool=False, repetitions:int=100, pred_width:float=95) -> pd.DataFrame:
    
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param method: str - one of {'naive','seasonal naive','drift','mean'}, the method to simulate the forecast
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - Seasonal period
        :param bootstrap: bool - toggle bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: float - 0 <= pred_width < 100 width of prediction interval
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
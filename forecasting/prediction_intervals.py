import pandas as pd
import numpy as np
from models import *
import random
from scipy.stats import norm
from math import *


def bs_error(df:pd.DataFrame) -> pd.DataFrame:

    """
    Inputs:
        df (pandas.Series): Historical time series data
    Ouputs:
        pandas.DataFrame: 
    """
    bs = pd.DataFrame()
    bs['ds'] = df.iloc[:,0]
    bs['naive'] = df.iloc[:,1].shift(1)  
    bs['error'] = df.iloc[:,1] - bs['naive']
    bs['error'] = bs['error'].dropna()

    return bs



def forecast_dates(df:pd.DataFrame, horizon:int) -> pd.DataFrame :
    """
    Inputs:
        df: Historical time series data to continue dates
        horizon: Number of timesteps forecasted into the future
    Ouputs:
        pandas.DataFrame: A data frame with dates continued from df to the forecast horizon
    """

    forecast_df = pd.DataFrame()

    ds = pd.to_datetime(df.iloc[:,0])
    forecast_df['ds'] = pd.date_range(start = ds.iloc[-1], periods = horizon+1, freq = ds.dt.freq)
    
    forecast_df['ds'] = forecast_df['ds'].shift(-1)

    return forecast_df.dropna()



def bs_forecast(df:pd.DataFrame,horizon:int) -> list:

    """
    Inputs:
        df: Historical time series data
        horizon: Number of timesteps forecasted into the future
    Ouputs:
        list: a bootstrapped forecast by randomly sampling from the errors
    """

    bs = bs_error(df)

    #using the last entry in df to start the sampling
    forecast_list = [ df.iloc[-1,1] + random.choice(bs['error']) ]

    for i in range(1,horizon):

        sample = forecast_list[-1] + random.choice(bs['error'])
        forecast_list.append(sample)

    return forecast_list



def bs_output(forecast_df:pd.DataFrame, pred_width:float = 95) -> pd.DataFrame :
    """
    Inputs:
        forecast_df: Data frame of simulated forecasts to calculate mean and prediction intervals from
        pred_width: Width of prediction interval (an integer between 0 and 100)
    Outputs:
        pd.DataFrame: Data frame with forecast dates as the index and the mean, the lower and upper bounds for the prediction interval
    """
    ds = forecast_df.iloc[:,0]
    new_pred_width = (100 - (100-pred_width)/2) / 100

    #storing the mean and quantiles for each forecast point in columns
    #the dates in forecast_df are 'ds' from the forecast_dates function)
    output_forecast = pd.DataFrame(forecast_df.set_index('ds').mean(axis=1), columns=['forecast'])
    output_forecast['lower_pi'] = forecast_df.set_index('ds').quantile(1 - new_pred_width, axis=1)
    output_forecast['upper_pi'] = forecast_df.set_index('ds').quantile(new_pred_width, axis=1)

    return output_forecast


###Naive functions###

def naive_output(forecast_df:pd.DataFrame, horizon:int, forecast_sd:float, pred_width:float = 95) -> pd.DataFrame:
    """
    Inputs:
        forecast_df: Data frame with extended dates and forecasted points
        horizon: Number of timesteps forecasted into the future
        forecast_sd: Multi-step standard deviation from the residuals
        pred_width: width of prediction interval (an integer between 0 and 100)
        period: Seasonal period
    Outputs:
        pd.DataFrame: Data frame with forecast dates as the index and the mean, the lower and upper bounds for the prediction interval

    """
    output_forecast = forecast_df
    output_forecast.set_index('ds')
    new_pred_width = (100 - (100 - pred_width) / 2) / 100 

    #calculating the multiplier for the forecast standard deviation depending on the prediction width
    pi_mult = norm.ppf(new_pred_width)
    print(forecast_df.iloc[0,1], pi_mult, forecast_sd[0] )
    
    output_forecast['lower_pi'] = [forecast_df.iloc[i,1] - pi_mult * forecast_sd[i] for i in range(horizon)]
    output_forecast['upper_pi'] = [forecast_df.iloc[i,1] + pi_mult * forecast_sd[i] for i in range(horizon)]

    return output_forecast

def residual_sd(df:pd.DataFrame, extended_forecast:list, no_missing_values:int, no_parameters:int=0) -> float :
    """
    Inputs: 
        df: Time series to calculate residuals from
        extended forecast: forecast extended back in time to compare df to
        no_ missing_values: number of values that are needed to calculate the forecast
        no_parameters: number of parameters used to calculate the forecast
    Outputs:
        float: the standard deviation of the residuals
    """
    square_residuals = [(df.iloc[i,1]-extended_forecast[i]) ** 2 for i in range(len(df))]
    residual_sd = np.sqrt(1 / (len(df)-no_missing_values-no_parameters) * sum(square_residuals))

    return residual_sd




def naive_pi(df:pd.DataFrame, horizon:int, period=1, bootstrap=False, repetitions=100, pred_width = 95.0) -> pd.DataFrame:
   
    """
    Inputs:
        df: Historical time series data
        horizon: Number of timesteps forecasted into the future
        period: Seasonal period
        bootstrap: bootstrap or normal prediction interval
        repetitions: Number of bootstrap repetitions
        pred_width: width of prediction interval interval (an integer between 0 and 100)
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    if period == 1:

        if bootstrap:

            ###Regular bootstrap###

            #continuing the dates from df
            forecast_df = forecast_dates(df,horizon)
            
            #creating repetitions of the bootstrapped forecasts and storing them
            for run in range(repetitions):
                forecast_df[f'forecast_run_{run}'] = bs_forecast(df, horizon) 

            #calculating the mean and quantiles
            output_forecast = bs_output(forecast_df,pred_width,bootstrap)

            return output_forecast
            
        else:

            ###Naive###
    
            #storing the extended dates and forecasted values in a dataframe
            forecast_df = forecast_dates(df, horizon)
            forecast_df['forecast'] = naive(df,horizon)
            extended_forecast = [df.iloc[-1,1]] * len(df)

            #calculating the residuals using the forecast (the last observed value)
            sd_resiuals = residual_sd(df,extended_forecast,no_missing_values=1)
            forecast_sd = [sd_resiuals * np.sqrt(h) for h in range(1,horizon+1)]
            
            #creating an output forecast with the naive forecast and prediction interval
            output_forecast = naive_output(forecast_df,horizon,forecast_sd,pred_width)
            
            return output_forecast
        
    else:

        if bootstrap:

            ###Seasonal bootstrap###

            return
        
        else:

            ###Seasonal naive###

            #calculating the seasonal naive forecast for df
            forecast_df = forecast_dates(df,horizon)
            forecast_df['forecast'] = s_naive(df,period,horizon)

            #extending the forecast backwards in time so we can calculate the residuals
            season = df.iloc[-period:,1].tolist()
            mult_season = int(len(df) / period) + 1
            extended_forecast = (season * mult_season)[-len(df) % period: ]

            sd_residuals = residual_sd(df,extended_forecast, no_missing_values=period)

            #using the number of seasons prior to each point to calculate the standard deviation of forecasted points
            seasons_in_forecast = [np.floor((h-1) / period) for h in range(1,horizon+1)] 
            forecast_sd = [sd_residuals * np.sqrt(seasons_in_forecast[h-1]+1) for h in range(1,horizon+1)]

            #creating an output forecast with the seasonal naive forecast and prediction interval
            output_forecast = naive_output(forecast_df,horizon,forecast_sd, pred_width)
            
            return output_forecast



def normal_drift_pi(df,horizon):

    """
    Inputs:
        df (pandas.Series): Historical time series observations (in order)
        horizon (int): Number of timesteps forecasted into the future
    Outputs:
        list: the 95% prediction interval for eah point forecast
    """
    forecast = drift_method(df,horizon)

    latest_obs = df.iloc[-1]
    first_obs = df.iloc[0]

    slope = (latest_obs - first_obs) / (len(df) - 1)

    #extending the forecast to calculate the residuals
    extended_forecast = [first_obs + slope * h for h in range(1, len(df) + 1)]

    #calculating the residuals missing out the first and last entries as these are the same as these are not forecasted
    residuals = [df.iloc[i]-extended_forecast[i] for i in range(1,len(df)-1)]

    #calculating the standard deviation for the residuals and the forecasted points
    residual_sd = np.sqrt(1 / (len(df)-1-2) * sum(residuals ** 2))

    forecast_sd = residual_sd * np.sqrt(i * (1+i/(len(df)-1)) for i in range(1,horizon+1))

    #calculating the 95% prediction intervals for each point
    pred_int = [[forecast[i] - 1.96 * forecast_sd[i], forecast[i] + 1.96 * forecast_sd]
                for i in range(horizon)]
    
    return pred_int

def bootstrap_naive(df,horizon,width):
    """
    Inputs:
        df (pandas.Series): Historical time series observations (in order)
        horizon (int): Number of timesteps forecasted into the future
    Outputs:
        list: Bootstrapped prediction interval, of the width secified
    """
    df['naive'] = df.iloc[:,1].shift(1)  
    df['errors'] = df.iloc[:,1] - df['naive']
    forecast_df = pd.DataFrame()

    return
import pandas as pd
import numpy as np
from models import *
import random
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL





def bs_error(df:pd.DataFrame) -> pd.DataFrame:

    """
    Inputs:
        df (pandas.Series): Historical time series data
    Ouputs:
        pandas.DataFrame: 
    """

    bs = pd.DataFrame()
    bs.index = df.index
    bs['naive'] = df.shift(1)  
    bs['error'] = df.iloc[:,0] - bs['naive']

    return bs.dropna()



def forecast_dates(df:pd.DataFrame, horizon:int) -> pd.DataFrame :
    """
    Inputs:
        df: Historical time series data to continue dates
        horizon: Number of timesteps forecasted into the future
    Ouputs:
        pandas.DataFrame: A data frame with dates continued from df to the forecast horizon
    """

    ds = pd.to_datetime(df.index)
    forecast_ds = pd.date_range(start = ds[-1], periods = horizon+1, freq = ds.freq)
    
    return pd.DataFrame(index=forecast_ds[1:])



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
    forecast_list = [ df.iloc[-1,0] + random.choice(bs['error']) ]

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

    new_pred_width = (100 - (100-pred_width)/2) / 100

    #storing the mean and quantiles for each forecast point in columns
    #the dates in forecast_df are 'ds' from the forecast_dates function
    output_forecast = pd.DataFrame(forecast_df.mean(axis=1), columns=['forecast'])
    output_forecast['lower_pi'] = forecast_df.quantile(1 - new_pred_width, axis=1)
    output_forecast['upper_pi'] = forecast_df.quantile(new_pred_width, axis=1)

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
    square_residuals = [(df.iloc[i,0]-extended_forecast[i]) ** 2 for i in range(len(df))]
    residual_sd = np.sqrt(1 / (len(df)-no_missing_values-no_parameters) * sum(square_residuals))

    return residual_sd



def pi_output(forecast_df:pd.DataFrame, horizon:int, forecast_sd:list, pred_width:float = 95) -> pd.DataFrame:
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

    new_pred_width = (100 - (100 - pred_width) / 2) / 100 

    #calculating the multiplier for the forecast standard deviation depending on the prediction width
    pi_mult = norm.ppf(new_pred_width)
    
    forecast_df['lower_pi'] = [forecast_df.iloc[i,0] - pi_mult * forecast_sd[i] for i in range(horizon)]
    forecast_df['upper_pi'] = [forecast_df.iloc[i,0] + pi_mult * forecast_sd[i] for i in range(horizon)]

    return forecast_df





######################################################################################################################################





def naive_pi(df:pd.DataFrame, horizon:int, bootstrap=False, repetitions=100, pred_width = 95.0) -> pd.DataFrame:
   
    """
    Inputs:
        df: Historical time series data with dates as index
        horizon: Number of timesteps forecasted into the future
        bootstrap: bootstrap or normal prediction interval
        repetitions: Number of bootstrap repetitions
        pred_width: width of prediction interval interval (an integer between 0 and 100)
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    if bootstrap:

        ###Regular bootstrap###

        #continuing the dates from df
        forecast_df = forecast_dates(df,horizon)
        
        #creating repetitions of the bootstrapped forecasts and storing them
        for run in range(repetitions):
            forecast_df[f'run_{run}'] = bs_forecast(df, horizon) 

        #calculating the mean and quantiles
        output_forecast = bs_output(forecast_df,pred_width)

        return output_forecast
        
    else:

        ###Naive###

        #storing the extended dates and forecasted values in a dataframe
        forecast_df = forecast_dates(df,horizon)
        forecast_df['forecast'] = naive(df,horizon)

        extended_forecast = [df.iloc[-1,0]] * len(df)

        #calculating the residuals using the forecast (the last observed value)
        sd_resiuals = residual_sd(df,extended_forecast,no_missing_values=1)
        forecast_sd = [sd_resiuals * np.sqrt(h) for h in range(1,horizon+1)]
        
        #creating an output forecast with the naive forecast and prediction interval
        output_forecast = pi_output(forecast_df,horizon,forecast_sd,pred_width)
        
        return output_forecast
    



def s_naive_pi(df:pd.DataFrame, horizon:int,  period, bootstrap=False, repetitions=100, pred_width = 95.0) -> pd.DataFrame:
   
    """
    Inputs:
        df: Historical time series data
        horizon: Number of timesteps forecasted into the future
        period: Seasonal period
        bootstrap: toggle bootstrap or normal prediction interval
        repetitions: Number of bootstrap repetitions
        pred_width: width of prediction interval interval (an integer between 0 and 100)
    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    if bootstrap:

        ###Seasonal bootstrap###

        #setting up df to pass into stl
        seasonal_bs = df
        stl = STL(seasonal_bs.iloc[:,0],period=period).fit()

        #bootstrapping the trend for df
        trend = stl.trend.to_frame()
        trend_pi = naive_pi(trend,horizon,bootstrap=True, repetitions=repetitions, pred_width=pred_width)

        #performing seasonal naive on the seasonal data
        seasonal = stl.seasonal.to_frame()
        seasonal_pi = s_naive_pi(seasonal, horizon,repetitions=repetitions, period=period)

        return trend_pi + seasonal_pi
        
    else:

        ###Seasonal naive###

        #calculating the seasonal naive forecast for df
        forecast_df = forecast_dates(df,horizon)
        forecast_df['forecast'] = s_naive(df,period,horizon)

        #extending the forecast backwards in time so we can calculate the residuals
        season = df.iloc[-period:,0].to_list()
        mult_season = int(len(df) / period) + 1
        extended_forecast = (season * mult_season)[-len(df) % period: ]
        sd_residuals = residual_sd(df,extended_forecast, no_missing_values=period)

        #using the number of seasons prior to each point to calculate the standard deviation of forecasted points
        seasons_in_forecast = [np.floor((h-1) / period) for h in range(1,horizon+1)] 
        forecast_sd = [sd_residuals * np.sqrt(seasons_in_forecast[h-1]+1) for h in range(1,horizon+1)]

        #creating an output forecast with the seasonal naive forecast and prediction interval
        output_forecast = pi_output(forecast_df,horizon,forecast_sd, pred_width)
        
        return output_forecast



def drift_pi(df:pd.DataFrame,horizon:int,pred_width=95.0) -> pd.DataFrame:

    """
    Inputs:
        df: Historical time series observations
        horizon: Number of timesteps forecasted into the future
        period: Seasonal period
    Outputs:
        pandas.DataFrame: a normal prediction interval for df usuing the drift method
    """

    #setting up the forecast dataframe
    forecast_df = forecast_dates(df,horizon)
    forecast_df['forecast'] = drift_method(df,horizon)

    #extending the forecast to test the forecast on observed data
    latest_obs = df.iloc[-1,0]
    first_obs = df.iloc[0,0]

    slope = (latest_obs - first_obs) / len(df)

    extended_forecast = [first_obs + slope * i for i in range(1, len(df) + 1)]

    #calculating the residuals
    sd_residuals = residual_sd(df,extended_forecast,no_missing_values=2, no_parameters=1)

    #calculating the standard deviation for the residuals and the forecasted points
    forecast_sd = [sd_residuals * np.sqrt(i * (1 + i/(len(df)-1))) for i in range(1,horizon+1)]

    #outputting the result
    output_forecast = pi_output(forecast_df,horizon,forecast_sd,pred_width)
    
    return output_forecast



def mean_pi(df:pd.DataFrame, horizon=int, pred_width=95.0) -> pd.DataFrame:

    """
    Inputs:
        df: Historical time series observations
        horizon: Number of timesteps forecasted into the future
        period: Seasonal period
    Outputs:
        pandas.DataFrame: a normal prediction interval for df usuing the mean method
    """

    #setting up the forecast dataframe
    forecast_df = forecast_dates(df,horizon)
    forecast_df['forecast'] = mean_method(df,horizon)

    #extending the forecast back in time to test against observed data
    extended_forecast = [forecast_df['forecast'].iloc[0]] * len(df)

    #calculating the standard deviation of the residuals and the forecasted points
    sd_residuls = residual_sd(df,extended_forecast, no_missing_values=2, no_parameters=1)
    forecast_sd = [sd_residuls * np.sqrt(1 + 1/len(df))] * horizon

    #outputting the result
    output_forecast = pi_output(forecast_df,horizon,forecast_sd, pred_width)

    return output_forecast

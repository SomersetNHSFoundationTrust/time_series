import pandas as pd
import numpy as np
from .bootstrap_naive_models import naive_method, naive_error,  drift_method,  drift_error, mean_method, mean_error, forecast_dates
from scipy.stats import norm



def pi_output(forecast_df:pd.DataFrame, horizon:int, forecast_sd:list, pred_width:list = [95,80]) -> pd.DataFrame:
    """
    Uses the forecast with extended dates and standard deviation in each step of the forecast
    to calculate the prediction intervals for the forecast.

    The standard deviation for each forecast step is calculated using the standard deviation of the residuals
    for the model forecast.

    Inputs:
        :param forecast_df: pandas.DataFrame - Data frame with extended dates and forecasted points.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param forecast_sd: list - Multi-step standard deviation from the most recent observation for each forecasted point.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
    Outputs:
        pandas.DataFrame: Data frame with the forecasted dates as the index,
                          and the lower and upper bounds for the prediction intervals as columns.

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





def naive_pi(df:pd.DataFrame, target_col:str, horizon:int, period:int=1, pred_width:list = [95,80],**kwargs) -> pd.DataFrame:
   
    """
    Takes in a dataframe with date-time index and forecast horizon and outputs a data frame with dates continued
    from df, with the naive forecast and upper and lower bounds for the prediction intervals as columns.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data with dates as index.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
    Output:
        pandas.DataFrame: Naive or seasonal naive forecast and prediction intervals for df.
    """

    #extending the dates from df and storing the forecast in forecast_df
    forecast_df = forecast_dates(df,horizon)
    forecast_df['forecast'] = naive_method(df,target_col, horizon,period)

    #calculating thr errors from the fitted forecast to calculate the residuals
    naive_errors = naive_error(df,target_col)['error']

    #calculating the standard deviation of the residuals, removing the first seasonal period as we cannot forecast this using this model 
    sd_residuals = np.std(naive_errors)

    #calculating the forecast standard deviation
    seasons_in_forecast = [int((h-1) / period) for h in range(1,horizon+1)] 
    forecast_sd = [sd_residuals * np.sqrt(seasons_in_forecast[h]+1) for h in range(horizon)]

    output_forecast = pi_output(forecast_df,horizon,forecast_sd, pred_width)

    return output_forecast




def drift_pi(df:pd.DataFrame,target_col:str,horizon:int, period:int = 1, pred_width:list = [95,80],**kwargs) -> pd.DataFrame:
    """
    Takes in a dataframe with date-time index and forecast horizon and outputs a data frame with dates continued
    from df, with the drift forecast and upper and lower bounds for the prediction intervals as columns.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data with dates as index.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
    Output:
        pandas.DataFrame: Drift forecast and prediction intervals for df.
    """
    #extending the dates from df and storing the forecast in forecast_df
    forecast_df = forecast_dates(df,horizon)
    forecast_df['forecast'] = drift_method(df,target_col,horizon)

    #calculating the errors from the fotted forecast to calculate the residuals
    drift_errors = drift_error(df,target_col)['error']

    #calculating the standard deviation of the residuals, with one degree of freedom as we have a parameter
    sd_residuals = np.std(drift_errors,ddof = 1)

    #calculating the standard deviation for the forecasted points
    forecast_sd = [sd_residuals * np.sqrt(i * (1 + i/(len(df)-1))) for i in range(1,horizon+1)]

    #outputting the result
    output_forecast = pi_output(forecast_df,horizon,forecast_sd,pred_width)
    
    return output_forecast



def mean_pi(df:pd.DataFrame,target_col:str, horizon:int, period:int = 1, pred_width:list = [95,80],**kwargs) -> pd.DataFrame:
    """
    Takes in a dataframe with date-time index and forecast horizon and outputs a data frame with dates continued
    from df, with the mean forecast and upper and lower bounds for the prediction intervals as columns.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data with dates as index.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
    Output:
        pandas.DataFrame: Mean forecast and prediction intervals for df.
    """

    #extending the dates from df and storing the forecast in forecast_df
    forecast_df = forecast_dates(df,horizon)
    forecast_df['forecast'] = mean_method(df,target_col,horizon)

    #calculating the errors from the fitted forecast to calculate the residuals
    mean_errors = mean_error(df,target_col)['error']

    #calculating the standard deviation of the residuals, with one degree of freedom as we have a parameter
    sd_residuls = np.std(mean_errors,ddof=1)
    forecast_sd = [sd_residuls * np.sqrt(1 + 1/len(df))] * horizon

    #outputting the result
    output_forecast = pi_output(forecast_df,horizon,forecast_sd, pred_width)

    return output_forecast



def normal_benchmark_forecast(df:pd.DataFrame, target_col:str, model:str, horizon:int, period:int=1,
                              pred_width:list = [95,80],**kwargs) -> pd.DataFrame:
    """
    Creates a forecast of the desired model using the forecast functions.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column with historical data.
        :param model: str - One of {'naive','drift','mean'}, the model to calculate the forecast.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
    Output:
        pandas.DataFrame: Forecast and prediction intervals for df with the specified model.
    """

    if model == 'naive':
        forecast = naive_pi(df, target_col, horizon, period, pred_width)
    
    elif model == 'drift':
        forecast = drift_pi(df,target_col, horizon, pred_width)
    
    elif model == 'mean':
        forecast = mean_pi(df,target_col, horizon, pred_width)

    return forecast
    

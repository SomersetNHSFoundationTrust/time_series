import pandas as pd

import numpy as np

from .naive_method import forecast_dates, bs_pi_output





def drift_method(df:pd.DataFrame, target_col:str, horizon:int) -> list:
    """

    Creates a drift forecast in the form of a list.

    Inputs:

        :param df: pandas.DataFrame - Historical time series data with date-time index.
        
        :param target_col: str - column with historical data.
        
        :param horizon: int - Number of timesteps forecasted into the future.
            
    Outputs:
    
        list: Forecasted time series with drift method.
    """


    latest_obs = df[target_col].iloc[-1]

    first_obs = df[target_col].iloc[0]



    slope = (latest_obs - first_obs) / (len(df) - 1)


    forecast_list = [latest_obs + slope * h for h in range(1, horizon + 1)]



    return forecast_list




def drift_one_step_forecast(df:pd.DataFrame, target_col:str):
    """

    Returns a one step drift forecast.

    Inputs:

        df: pandas.DataFrame - Historical timeseries data with date-time index.

        target_col: str - Column with historical timeseries data.

    Outputs:

        float - One step drift forecast.
    """


    return df[target_col].iloc[-1] + (df[target_col].iloc[-1] - df[target_col].iloc[0]) / (len(df)-1)






def drift_fitted_forecast(df:pd.DataFrame, target_col:str) -> np.array:

   

    """

    Creates and outputs a fitted forecasting using the drift method.

    Inputs:

        df: pandas.DataFrame - Historical timeseries data with date-time index.

        target_col: str - Column with historical data.

    Outputs:

        np.array - an array of the one step forecast errors.


    """


    fitted_values = [drift_method(df.iloc[:i],target_col,horizon=1)[0] for i in range(2,len(df))]
   


    return fitted_values






def drift_single_step_error(df:pd.DataFrame,target_col:str) -> np.array:

    """

    Creates a fitted forecasting using the drift method and outputs the error from the data.
    Can be used to create naive bootstrap forecasts.  

    Inputs:

        :param df: pandas.DataFrame - Historical time series data with date-time index.
        
        :param target_col: str - Column with historical data.
            
    Ouputs:
    
        pandas.DataFrame: Dataframe with errors of drift one-step forecasts to the fitted forecast.
    """


    fitted_values = [drift_method(df.iloc[:i],target_col,horizon=1)[0] for i in range(2,len(df))]



    return df[target_col].iloc[2:].values - fitted_values





def drift_forecaster(df:pd.DataFrame, target_col:str, horizon:int=1, bootstrap_samples: int = 100) -> pd.DataFrame:

 

    """

    Uses the one step fitted forecast errors to create bootstrap forecasts.

    Inputs:

        df: pandas.DataFrame - Historical timeseries data with date-time index.

        target_col: str - Column with historical data.

        horizon: int - Number of time steps forecasted into the future.

        bootstrap_samples: int - Number of samples outputted.

    Outputs:

        pd.DataFrame: A data frame with date-time index continued from the data and each forecast as columns.

    """

 
    initial_forecast = drift_method(df = df,
                                    target_col=target_col,
                                    horizon=2)
    
    one_step_errors = drift_one_step_forecast(df, target_col)

    forecast = initial_forecast + np.random.choice(one_step_errors, 2)
    
    #appending forecast_list to the observed data to forecast from
    new_df = pd.DataFrame(np.append(df[target_col].values, forecast), columns = [target_col])


    #calculating forecasts based on the bootstrapped forecasts before
    if horizon > 2:

        for index in range(2, horizon):

            forecast = drift_one_step_forecast(df = df,
                                               target_col=target_col)
            
            sample = forecast + np.random.choice(one_step_errors)
            new_df.loc[len(df)+index] = sample
        
    
    return new_df[target_col].iloc[-horizon:].values
    













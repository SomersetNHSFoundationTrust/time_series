import pandas as pd

import numpy as np

from .naive_method import forecast_dates, bs_pi_output





def mean_method(df:pd.DataFrame,target_col:str,horizon:int) -> list:

    """
    Creates a mean forecast in the form of a list

    Inputs:

        :param df: pandas.DataFrame - Historical time series data with date-time index.
        
        :param target_col: str - column with historical data.
        
        :param horizon: int - Number of timesteps forecasted into the future.
            
    Outputs:
        
        list: Forecasted time series with mean method.
    """


    mean = np.mean(df[target_col])



    return [mean] * horizon





def mean_fitted_forecast(df:pd.DataFrame, target_col:str) -> np.array:

   

    """

    Creates and outputs a fitted forecasting using the drift method.

    Inputs:

        df: pandas.DataFrame - Historical timeseries data with date-time index.

        target_col: str - Column with historical data.

    Outputs:

        np.array - an array of the one step forecast errors.


    """


    fitted_values = df.copy()[target_col].iloc[:-1].expanding().mean().values[:-1]

   


    return fitted_values




def mean_single_step_error(df:pd.DataFrame, target_col:str) -> np.array:

   

    """

    Creates a fitted forecasting using the naive method and outputs the error from the data.
    Can be used to create naive bootstrap forecasts.

    Inputs:

        data: pandas.DataFrame - Historical timeseries data with date-time index.

        target_col: str - Column with historical data.

    Outputs:

        np.array - an array of the one step forecast errors


    """

   
    #calculating each one step forecast
    one_step_forecasts = df.copy()[target_col].iloc[:-1].expanding().mean().values[:-1]



    #calculating the error
    one_step_error = df[target_col].iloc[1:].values - one_step_forecasts



    return one_step_error





def mean_forecaster(df:pd.DataFrame, target_col:str, horizon:int=1, bootstrap_samples: int = 100) -> pd.DataFrame:

 

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

 
    one_step_error = mean_single_step_error(df, target_col)

   

    forecast_ds = forecast_dates(df, horizon)

 

    # Precompute random choices

    random_choices = pd.DataFrame(np.random.choice(one_step_error, size=(horizon + 1, bootstrap_samples)), index = forecast_ds)

 

    # Generate forecast using vectorized operations

    df_mean = np.mean(df[target_col].values)

    forecast_matrix = random_choices[0,:] + df_mean



    for i in range(bootstrap_samples):

        pass



    forecast_matrix = np.cumsum(random_choices, axis=0)

   

    # Convert the matrix to a DataFrame with appropriate index

    output_forecast = pd.DataFrame(forecast_matrix, index=forecast_ds, columns=[f'sim_{i}' for i in range(bootstrap_samples)])

   

    return output_forecast
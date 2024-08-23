import pandas as pd

import numpy as np





def naive_method(df:pd.DataFrame, target_col:str, horizon:int, period:int=1) -> np.array:
    """

    Creates a naive forecast in the from of a list.

    Inputs:

        :param df: pandas.DataFrame - Historical time series data with date-time index.
        
        :param target_col: str - Column with historical data.
        
        :param horizon: int - Number of timesteps forecasted into the future.
        
        :param period: int - Seasonal period.
    
    Outputs:

        list: Forecasted time series with naive or seasonal naive method.

    """


    most_recent_values = df[target_col].iloc[-period:].values


    reps = int(np.ceil(horizon / period))


    return np.tile(most_recent_values,reps)[:horizon]




 

def forecast_dates(df:pd.DataFrame, horizon:int) -> pd.DataFrame :

    """

    Extends the dates of the date-time index of df until the horizon.

    Inputs:

        :param df: pd.DataFrame - Historical time series data with date-time index.

        :param horizon: int - Number of timesteps forecasted into the future.

    Ouputs:

        pandas.DataFrame: A data frame with dates continued from df to the forecast horizon.

    """


    ds = pd.to_datetime(df.index)


    forecast_ds = pd.date_range(start = ds[-1], periods = horizon+1, freq = ds.freq)

   
    return forecast_ds


def naive_fitted_forecast(df:pd.DataFrame, target_col:str, period:int = 1) -> np.array:

   

    """

    Creates and outputs a fitted forecasting using the naive method

    Inputs:

        df: pandas.DataFrame - Historical timeseries data with date-time index.

        target_col: str - Column with historical data.

        period: int - Seasonal period.

    Outputs:

        np.array - an array of the one step forecast errors


    """

   
    return df[target_col].shift(period)






 
def naive_single_step_error(df:pd.DataFrame, target_col:str, period:int = 1) -> np.array:

   

    """

    Creates a fitted forecasting using the naive method and outputs the error from the data.
    Can be used to create naive bootstrap forecasts.

    Inputs:

        df: pandas.DataFrame - Historical timeseries data with date-time index.

        target_col: str - Column with historical data.

        period: int - Seasonal period.

    Outputs:

        np.array - an array of the one step forecast errors


    """

   
    one_step_error = df[target_col] - df[target_col].shift(period)

   
    return one_step_error.values

 

 

 

def naive_forecaster(df:pd.DataFrame, target_col:str, horizon:int=1, period:int = 1, bootstrap_samples: int = 100) -> pd.DataFrame:

 

    """

    Uses the one step fitted forecast errors to create bootstrap forecasts.

    Inputs:

        df: pandas.DataFrame - Historical timeseries data with date-time index.

        target_col: str - Column with historical data.

        horizon: int - Number of time steps forecasted into the future.

        period: int - Seasonal period.

        bootstrap_samples: int - Number of samples outputted.

    Outputs:

        pd.DataFrame: A data frame with date-time index continued from the data and each forecast as columns.

    """

 
    one_step_error = naive_single_step_error(df, target_col, period)

   

    forecast_ds = forecast_dates(df, horizon)

 

    # Precompute random choices

    random_choices = np.random.choice(one_step_error, size=(horizon + 1, bootstrap_samples))

 

    # Generate forecast using vectorized operations

    last_value = df[target_col].iloc[-1]

    forecast_matrix = np.cumsum(random_choices, axis=0) + last_value

   

    # Convert the matrix to a DataFrame with appropriate index

    output_forecast = pd.DataFrame(forecast_matrix, index=forecast_ds, columns=[f'sim_{i}' for i in range(bootstrap_samples)])

   

    return output_forecast





def bs_pi_output(forecast_data:pd.DataFrame, pred_width:list = [95,80]) -> pd.DataFrame:
    
    """

    Uses a data frame with continued dates as an index and the bootstrap forecasts as columns
    and calculates the mean and quartiles of each step of the forecast.
    These are the forecast and the upper and lower bounds of the prediction intervals for the data.

    Inputs:

        :param forecast_data: pandas.DataFrame - Data frame of simulated forecasts
                                               to calculate the forecast and prediction intervals from.
        
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
    
    Outputs:
        
        pandas.DataFrame: Data frame with the forecasted dates as the index,
                          and the lower and upper bounds for the prediction intervals as columns.
    """

    #storing the mean and quantiles for each forecast point in columns
    output_forecast = pd.DataFrame(forecast_data.mean(axis=1), columns=['forecast'])



    #sorting the widths in reverse order

    pred_width = np.sort(pred_width)


    pred_width = reversed(pred_width)



    for width in pred_width:


        new_pred_width = (100 - (100-width)/2) / 100 



        output_forecast[f'{width}% lower_pi'] = forecast_data.quantile(1 - new_pred_width, axis=1)

        output_forecast[f'{width}% upper_pi'] = forecast_data.quantile(new_pred_width, axis=1)



    return output_forecast





def bs_naive_pi(df: pd.DataFrame, target_col:str, horizon: int, period: int=1, bootstrap_samples:int=100,
                pred_width:list=[95,80]) -> pd.DataFrame:
    
    """

    Takes in a dataframe with date-time index and forecast horizon and outputs a data frame with dates continued
    from df and the bootstrapped naive forecast and upper and lower bounds for the prediction intervals as columns.

    Inputs:

        :param df: pandas.DataFrame - Historical time series data.
        
        :param target_col: str - Column with historical data.
        
        :param horizon: int - Number of timesteps forecasted into the future.
        
        :param period: int - Seasonal period.
        
        :param bootstrap_samples: int - Number of bootstrap repetitions.
        
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
           
    Output:

        pandas.DataFrame: Bootstrapped naive or seasonal naive forecast and prediction intervals for df.
        
    """

    #calculating the bootstrap forecasts
    forecast_data = naive_forecaster(data=df,
                                     target_col=target_col,
                                     horizon=horizon,
                                     period=period,
                                     bootstrap_samples=bootstrap_samples)
    

    
    output_forecast = bs_pi_output(forecast_data=forecast_data,
                                   pred_width=pred_width)
    


    return output_forecast
    






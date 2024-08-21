import pandas as pd
import numpy as np
import random



# **************************************
# Naive Models - Standard implementation
# **************************************


def naive_method(df:pd.DataFrame, target_col:str, horizon:int, period:int=1) -> list:
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
    most_recent_value = df[target_col].iloc[-period:].tolist()

    mult_list = int(np.ceil(horizon / period))
    return (most_recent_value * mult_list)[:horizon]



def drift_method(df:pd.DataFrame, target_col:str, horizon:int, period:int = 1) -> list:
    """
    Creates a drift forecast in the form of a list

    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - column with historical data.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
    Outputs:
        list: Forecasted time series with drift method.
    """
    latest_obs = df[target_col].iloc[-1]
    first_obs = df[target_col].iloc[0]

    slope = (latest_obs - first_obs) / (len(df) - 1)

    forecast_list = [latest_obs + slope * h for h in range(1, horizon + 1)]

    return forecast_list


def mean_method(df:pd.DataFrame,target_col:str,horizon:int, period:int = 1) -> list:
    """
    Creates a mean forecast in the form of a list

    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - column with historical data.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
    Outputs:
        list: Forecasted time series with mean method.
    """

    mean = np.mean(df[target_col])

    return [mean] * horizon

# *****************
# Utility functions
# *****************


def forecast_dates(df:pd.DataFrame, horizon:int) -> pd.DataFrame :
    """
    Extends the dates of the date-time index of df until the horizon.
    
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param horizon: int - Number of timesteps forecasted into the future.
    Ouputs:
        pandas.DataFrame: A dataframe with index of  future dates continued from df to the forecast horizon.
    """

    ds = pd.to_datetime(df.index)
    forecast_ds = pd.date_range(start = ds[-1], periods = horizon+1, freq = ds.freq)
    
    return pd.DataFrame(index=forecast_ds[1:])



def fitted_forecast_error(df:pd.DataFrame, target_col:str,
                          method, no_missing_values:int, period:int=1) -> pd.DataFrame:
    """
    Calculates the fitted forecast for the chosen method and the error from the observed data
    the error from the resulting data frame will be used to randomly sampled from when bootstrapping

    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with historical data.
        :param method - The method to calculate the forecast. Must output in list form, and can be one of
                        {naive_method, drift_method, mean_method}.
        :param no_missing_values: int - The number of values used to calculate the first one step forecast.
        :param period: int - Seasonal period
    Outputs:
        pandas.DataFrame - Dataframe of the fitted forecast and errors to the observed data as columns
    """

    #starting the values list with some missing values
    fitted_values = [np.nan] * no_missing_values

    #calculating each one-step forecast
    for i in range(no_missing_values, len(df)):
        forecast = method(df = df.iloc[:i],
                          target_col = target_col,
                          period=period,
                          horizon = 1)
        
        fitted_values.append(forecast[0])

    #collecting the fitted forecast and errors in a dataframe
    error = pd.DataFrame(index = df.index)
    error['fitted forecast'] = fitted_values
    error['error'] = df[target_col] - error['fitted forecast']

    return error.dropna()



def bs_forecast_values(df:pd.DataFrame, target_col:str, horizon:int,
                       method, one_step_fcst_errors:pd.Series, no_missing_values:int, period:int=1) -> list:
    
    """
    Calculates a forecast of the chosen method by randomly sampling from the one step forecast errors calculated in fitted_forecast_error.
    Outputs the forecast as a list.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of time steps forecasted into the future
        :param method - The method to calculate the forecast. Must output in list form, and can be one of
                        {naive_method, drift_method, mean_method}.
        :param one_step_forecast_errors: pandas.Series - the 'error' column of the output of fitted_forecast_error using the same method
        :param no_missing_values: int - The number of values used to calculate the first one step forecast.
        :param period: int - Seasonal period
    Outputs:
        pandas.DataFrame - The bootstrapped forecast using the chosen method as a list
    """
    
    #creating an initial forecast to add random samples to
    initial_forecast = method(df = df,
                              target_col=target_col,
                              horizon=no_missing_values,
                              period=period)

    forecast_list = [initial_forecast[i] +
                     random.choice(one_step_fcst_errors.values)
                     for i in range(no_missing_values)]
    
    #appending forecast_list to the observed data to forecast from
    new_df = pd.DataFrame(df[target_col].to_list() + forecast_list, columns = [target_col])


    #calculating forecasts based on the bootstrapped forecasts before
    if no_missing_values < horizon:

        for index in range(no_missing_values, horizon):

            forecast = method(df = new_df,
                            target_col = target_col,
                            horizon = 1,
                            period=period)[0]
            
            sample = forecast + random.choice(one_step_fcst_errors.values)
            new_df.loc[len(df)+index] = sample
        
    
    return new_df[target_col].iloc[-horizon:].to_list()
    

    



def bs_output(forecast_df:pd.DataFrame, pred_width:list = [95,80]) -> pd.DataFrame :
    """
    Uses the bs_forecast functions to output a dataframe with the extended dates from forecast_dates, the forecast
    and prediction intervals. The forecast is the mean of each step of the simulated forecasts, and the prediction intervals
    are quantiles of the simulated forecasts.

    Inputs:
        :param forecast_df: pandas.DataFrame - Data frame of simulated forecasts to calculate mean and prediction intervals from.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
    Outputs:
        pandas.DataFrame: Data frame with the forecasted dates as the index,
                          and the lower and upper bounds for the prediction intervals as columns.
    """

    #storing the mean and quantiles for each forecast point in columns
    output_forecast = pd.DataFrame(forecast_df.mean(axis=1), columns=['forecast'])

    #sorting the widths in reverse order
    pred_width = np.sort(pred_width)
    pred_width = reversed(pred_width)

    for width in pred_width:
        new_pred_width = (100 - (100-width)/2) / 100 

        output_forecast[f'{width}% lower_pi'] = forecast_df.quantile(1 - new_pred_width, axis=1)
        output_forecast[f'{width}% upper_pi'] = forecast_df.quantile(new_pred_width, axis=1)

    return output_forecast




# ******************************
# Naive forecasts with bs p.i's
# ******************************

def bs_naive_pi(df: pd.DataFrame, target_col:str, horizon: int, period: int=1, repetitions:int=100,
                pred_width:list=[95,80], simulations:bool=False) -> pd.DataFrame:
    """
    Takes in a dataframe with date-time index and forecast horizon and outputs a data frame with dates continued
    from df and the bootstrapped naive forecast and upper and lower bounds for the prediction intervals as columns.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param repetitions: int - Number of bootstrap repetitions.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
        :param simulations: bool - Toggle whether to additionally return the simulations.
    Output:
        pandas.DataFrame: Bootstrapped naive or seasonal naive forecast and prediction intervals for df.
    """

    #creating a dataframe to store the simulated forecasts and finding the errors to randomly sample from
    forecast_df = forecast_dates(df,horizon)
    naive_errors = fitted_forecast_error(df = df,
                                         target_col = target_col,
                                         method = naive_method,
                                         no_missing_values = period,
                                         period = period)['error']
    #running the simulations
    for run in range(repetitions):
        sim_forecast= bs_forecast_values(df = df,
                                         target_col = target_col,
                                         horizon = horizon,
                                         one_step_fcst_errors = naive_errors,
                                         no_missing_values = period,
                                         period = period)
        
        forecast_df = pd.concat([forecast_df, pd.Series(sim_forecast, index = forecast_df.index, name=f'run_{run}')],axis=1)

    #finding the mean and quartiles of the simulations for the forecast and prediction intervals 
    output_forecast = bs_output(forecast_df, pred_width)

    if simulations:

        return output_forecast, forecast_df
    
    else:

        return output_forecast


def bs_drift_pi(df: pd.DataFrame,target_col:str, horizon: int, period:int = 1, repetitions: int = 100,
                pred_width:list=[95,80],simulations:bool=False) -> pd.DataFrame:
    """
    Takes in a dataframe with date-time index and forecast horizon and outputs a data frame with dates continued
    from df and the bootstrapped drift forecast and upper and lower bounds for the prediction intervals as columns.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param repetitions: int - Number of bootstrap repetitions.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
        :param simulations: bool - Toggle whether to additionally return the simulations.
    Output:
        pandas.DataFrame: Bootstrapped drift forecast and prediction intervals for df.
    """

    #creating a dataframe to store the simulated forecasts and finding the errors to randomly sample from
    forecast_df = forecast_dates(df,horizon)
    drift_errors = fitted_forecast_error(df = df,
                                         target_col = target_col,
                                         method = drift_method,
                                         no_missing_values = 2)['error']
    #running the simulations
    for run in range(repetitions):
        sim_forecast= bs_forecast_values(df = df,
                                         target_col = target_col,
                                         horizon = horizon,
                                         one_step_fcst_errors = drift_errors,
                                         no_missing_values = 2)
        
        forecast_df = pd.concat([forecast_df, pd.Series(sim_forecast, index = forecast_df.index, name=f'run_{run}')],axis=1)

    #finding the mean and quartiles of the simulations for the forecast and prediction intervals
    output_forecast = bs_output(forecast_df, pred_width)

    if simulations:

        return output_forecast, forecast_df
    
    else:
        return output_forecast


def bs_mean_pi(df: pd.DataFrame,target_col:str, horizon=int, repetitions: int = 100,
               pred_width:list=[95,80],simulations:bool=False) -> pd.DataFrame:
    """
    Takes in a dataframe with date-time index and forecast horizon and outputs a data frame with dates continued
    from df and the bootstrapped mean forecast and upper and lower bounds for the prediction intervals as columns.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param repetitions: int - Number of bootstrap repetitions.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
        :param simulations: bool - Toggle whether to additionally return the simulations.
    Output:
        pandas.DataFrame: Bootstrapped mean forecast and prediction intervals for df.
    """

    #creating a dataframe to store the simulated forecasts and finding the errors to randomly sample from
    forecast_df = forecast_dates(df,horizon)
    mean_errors = fitted_forecast_error(df = df,
                                         target_col = target_col,
                                         method = mean_method,
                                         no_missing_values = 1)['error']

    #running the simulations
    for run in range(repetitions):
        sim_forecast= bs_forecast_values(df = df,
                                         target_col = target_col,
                                         horizon = horizon,
                                         one_step_fcst_errors = mean_errors,
                                         no_missing_values = 1)
        
        forecast_df = pd.concat([forecast_df, pd.Series(sim_forecast, index = forecast_df.index, name=f'run_{run}')],axis=1)

    #finding the mean and quartiles of these simulations for the forecast and predictoin intervals.
    output_forecast = bs_output(forecast_df, pred_width)

    if simulations:

        return output_forecast, forecast_df
    
    else:
        return output_forecast
    
    
#to match method names to forecasting functions
bs_forecast_dict = {'naive':bs_naive_pi, 'drift':bs_drift_pi, 'mean':bs_mean_pi}


def bs_benchmark_forecast(df:pd.DataFrame, target_col:str, model:str, horizon:int, period:int=1,
                          repetitions:int=100, pred_width:list = [95,80], simulations:bool = False) -> pd.DataFrame:
    """
    Creates a bootstrapped forecast of the desired model using the forecast functions.
    
    Inputs:
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column with historical data.
        :param model: str - One of {'naive','drift','mean'}, the model to simulate the forecast.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param repetitions: int - Number of bootstrap repetitions.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
        :param simulations: bool - Toggle whether to additionally return the simulations.
    Output:
        pandas.DataFrame: Bootstrapped forecast and prediction intervals for df with the specified model.
    """
    #using the method dictionary to get the forecasting function
    method = bs_forecast_dict[model]

    output_forecast = method(df = df,
                             target_col = target_col,
                             horizon = horizon,
                             period = period,
                             repetitions = repetitions,
                             pred_width = pred_width,
                             simulations = simulations)
    
    return output_forecast
    
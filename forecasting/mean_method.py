import pandas as pd

import numpy as np

from .naive_method import bs_pi_output, pi_output, forecast_dates




##########
#Bootstrap
##########




def mean_method(df:pd.DataFrame,target_col:str,horizon:int, window:int = None) -> np.ndarray:

    """
    Creates a mean forecast in the form of an array.

    Parameters:
        df: pandas.DataFrame - Historical time series data with date-time index.
        target_col: str - column with historical data.
        horizon: int - Number of timesteps forecasted into the future.
        window: int - Number of values before the last data point to take the mean over.
                      If the data is seasonal, use the seasonal period.
                      Defaults to none, in which case all of the data will be used.
            
    Returns:
        numpy.ndarray: Forecasted time series with mean method.
    """

    if not window:
    
        window = len(df)

    
    mean = np.mean(df[target_col].iloc[-window:])


    return np.tile(mean,horizon)




def mean_one_step_forecast(df:np.ndarray, window:int = None) -> np.ndarray:

    """

    Takes in a list or numpy array of data and returns a one step mean forecast for each column of the data.

    Parameters:
        df: numpy.array - An array of data, with each column to be forecasted.
        window: int - Number of values before the last data point to take the mean over.
                      If the data is seasonal, use the seasonal period.
                      Defaults to none, in which case all of the data will be used.

    Returns:
        numpy.ndarray: One step mean forecasts
    """


    if not window:

        window = len(df)



    return np.mean(df[-window:], axis=0)






def mean_fitted_forecast(df:pd.DataFrame, target_col:str, window:int = None) -> pd.Series:

   

    """

    Creates and Returns a fitted forecasting using the drift method.

    Parameters:
        df: pandas.DataFrame - Historical timeseries data with date-time index.
        target_col: str - Column with historical data.
        window: int - Number of values before the last data point to take the mean over.
                      If the data is seasonal, use the seasonal period.
                      Defaults to none, in which case all of the data will be used.

    Returns:
        pandas.Series: One step forecasts with the same date-time index as df


    """

    if not window:


        fitted_values = df[target_col].expanding().mean().shift(1)


    else:


        fitted_values = df[target_col].rolling(window).mean().shift(1)


    fitted_values.name = None

    return fitted_values.dropna()






def mean_single_step_error(df:pd.DataFrame, target_col:str, window:int = None) -> pd.Series:

    """

    Creates a fitted forecasting using the naive method and Returns the error from the data.
    Can be used to create naive bootstrap forecasts.

    Parameters:
        data: pandas.DataFrame - Historical timeseries data with date-time index.
        target_col: str - Column with historical data.
        window: int - Number of values before the last data point to take the mean over.
                      If the data is seasonal, use the seasonal period.
                      Defaults to none, in which case all of the data will be used.

    Returns:
        pandas.Series: One step forecast errors with the same date-time index as df

    """

   
    #calculating each one step forecast
    one_step_forecasts = mean_fitted_forecast(df,target_col, window)


    #calculating the error
    if not window:

        one_step_error = df[target_col].iloc[1:] - one_step_forecasts


    else:

        one_step_error = df[target_col].iloc[window:] - one_step_forecasts




    return one_step_error





def mean_forecaster(df:pd.DataFrame, target_col:str, horizon:int=1, bootstrap_samples: int = 100, window:int = None) -> pd.DataFrame:

    """

    Uses the one step fitted forecast errors to create bootstrap forecasts.

    Parameters:
        df: pandas.DataFrame - Historical timeseries data with date-time index.
        target_col: str - Column with historical data.
        horizon: int - Number of time steps forecasted into the future.
        bootstrap_samples: int - Number of samples outputted.
        window: int - Number of values before the last data point to take the mean over.
                      If the data is seasonal, use the seasonal period.
                      Defaults to none, in which case all of the data will be used.

    Returns:
        pandas.DataFrame: A data frame with date-time index continued from the data and each forecast as columns.

    """

     #finding the fitted forecast errors to randomly sample from
    one_step_errors = mean_single_step_error(df, target_col, window)



    #randomly sampling from the errors
    forecasts = np.random.choice(one_step_errors, size = (horizon, bootstrap_samples))

    
    #manually calculating the first forecast
    first_forecast = mean_one_step_forecast(df[target_col].values, window)
    
    forecasts[0] += first_forecast


    #copying the dataframe to forecast using the randomly sampled errors
    df_copies = pd.concat([df[target_col]] * bootstrap_samples, axis=1)



    #defining a function to iterate over each step in the horizon
    def forecast_function(data):


        appended_data = np.append(df_copies, data[:-1], axis=0)


        return mean_one_step_forecast(appended_data, window) + data[-1]
    



    for index in range(1,horizon):


        forecasts[index] = forecast_function(forecasts[:index+1])



    forecast_ds = forecast_dates(df,horizon)


    output_forecast = pd.DataFrame(forecasts, index=forecast_ds, columns=[f'sim_{i}' for i in range(bootstrap_samples)])



    return output_forecast






def bs_mean_pi(df: pd.DataFrame,target_col:str, horizon: int, window:int = 1, bootstrap_samples: int = 100,
                pred_width:list=[95,80]) -> pd.DataFrame:
    
    """

    Takes in a dataframe with date-time index and forecast horizon and Returns a data frame with dates continued
    from df and the bootstrapped mean forecast and upper and lower bounds for the prediction intervals as columns.

    Parameters:
        df: pandas.DataFrame - Historical time series data.
        target_col: str - Column with historical data.
        horizon: int - Number of timesteps forecasted into the future.
        window: int - Number of values before the last data point to take the mean over.
                      If the data is seasonal, use the seasonal period.
                      Defaults to none, in which case all of the data will be used.
        bootstrap_samples: int - Number of bootstrap repetitions.
        pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
        
    Returns:
        pandas.DataFrame: Bootstrapped mean forecast and prediction intervals for df.
    """


    #simulating the forecasts
    sim_forecasts = mean_forecaster(df = df,
                                    target_col = target_col,
                                    horizon = horizon,
                                    bootstrap_samples = bootstrap_samples,
                                    window = window)
    
    

    #calculating the mean and quantiles of these forecasts or the forecast and prediction intervals
    output_forecast = bs_pi_output(sim_forecasts, pred_width)



    return output_forecast




#######
#Normal
#######





def mean_pi(df:pd.DataFrame,target_col:str, horizon:int, window:int = 1, pred_width:list = [95,80]) -> pd.DataFrame:

    """

    Takes in a dataframe with date-time index and forecast horizon and Returns a data frame with dates continued
    from df, with the mean forecast and upper and lower bounds for the prediction intervals as columns.

    Parameters:
        df: pandas.DataFrame - Historical time series data with dates as index.
        target_col: str - Column with historical data.
        horizon: int - Number of timesteps forecasted into the future.
        window: int - Number of values before the last data point to take the mean over.
                      If the data is seasonal, use the seasonal period.
                      Defaults to none, in which case all of the data will be used.
        pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                  If none are needed, set to None

    Returns:
        pandas.DataFrame: Mean forecast and prediction intervals for df.

    """



    #extending the dates from df and storing the forecast in forecast_df
    forecast_ds = forecast_dates(df,horizon)


    forecast_df = pd.DataFrame(mean_method(df = df,
                                           target_col = target_col,
                                           horizon = horizon,
                                           window = window),
                               index = forecast_ds,
                               columns = ['forecast'])



    if pred_width:

        #calculating the errors from the fitted forecast to calculate the residuals
        mean_errors = mean_fitted_forecast(df = df,
                                           target_col = target_col,
                                           window = window)
        
        if not window:

            window = len(df)



        #calculating the standard deviation of the residuals, with one degree of freedom as we haveter
        sd_residuls = np.std(mean_errors,ddof=1)

        forecast_sd = np.tile(sd_residuls * np.sqrt(1 + 1/window), horizon)



        #outputting the result
        output_forecast = pi_output(forecast_df, forecast_sd, pred_width)



        return output_forecast
    

    
    else:



        return forecast_df
    


class Mean_forecast:

    """

    Creates bootstrap or normal forecast and prediction intervals using the drift method.

    Attributes:
        window: int - Number of values before the last data point to take the mean over.
                      If the data is seasonal, use the seasonal period.
                      Defaults to none, in which case all of the data will be used.
        pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                            If none are needed, set to None.
        bootstrap: bool - Toggle whether to simulate forecasts.
        bootstrap_samples: int - Number of bootstrap repetitions.
        df: pandas.DataFrame - Historical timeseries data with a date-time index.
        target_col: str - Column with historical data.

    -------
    Methods
    -------
        fit(df:pandas.DataFrame, target_col:str) -
            Fits the inputted data to the drift model.

        predict(horizon : int = None) - 
            Creates a forecast and prediction intervals for the data fitted in Drift_forecast.fit() using the naive method.

    """


    def __init__(self, window:int=None, pred_width:list = [95,80],
                 bootstrap:bool = False, bootstrap_samples:int = 100):

        """

        Parameters:              
            window: int - Number of values before the last data point to take the mean over.
                                If the data is seasonal, use the seasonal period.
                                Defaults to none, in which case all of the data will be used.         
            pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                                If none are needed, set to None        
            bootstrap: bool - Toggle whether to simulate forecasts      
            bootstrap_samples: int - Number of bootstrap repetitions.

        """

        self.window = window
                
        self.pred_width = pred_width
        
        self.bootstrap = bootstrap
        
        self.bootstrap_samples = bootstrap_samples




    def fit(self, df:pd.DataFrame, target_col:str):
        
        """

        Fits the inputted data to the mean model.

        Parameters:    
            df: pandas.DataFrame - Historical time series data with date-time index.         
            target_col: str - Column with historical data.
        """


        self.data = df
        
        self.target_col = target_col




    def predict(self, horizon:int = None) -> pd.DataFrame:

        """

        Creates a forecast and prediction intervals for the data fitted in Mean_forecast.fit() using the mean method.
        
        Parameters:  
            horizon: int - Number of timesteps forecasted into the future.
                                  Defaults to None, in which case a fitted forecast is outputted.
        
        Returns: 
            pandas.DataFrame: A forecast or fitted forecast for the inputted data using the mean method.
        """

        if not horizon:


            fitted_forecast = mean_fitted_forecast(df = self.data,
                                                   target_col = self.target_col,
                                                   window = self.window)

            
            fitted_forecast = fitted_forecast.to_frame(name = 'fitted forecast')
            

            return fitted_forecast
        

        
        else:



            if self.bootstrap:



                output_forecast = bs_mean_pi(df = self.data,
                                             target_col = self.target_col,
                                             horizon = horizon,
                                             window = self.window,
                                             bootstrap_samples= self.bootstrap_samples,
                                             pred_width = self.pred_width)
                

            else:

                output_forecast = mean_pi(df = self.data,
                                          target_col = self.target_col,
                                          horizon = horizon,
                                          window = self.window,
                                          pred_width = self.pred_width)
                


            return output_forecast

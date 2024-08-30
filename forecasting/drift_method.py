import pandas as pd
import numpy as np
from .naive_method import bs_pi_output, pi_output, forecast_dates



##########
#Bootstrap
##########




def drift_method(df:pd.DataFrame, target_col:str, horizon:int) -> np.ndarray:

    """

    Creates a drift forecast in the form of an array.

    Parameters:
        df: pandas.DataFrame - Historical time series data with date-time index.
        target_col: str - column with historical data.
        horizon: int - Number of timesteps forecasted into the future.
  
    Returns:
        numpy.ndarray: Forecasted time series with drift method.
    """


    latest_obs = df[target_col].iloc[-1]

    first_obs = df[target_col].iloc[0]


    slope = (latest_obs - first_obs) / (len(df) - 1)


    forecast_list = np.array([latest_obs + slope * h for h in range(1, horizon + 1)])



    return forecast_list




def drift_one_step_forecast(df:np.ndarray) -> np.ndarray:

    """

    Takes in a list or numpy array of data and returns a one step drift forecast for each column of the data.

    Parameters:
        df: numpy.ndarray - An array of data, with each column to be forecasted.

    Returns:
        numpy.ndarray: One step drift forecasts
    """


    return df[-1] + (df[-1] - df[0]) / (len(df)-1)






def drift_fitted_forecast(df:pd.DataFrame, target_col:str) -> pd.Series:


    """

    Creates and Returns a fitted forecasting using the drift method.

    Parameters:
        df: pandas.DataFrame - Historical timeseries data with date-time index.
        target_col: str - Column with historical data.

    Returns:
        pandas.Series: One step forecasts with the same date-time index as df

    """

    quotient = np.arange(1,len(df)-1)

    first_value = df[target_col].iloc[0]

    slopes = (df[target_col].iloc[1:len(df)-1].values - first_value) / quotient

    fitted_values = df[target_col].iloc[1:len(df)-1].values + slopes
    
    output = pd.Series(fitted_values, index = df.iloc[2:].index)
   


    return output






def drift_single_step_error(df:pd.DataFrame,target_col:str) -> pd.Series:

    """

    Creates a fitted forecasting using the drift method and outputs the error from the data.
    Can be used to create naive bootstrap forecasts.  

    Parameters:
        df: pandas.DataFrame - Historical time series data with date-time index.
        target_col: str - Column with historical data.
            
    Returns:
        pandas.Series: One step forecast errors with the same date-time index as df
    """


    fitted_values = drift_fitted_forecast(df,target_col)



    return df[target_col].iloc[2:] - fitted_values





def drift_forecaster(df:pd.DataFrame, target_col:str, horizon:int=1, bootstrap_samples: int = 100) -> pd.DataFrame:

 

    """

    Uses the one step fitted forecast errors to create bootstrap forecasts.

    Parameters:
        df: pandas.DataFrame - Historical timeseries data with date-time index.
        target_col: str - Column with historical data.
        horizon: int - Number of time steps forecasted into the future.
        bootstrap_samples: int - Number of samples outputted.

    Returns:
        pandas.DataFrame: A dataframe with date-time index continued from the data and each forecast as columns.

    """

    #finding the fitted forecast errors to randomly sample from
    one_step_errors = drift_single_step_error(df, target_col).values


    #randomly sampling from these errors
    forecasts = np.random.choice(one_step_errors, size = (horizon, bootstrap_samples))


    #manually calculating the first forecast
    first_forecast = drift_one_step_forecast(df[target_col].values)

    forecasts[0] += first_forecast

                                            

    #copying the dataframe to forecast using the randomly sampled errors
    df_copies = pd.concat([df[target_col]] * bootstrap_samples, axis=1)



    #defining a function to iterate over each step in the horizon
    #each time calculating a one step drift forecast and adding the randomly sampled error

    def forecast_function(data):


        appended_data = np.append(df_copies, data[:-1], axis=0)


        return drift_one_step_forecast(appended_data) + data[-1]
    



    for index in range(1,horizon):


        forecasts[index] = forecast_function(forecasts[:index+1])



    #storing the samples in a dataframe with dates continued from df
    forecast_ds = forecast_dates(df,horizon)


    output_forecast = pd.DataFrame(forecasts, index=forecast_ds, columns=[f'sim_{i}' for i in range(bootstrap_samples)])



    return output_forecast




def bs_drift_pi(df: pd.DataFrame,target_col:str, horizon: int, bootstrap_samples: int = 100,
                pred_width:list=[95,80]) -> pd.DataFrame:
    
    """

    Takes in a dataframe with date-time index and forecast horizon and Returns a data frame with dates continued
    from df and the bootstrapped drift forecast and upper and lower bounds for the prediction intervals as columns.


    Parameters:
        df: pandas.DataFrame - Historical time series data.
        target_col: str - Column with historical data.
        horizon: int - Number of timesteps forecasted into the future.
        bootstrap_samples: int - Number of bootstrap repetitions.
        pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.

    Returns:
        pandas.DataFrame: Bootstrapped drift forecast and prediction intervals for df.
    """


    #simulating the forecasts
    sim_forecasts = drift_forecaster(df = df,
                                     target_col = target_col,
                                     horizon = horizon,
                                     bootstrap_samples = bootstrap_samples)
    
    

    #calculating the mean and quantiles of these forecasts or the forecast and prediction intervals
    output_forecast = bs_pi_output(sim_forecasts, pred_width)



    return output_forecast




def drift_pi(df:pd.DataFrame,target_col:str,horizon:int, pred_width:list = [95,80]) -> pd.DataFrame:

    """

    Takes in a dataframe with date-time index and forecast horizon and Returns a data frame with dates continued
    from df, with the drift forecast and upper and lower bounds for the prediction intervals as columns.

    Parameters:
        df: pandas.DataFrame - Historical time series data with dates as index.
        target_col: str - Column with historical data.
        horizon: int - Number of timesteps forecasted into the future.
        pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                  If none are needed, set to None.

    Output:

        pandas.DataFrame: Drift forecast and prediction intervals for df.
    """


    #extending the dates from df and storing the forecast in forecast_df

    forecast_ds = forecast_dates(df,horizon)


    forecast_df = pd.DataFrame(drift_method(df = df,
                                            target_col = target_col,
                                            horizon = horizon),
                               index = forecast_ds,
                               columns = ['forecast'])


    if pred_width:
        #calculating the errors from the fitted forecast to calculate the residuals
        drift_errors = drift_fitted_forecast(df = df,
                                             target_col = target_col)
        

        #calculating the standard deviation of the residuals, with one degree of freedom as we haveter
        sd_residuals = np.std(drift_errors, ddof = 1)



        #calculating the standard deviation for the forecasted points
        forecast_sd = np.array([sd_residuals * np.sqrt(i * (1 + i/(len(df)-1))) for i in range(1,horizon+1)])



        #outputting the result
        output_forecast = pi_output(forecast_df,forecast_sd,pred_width)


        
        return output_forecast
    
    
    else:


        return forecast_df
    




class Drift_forecast:

    """

    Creates bootstrap or normal forecast and prediction intervals using the drift method.

    Attributes:
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

    def __init__(self, pred_width:list = [95,80],
                 bootstrap:bool = False, bootstrap_samples:int = 100):

        """
        Parameters:             
            pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                             If none are needed, set to None
            bootstrap: bool - Toggle whether to simulate forecasts
            bootstrap_samples: int - Number of bootstrap repetitions.

        """

        
        self.pred_width = pred_width
        
        self.bootstrap = bootstrap
        
        self.bootstrap_samples = bootstrap_samples




    def fit(self, df:pd.DataFrame, target_col:str):
        
        """

        Fits the inputted data to the drift model.

        Parameters:
            df: pandas.DataFrame - Historical time series data with date-time index.
            target_col: str - Column with historical data.
        """


        self.data = df
        
        self.target_col = target_col




    def predict(self, horizon:int = None) -> pd.DataFrame:

        """

        Creates a forecast and prediction intervals for the data fitted in Drift_forecast.fit() using the drift method.
        
        Parameters:
            horizon: int - Number of timesteps forecasted into the future.
                                  Defaults to None, in which case a fitted forecast is outputted.
        
        Returns:
            pandas.DataFrame: A forecast or fitted forecast for the inputted data using the drift method.
        """

        if not horizon:


            fitted_forecast = drift_fitted_forecast(df = self.data,
                                                    target_col = self.target_col)

            
            fitted_forecast = fitted_forecast.to_frame(name = 'fitted forecast')

            

            return fitted_forecast
        

        
        
        else:



            if self.bootstrap:



                output_forecast = bs_drift_pi(df = self.data,
                                              target_col = self.target_col,
                                              horizon = horizon,
                                              bootstrap_samples= self.bootstrap_samples,
                                              pred_width = self.pred_width)
                

            else:

                output_forecast = drift_pi(df = self.data,
                                           target_col = self.target_col,
                                           horizon = horizon,
                                           pred_width = self.pred_width)
                

                
            return output_forecast












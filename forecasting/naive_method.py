import pandas as pd
import numpy as np
from scipy.stats import norm



##########
#Bootstrap
##########



def naive_method(df:pd.DataFrame, target_col:str, horizon:int, period:int=1) -> np.ndarray:
    
    """

    Creates a naive forecast in the from of an array.

    Parameters:
        df: pandas.DataFrame - Historical time series data with date-time index.
        target_col: str - Column with historical data.
        horizon: int - Number of timesteps forecasted into the future.
        period: int - Seasonal period.

    Returns:
        numpy.ndarray: Forecasted time series with naive or seasonal naive method.

    """


    most_recent_values = df[target_col].iloc[-period:].values


    reps = int(np.ceil(horizon / period))


    return np.tile(most_recent_values,reps)[:horizon]




 

def forecast_dates(df:pd.DataFrame, horizon:int) -> pd.DataFrame :

    """

    Extends the dates of the date-time index of df until the horizon.

    Parameters:
        df: pandas.DataFrame - Historical time series data with date-time index.
        horizon: int - Number of timesteps forecasted into the future.

    Returns:
        pandas.DataFrame: A data frame with dates continued from df to the forecast horizon.

    """


    ds = pd.to_datetime(df.index)


    forecast_ds = pd.date_range(start = ds[-1], periods = horizon+1, freq = ds.freq)

   
    return forecast_ds[1:]





def naive_fitted_forecast(df:pd.DataFrame, target_col:str, period:int = 1) -> pd.Series:

   

    """

    Creates and outputs a fitted forecasting using the naive method.

    Parameters:
        df: pandas.DataFrame - Historical timeseries data with date-time index.
        target_col: str - Column with historical data.
        period: int - Seasonal period.

    Returns:
        numpy.ndarray: An array of the one step forecast errors.

    """
  
   
    return df[target_col].shift(period).dropna()






 
def naive_single_step_error(df:pd.DataFrame, target_col:str, period:int = 1) -> pd.Series:

   

    """

    Creates a fitted forecast using the naive method and outputs the error from the data.
    Can be used to create naive bootstrap forecasts.

    Parameters:
        df: pandas.DataFrame - Historical timeseries data with date-time index.
        target_col: str - Column with historical data.
        period: int - Seasonal period.

    Returns:
        numpy.ndarray: An array of the one step forecast errors.

    """

   
    one_step_error = df[target_col][period:] - naive_fitted_forecast(df,target_col,period)

   
    return one_step_error

 

 

 

def naive_forecaster(df:pd.DataFrame, target_col:str, horizon:int=1, period:int = 1, bootstrap_samples: int = 100) -> pd.DataFrame:

 

    """

    Uses the one step fitted forecast errors to create bootstrap forecasts.

    Parameters:
        df: pandas.DataFrame - Historical timeseries data with date-time index.
        target_col: str - Column with historical data.
        horizon: int - Number of time steps forecasted into the future.
        period: int - Seasonal period.
        bootstrap_samples: int - Number of samples outputted.

    Returns:
        pandas.DataFrame: A data frame with date-time index continued from the data and each forecast as columns.

    """

 
    one_step_error = naive_single_step_error(df, target_col, period).values

   

    forecast_ds = forecast_dates(df, horizon)

 

    # Precompute random choices

    forecasts = np.random.choice(one_step_error, size=(horizon, bootstrap_samples))
 


    # Generate forecast using vectorized operations, treating each period stage seperately

    last_values = df[target_col].iloc[-period:]

    for i in range(period):


        forecasts[i: :period] = np.cumsum(forecasts[i: :period], axis=0) + last_values.iloc[i]

   

    # Convert the matrix to a DataFrame with appropriate index

    output_forecast = pd.DataFrame(forecasts, index=forecast_ds, columns=[f'sim_{i}' for i in range(bootstrap_samples)])

   

    return output_forecast







def bs_pi_output(forecast_data:pd.DataFrame, pred_width:list = [95,80]) -> pd.DataFrame:
    
    """

    Uses a data frame with continued dates as an index and the bootstrap forecasts as columns
    and calculates the mean and quartiles of each step of the forecast.
    These are the forecast and the upper and lower bounds of the prediction intervals for the data.

    Parameters:
        forecast_data: pandas.DataFrame - Data frame of simulated forecasts.
                                               to calculate the forecast and prediction intervals from.
        pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
    
    Returns:      
        pandas.DataFrame: Data frame with the forecasted dates as the index,
                          and the lower and upper bounds for the prediction intervals as columns.
    """

    #storing the mean and quantiles for each forecast point in columns
    output_forecast = pd.DataFrame(forecast_data.mean(axis=1), columns=['forecast'])



    #sorting the widths in reverse order

    if pred_width:

        pred_width = np.sort(pred_width)


        pred_width = reversed(pred_width)



        #iterating through the prediction widths and calculating the corresponding intervals
        for width in pred_width:


            new_pred_width = (100 - (100-width)/2) / 100 



            output_forecast[f'{width}% lower_pi'] = forecast_data.quantile(1 - new_pred_width, axis=1)

            output_forecast[f'{width}% upper_pi'] = forecast_data.quantile(new_pred_width, axis=1)



    return output_forecast






def bs_naive_pi(df: pd.DataFrame, target_col:str, horizon: int, period: int=1, bootstrap_samples:int=100,
                pred_width:list=[95,80]) -> pd.DataFrame:
    
    """

    Takes in a dataframe with date-time index and forecast horizon and Returns a data frame with dates continued
    from df and the bootstrapped naive forecast and upper and lower bounds for the prediction intervals as columns.

    Parameters:
        df: pandas.DataFrame - Historical time series data.
        target_col: str - Column with historical data.
        horizon: int - Number of timesteps forecasted into the future.
        period: int - Seasonal period.
        bootstrap_samples: int - Number of bootstrap repetitions.
        pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
           
    Returns:
        pandas.DataFrame: Bootstrapped naive or seasonal naive forecast and prediction intervals for df.
        
    """

    #calculating the bootstrap forecasts
    forecast_data = naive_forecaster(df=df,
                                     target_col=target_col,
                                     horizon=horizon,
                                     period=period,
                                     bootstrap_samples=bootstrap_samples)
    

    
    output_forecast = bs_pi_output(forecast_data=forecast_data,
                                   pred_width=pred_width)
    


    return output_forecast



#######
#Normal
#######





def pi_output(forecast_df:pd.DataFrame, forecast_sd:np.array, pred_width:list = [95,80]) -> pd.DataFrame:

    """

    Uses the forecast with extended dates and standard deviation in each step of the forecast
    to calculate the prediction intervals for the forecast.

    The standard deviation for each forecast step is calculated using the standard deviation of the residuals
    for the model forecast.

    Parameters:
        forecast_df: pandas.DataFrame - Data frame with extended dates and forecasted points.
        forecast_sd: numpy.ndarray - Multi-step standard deviation from the most recent observation for each forecasted point.
        pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
    
    Returns:
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


        output_forecast[f'{width}% lower_pi'] = forecast_df['forecast'].values - pi_mult * forecast_sd

        output_forecast[f'{width}% upper_pi'] = forecast_df['forecast'].values + pi_mult * forecast_sd

    return output_forecast







def naive_pi(df:pd.DataFrame, target_col:str, horizon:int, period:int=1, pred_width:list = [95,80]) -> pd.DataFrame:
   
    """

    Takes in a dataframe with date-time index and forecast horizon and Returns a data frame with dates continued
    from df, with the naive forecast and upper and lower bounds for the prediction intervals as columns.

    Parameters:
        df: pandas.DataFrame - Historical time series data with dates as index.
        target_col: str - Column with historical data.
        horizon: int - Number of timesteps forecasted into the future.
        period: int - Seasonal period.
        pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                         If none are needed, set to None.
    Returns:
        pandas.DataFrame: Naive or seasonal naive forecast and prediction intervals for df.

    """

    #extending the dates from df and storing the forecast in forecast_df

    forecast_ds = forecast_dates(df,horizon)


    forecast_df = pd.DataFrame(naive_method(df = df,
                                            target_col = target_col,
                                            horizon = horizon,
                                            period = period),
                               index = forecast_ds,
                               columns = ['forecast'])

    if pred_width:


        #calculating the errors from the fitted forecast to calculate the residuals
        naive_errors = naive_fitted_forecast(df = df,
                                             target_col = target_col,
                                             period = period)
        


        #calculating the standard deviation of the residuals, removing the first seasonal period as we cannot forecast this using this model 
        sd_residuals = np.std(naive_errors)



        #calculating the forecast standard deviation
        seasons_in_forecast = [int((h-1) / period) for h in range(1,horizon+1)]

        forecast_sd = np.array([sd_residuals * np.sqrt(seasons_in_forecast[h]+1) for h in range(horizon)])



        output_forecast = pi_output(forecast_df,forecast_sd, pred_width)



        return output_forecast
    
    
    else:


        return forecast_df
    






class Naive_forecast:

    """

    Creates bootstrap or normal forecast and prediction intervals using the naive method.

    Attributes:
        period: int - Seasonal period.
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
            Fits the inputted data to the naive model.

        predict(horizon : int = None) - 
            Creates a forecast and prediction intervals for the data fitted in Naive_forecast.fit() using the naive method.

    """

    def __init__(self, period:int=1, pred_width:list = [95,80],
                 bootstrap:bool = False, bootstrap_samples:int = 100):

        """
        Parameters:
            period: int - Seasonal period.
            pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                             If none are needed, set to None.
            bootstrap: bool - Toggle whether to simulate forecasts.
            bootstrap_samples: int - Number of bootstrap repetitions.

        """

        self.period = period
                
        self.pred_width = pred_width
        
        self.bootstrap = bootstrap
        
        self.bootstrap_samples = bootstrap_samples




    def fit(self, df:pd.DataFrame, target_col:str):
        
        """

        Fits the inputted data to the naive model.

        Parameters:
            df: pandas.DataFrame - Historical time series data with date-time index.
            target_col: str - Column with historical data.

        """


        self.data = df
        
        self.target_col = target_col




    def predict(self, horizon:int = None) -> pd.DataFrame:

        """

        Creates a forecast and prediction intervals for the data fitted in Naive_forecast.fit() using the naive method.
        
        Parameters:
            horizon: int - Number of timesteps forecasted into the future.
                                  Defaults to None, in which case a fitted forecast is outputted.
        
        Returns:
            pandas.DataFrame: A forecast or fitted forecast for the inputted data using the naive method.
        """

        if not horizon:


            fitted_forecast = naive_fitted_forecast(df = self.data,
                                                    target_col = self.target_col,
                                                    period = self.period)

            
            fitted_forecast = fitted_forecast.to_frame(name = 'fitted forecast')

            

            return fitted_forecast
        

        

        else:



            if self.bootstrap:



                output_forecast = bs_naive_pi(df = self.data,
                                              target_col = self.target_col,
                                              horizon = horizon,
                                              period = self.period,
                                              bootstrap_samples= self.bootstrap_samples,
                                              pred_width = self.pred_width)
                

            else:

                output_forecast = naive_pi(df = self.data,
                                           target_col = self.target_col,
                                           horizon = horizon,
                                           period = self.period,
                                           pred_width = self.pred_width)
                

                
            return output_forecast

    






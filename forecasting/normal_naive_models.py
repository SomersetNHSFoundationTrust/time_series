import pandas as pd
import numpy as np
from .bootstrap_naive_models import naive_method,  drift_method, mean_method, fitted_forecast_error, forecast_dates, bs_benchmark_forecast
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





def naive_pi(df:pd.DataFrame, target_col:str, horizon:int, period:int=1, pred_width:list = [95,80]) -> pd.DataFrame:
   
    """
    Takes in a dataframe with date-time index and forecast horizon and outputs a data frame with dates continued
    from df, with the naive forecast and upper and lower bounds for the prediction intervals as columns.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data with dates as index.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                         If none are needed, set to None
    Output:
        pandas.DataFrame: Naive or seasonal naive forecast and prediction intervals for df.
    """

    #extending the dates from df and storing the forecast in forecast_df
    forecast_df = forecast_dates(df,horizon)
    forecast_df['forecast'] = naive_method(df,target_col, horizon,period)

    if pred_width:

        #calculating the errors from the fitted forecast to calculate the residuals
        naive_errors = fitted_forecast_error(df = df,
                                             target_col = target_col,
                                             method = naive_method,
                                             no_missing_values = 2,
                                             period = period)['error']

        #calculating the standard deviation of the residuals, removing the first seasonal period as we cannot forecast this using this model 
        sd_residuals = np.std(naive_errors)

        #calculating the forecast standard deviation
        seasons_in_forecast = [int((h-1) / period) for h in range(1,horizon+1)] 
        forecast_sd = [sd_residuals * np.sqrt(seasons_in_forecast[h]+1) for h in range(horizon)]

        output_forecast = pi_output(forecast_df,horizon,forecast_sd, pred_width)

        return output_forecast
    
    else:

        return forecast_df




def drift_pi(df:pd.DataFrame,target_col:str,horizon:int, period:int = 1, pred_width:list = [95,80]) -> pd.DataFrame:
    """
    Takes in a dataframe with date-time index and forecast horizon and outputs a data frame with dates continued
    from df, with the drift forecast and upper and lower bounds for the prediction intervals as columns.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data with dates as index.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                         If none are needed, set to None
    Output:
        pandas.DataFrame: Drift forecast and prediction intervals for df.
    """
    #extending the dates from df and storing the forecast in forecast_df
    forecast_df = forecast_dates(df,horizon)
    forecast_df['forecast'] = drift_method(df,target_col,horizon)


    if pred_width:
        #calculating the errors from the fitted forecast to calculate the residuals
        drift_errors = fitted_forecast_error(df = df,
                                             target_col = target_col,
                                             method = drift_method,
                                             no_missing_values = 2)['error']

        #calculating the standard deviation of the residuals, with one degree of freedom as we have a parameter
        sd_residuals = np.std(drift_errors,ddof = 1)

        #calculating the standard deviation for the forecasted points
        forecast_sd = [sd_residuals * np.sqrt(i * (1 + i/(len(df)-1))) for i in range(1,horizon+1)]

        #outputting the result
        output_forecast = pi_output(forecast_df,horizon,forecast_sd,pred_width)
        
        return output_forecast
    
    else:

        return forecast_df



def mean_pi(df:pd.DataFrame,target_col:str, horizon:int, period:int = 1, pred_width:list = [95,80]) -> pd.DataFrame:
    """
    Takes in a dataframe with date-time index and forecast horizon and outputs a data frame with dates continued
    from df, with the mean forecast and upper and lower bounds for the prediction intervals as columns.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data with dates as index.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                         If none are needed, set to None
    Output:
        pandas.DataFrame: Mean forecast and prediction intervals for df.
    """

    #extending the dates from df and storing the forecast in forecast_df
    forecast_df = forecast_dates(df,horizon)
    forecast_df['forecast'] = mean_method(df,target_col,horizon)

    if pred_width:

        #calculating the errors from the fitted forecast to calculate the residuals
        mean_errors = fitted_forecast_error(df = df,
                                            target_col = target_col,
                                            method = mean_method,
                                            no_missing_values = 1)['error']

        #calculating the standard deviation of the residuals, with one degree of freedom as we have a parameter
        sd_residuls = np.std(mean_errors,ddof=1)
        forecast_sd = [sd_residuls * np.sqrt(1 + 1/len(df))] * horizon

        #outputting the result
        output_forecast = pi_output(forecast_df,horizon,forecast_sd, pred_width)

        return output_forecast
    
    else:

        return forecast_df


# to calculate fitted forecasts using fitted_forecast_error from bootstrap_naive-models.py
naive_method_dict = {'naive':naive_method, 'drift':drift_method, 'mean':mean_method}
#to calculate forecast and prediction intervals
naive_forecast_dict = {'naive':naive_pi, 'drift':drift_pi, 'mean':mean_pi}


class Benchmark_forecast:
    """
    A class to bring together the naive models to use in the Forecast class
    """

    def __init__(self, model:str, period:int=1, pred_width:list = [95,80],
                 bootstrap:bool = False, repetitions:int = 100):

        """
        Inputs:
            :param model: str - Desired model to use, one of {'naive', 'drift', 'mean'}.
            :param period: int - Seasonal period.
            :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                             If none are needed, set to None
            :param bootstrap: bool - Toggle whether to simulate forecasts
            :param simulations:int - Number of bootstrap repetitions.
        """

        self.period = period
        self.model = model
        self.pred_width = pred_width
        self.bootstrap = bootstrap
        self.repetitions = repetitions

    def fit(self, df:pd.DataFrame, target_col:str):
        
        """
        Fits the inputted data to the desired model.

        Inputs:
            :param df: pandas.DataFrame - Historical time series data with date-time index.
            :param target_col: str - Column with historical data.
        """

        self.data = df
        self.target_col = target_col

    def predict(self, horizon:int = None) -> pd.DataFrame:
        """
        Creates a forecast and predictino intervals for the data fitted in Benchmark_forecast.fit() using the model provided.
        Inputs:
            :param horizon: int - Number of timesteps forecasted into the future.
                                  Defaults to None, in which case a fitted forecast is outputted.
        Outputs:
            pd.DataFrame - A forecast or fitted forecast for the inputted data using the inputted naive model.
        """

        if not horizon:

            method = naive_method_dict[self.model]

            if self.model == 'drift':
                no_missing_values = 2
            else:
                no_missing_values = self.period

            fitted_forecast = pd.DataFrame(index = self.data.index)
            fitted_forecast['fitted forecast'] = fitted_forecast_error(df = self.data,
                                                                       target_col = self.target_col,
                                                                       method = method,
                                                                       no_missing_values = no_missing_values,
                                                                       period = self.period)['fitted forecast']

            return fitted_forecast
        
        else:

            if self.bootstrap:

                output_forecast = bs_benchmark_forecast(df = self.data,
                                                        target_col = self.target_col,
                                                        model = self.model,
                                                        horizon = horizon,
                                                        period = self.period,
                                                        repetitions = self.repetitions,
                                                        pred_width = self.pred_width)
                

            else:

                forecaster =  naive_forecast_dict[self.model]

                output_forecast = forecaster(df = self.data,
                                             target_col = self.target_col,
                                             horizon = horizon,
                                             period = self.period,
                                             pred_width = self.pred_width)
                
            return output_forecast




    

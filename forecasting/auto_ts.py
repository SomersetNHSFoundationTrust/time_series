from forecasting.model_selection import cross_val
from forecasting.more_models import *
from forecasting.graphs import resid_diagnostic, future_forecast,cross_val_graph
from forecasting.normal_naive_models import naive_pi, drift_pi, mean_pi
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import *




class Forecast:

    def __init__(self, data:pd.DataFrame,target_col:str=None, period:int=1):
        
        """
        :param data: pandas.DataFrame - Historical time series data with a date-time index.
        :param target_col: str - Column with historical data. If none will default to the first column.
        :param period: int - Seasonal period.
        """

        self.data = data
        self.period = period

        if not target_col:
            target_col = self.data.columns[0]

        self.target_col = target_col

    def select_best_model(self, n_splits:int=5, test_size:int=None, models:dict=None, time_taken:bool=False):

        """
        Cross validates the models specified.

        
        The model that minimises the mean squared error will be set as the attribute best_model.

        Outputs a data frame that shows the values of the mean absolute error, mean absolute percentage error,
        mean squared error and the max error for the forecast of each model and the observed data.

        Inputs:

            :param n_splits: int - Number of folds.
            :param test_size: int - Forecast horizon during each fold.
            :param models: dict - A dictionary with the models (str) as keys
                                  and their respective forecast functions as values.
            :param time_taken: bool - Toggle woether to print the time taken for each fold of each model

        Outputs:
            pandas.DataFrame: A summary data frame of the metric values for each forecast model.

        """

        #defining a model dictionary to default to
        model_dict = {'naive':naive_pi, 'drift':drift_pi, 'mean':mean_pi,
                      'ETS':ETS_forecast, 'ARIMA':ARIMA_forecast, 'prophet':prophet_forecast}

        if not models:
            models = model_dict

        #performing cross validation on self.data
        cross_val_frame = cross_val(df=self.data,
                                    target_col=self.target_col,
                                    period=self.period,
                                    n_splits=n_splits,
                                    test_size=test_size,
                                    models=models,
                                    time_taken=time_taken)
        
        #setting the index of the data the same as the cross validation
        first_model = cross_val_frame.index[0][0]
        cross_val_index = cross_val_frame.loc[first_model].index

        obs_data = self.data[self.target_col][cross_val_index]

        #grouping the cross validation data and calculating the metrics for each of the model forecasts
        grouped_frame = cross_val_frame.groupby(by='model')

        output_frame = grouped_frame.agg(mean_absolute_error = ('forecast', lambda x: mean_absolute_error(obs_data,x)),
                                         mean_absolute_percentage_error = ('forecast', lambda x: mean_absolute_percentage_error(obs_data,x)),
                                         mean_squared_error = ('forecast', lambda x: mean_squared_error(obs_data,x)),
                                         max_error = ('forecast', lambda x: max_error(obs_data,x))
                                        )
        
        #setting the model that minimises the mean squared error to self.best_model
        self.best_model = output_frame['mean_squared_error'].idxmin()
        
        return output_frame
    
    
    def plot_diagnostics(self, model:str=None) -> go.Figure:

        """
        A summary of the residual diagnostics.

        This includes a plot of the residuals and their mean value,
        the Autocorrelation funciton (ACF) and its 95% bounds and a histogram
        of the residuals with a theoretical normal distribution.

        The ACF is a collection of the autocorrelation coefficients
        between the residuals and the lagged or shifted residuals.
        A 'good' forecast should produce ACF that is similar to white noise
        or is random, to ensure all information is captured by the forecast.

        Inputs
            :param model: str - Model used to perform the residual diagnostics, a key from model_dict
                                If left None this will default to best_model.
        Outputs:
            go.Figure - Subplots of the residuals, the ACF and a histogram of the residuals.
        """

        if not model:
            model = self.best_model

        return resid_diagnostic(df = self.data,
                                target_col=self.target_col,
                                model = model,
                                period = self.period)
    
    def forecast(self, horizon:int, model:str=None) -> pd.DataFrame:

        """ 
        Creates a summary data frame of the forecast using the model specified.
        The continued dates from the data are the index, and the forecast and 95% and 80%
        prediction intervals are columns.

        Inputs:
            :param horizon: The number of timesteps forecasted into the future.
            :param model: str - Model used to perform the residual diagnostics, a key from model_dict
                                If left None this will default to best_model.
        Ouputs:
            pandas.DataFrame - A forecast using the model specified.

        """

        if not model:
            model = self.best_model

        self.horizon=horizon

        return benchmark_forecast(df = self.data,
                                  target_col=self.target_col,
                                  horizon = horizon,
                                  period = self.period,
                                  model = model,
                                  pred_width=[95,80])
    
    def plot_forecast(self, horizon:int, model:str=None) -> go.Figure:

        """ 
        Creates a plot of the data, the forecast of the desired model
        and the 95% and 80% prediction intervals

        Inputs:
            :param horizon: The number of timesteps forecasted into the future.
            :param model: str - Model used to perform the residual diagnostics, a key from model_dict
                                If left None this will default to best_model.
        Ouputs:
            go.Figure: A plot of the observed data, the forecast and 95% adn 80% prediction intervals.

        """

        if not model:
            model = self.best_model

        return future_forecast(df = self.data,
                               target_col=self.target_col,
                               period = self.period,
                               model = model,
                               horizon=horizon)
    
    
    def plot_cross_validation(self, n_splits:int = 5, test_size:int = None, models:dict=None) -> go.Figure:

        """ 
        Cross validates the models specified and plots the results, indicating where each fold is located.

        Inputs:
            :param n_splits: int - Number of folds.
            :param test_size: int - Forecast horizon during each fold.
            :param models: dict - A dictionary with the models (str) as keys
                                  and their respective forecast functions as values.
        Outputs:
            go.Figure - A plot of the cross validation forecast and observed data.
        """

        model_dict = {'naive':naive_pi, 'drift':drift_pi, 'mean':mean_pi,
                      'prophet':prophet_forecast}

        if not models:
            models = model_dict

        return cross_val_graph(df = self.data,
                               target_col=self.target_col,
                               period = self.period,
                               n_splits=n_splits,
                               test_size=test_size,
                               models=models)


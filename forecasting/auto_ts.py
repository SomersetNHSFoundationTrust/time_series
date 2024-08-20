from forecasting.model_selection import cross_val
from forecasting.more_models import *
from forecasting.graphs import resid_diagnostic, future_forecast, cross_val_graph
from forecasting.normal_naive_models import Benchmark_forecast
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import *




class Forecast:

    """
    Forecasting class

    ----------
    Attributes
    ----------

    data: pandas.DataFrame - The data to forecast.
                                    Must have an index of dates which the forecast will then be indexed by.
    target_col: str - The column which the data to be forecasted is located in
    period: int - The seasonal period
    model_dict: dict - A dictionary of models that will be cross validated.
                       The strings representing the models are keys,
                       and the models with the desired parameters are the values.
                       The model can be one of
                       {Benchmark_forecast(), ETS_forecast(), ARIMA_forecast(), MSTL_forecast(), Prophet_forecast()}.
    best_model: str - The key from model_dict of the model that minimises
                      the chosen evaluation metric in the "select_best_model()" method.

    -------
    Methods
    -------

    select_best_model(self, n_splits:int = 5, test_size:int = None,
                      models:dict = None, min_error_score:str = 'mean_squared_error') -

        Selects the model that minises the chosen evaluation metric
        and returns a dataframe with all of the values of the evaluation metrics for each model in model_dict

        
    plot_diagnostics(self, model = None) - 
        Outputs a figure including bar graph of the residuals, the ACF and a histogram of the residuals for a given model and its parameters

    
    forecast(self, horizon:int, model=None) - 
        Outputs a dataframe including forecast the prediction intervals of a given model and its parameters

    plot_forecast(self, horizon:int, model=None) - 
        Plots the forecast and prediction intervals of the observed data against a forecast using the given model and its parameters

    plot_cross_validation(self, n_splits:int = 5, test_size:int = None, models:dict=None) - 
        Plots the cross validation of one or more methods and their given parameters, indicating where each fold starts and ends
    
    """


    def __init__(self, data:pd.DataFrame, target_col:str, period:int=1):
        
        """
        :param data: pandas.DataFrame - Historical time series data with a date-time index.
        :param target_col: str - Column with historical data. If none will default to the first column.
        :param period: int - Seasonal period.
        """

        self.data = data
        self.period = period
        self.target_col = target_col

        #defining benchmark models to test if none are inputted
        self.model_dict = {'naive': Benchmark_forecast(model='naive', period=period, pred_width=None),
                           'drift': Benchmark_forecast(model='drift', pred_width=None),
                           'mean': Benchmark_forecast(model='mean', pred_width=None),
                           'prophet': Prophet_forecast(pred_width=None),
                           'MSTL': MSTL_forecast(multi_period=period, pred_width=None)}
        

    def select_best_model(self, n_splits:int=5, test_size:int=None, models:dict=None, min_error_score:str = 'mean_squared_error'):

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
            :param min_error_score - Metric used to score the best model, one of
                                     {'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error', 'max_error'}
        Outputs:
            pandas.DataFrame: A summary data frame of the metric values for each forecast model.

        """

        #setting models to default if none are given
        if not models:
            models = self.model_dict

        #performing cross validation on self.data
        cross_val_frame = cross_val(df=self.data,
                                    target_col=self.target_col,
                                    period=self.period,
                                    n_splits=n_splits,
                                    test_size=test_size,
                                    models=models)
                
        #setting the index of the data the same as the cross validation
        first_model = cross_val_frame.index[0][0]
        idx = cross_val_frame.loc[first_model].index
        obs_data = self.data[self.target_col][idx]

        #grouping the cross validation data and calculating the metrics for each of the model forecasts
        grouped_frame = cross_val_frame.groupby(by='model')

        output_frame = grouped_frame.agg(mean_absolute_error = ('forecast', lambda x: mean_absolute_error(obs_data,x)),
                                         mean_absolute_percentage_error = ('forecast', lambda x: mean_absolute_percentage_error(obs_data,x)),
                                         mean_squared_error = ('forecast', lambda x: mean_squared_error(obs_data,x)),
                                         max_error = ('forecast', lambda x: max_error(obs_data,x))
                                        )
        
        #setting the model that minimises the mean squared error to self.best_model
        self.best_model = output_frame[min_error_score].idxmin()
        
        return output_frame
    
    
    def plot_diagnostics(self, model = None) -> go.Figure:

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
            :param model: str - The model used to calculate the forecast with desired parameters, one of
                            {Benchmark_forecast(), ETS_forecast(), ARIMA_forecast(), MSTL_forecast(), Prophet_forecast()}
        Outputs:
            go.Figure - Subplots of the residuals, the ACF and a histogram of the residuals.
        """

        #defaulting to the best model if a model is not given
        if not model:
            model = self.model_dict[self.best_model]

        return resid_diagnostic(df = self.data,
                                target_col=self.target_col,
                                model = model)
    
    def forecast(self, horizon:int, model=None) -> pd.DataFrame:

        """ 
        Creates a summary data frame of the forecast using the model specified.
        The continued dates from the data are the index, and the forecast and 95% and 80%
        prediction intervals are columns.

        Inputs:
            :param horizon: The number of timesteps forecasted into the future.
            :param model: str - The model used to calculate the forecast with desired parameters, one of
                            {Benchmark_forecast(), ETS_forecast(), ARIMA_forecast(), MSTL_forecast(), Prophet_forecast()}
        Ouputs:
            pandas.DataFrame - A forecast using the model specified.

        """
        #defaulting to the best model if a model is not given
        if not model:
            model = self.model_dict[self.best_model]

        return model_forecast(df = self.data,
                              target_col=self.target_col,
                              horizon = horizon,
                              model = model)
    
    def plot_forecast(self, horizon:int, model=None) -> go.Figure:

        """ 
        Creates a plot of the data, the forecast of the desired model
        and the 95% and 80% prediction intervals

        Inputs:
            :param horizon: The number of timesteps forecasted into the future.
            :param model: str - The model used to calculate the forecast with desired parameters, one of
                            {Benchmark_forecast(), ETS_forecast(), ARIMA_forecast(), MSTL_forecast(), Prophet_forecast()}
        Ouputs:
            go.Figure: A plot of the observed data, the forecast and 95% adn 80% prediction intervals.

        """
        #defaulting to the best model if a model is not given
        if not model:
            model = self.model_dict[self.best_model]

        return future_forecast(df = self.data,
                               target_col=self.target_col,
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

        if not models:
            models = self.model_dict

        return cross_val_graph(df = self.data,
                               target_col=self.target_col,
                               period = self.period,
                               n_splits=n_splits,
                               test_size=test_size,
                               models=models)


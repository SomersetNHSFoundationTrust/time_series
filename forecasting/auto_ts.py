from .model_selection import cross_val
from .more_models import *
from .naive_method import Naive_forecast
from .drift_method import Drift_forecast
from .mean_method import Mean_forecast
from .graphs import resid_diagnostic, future_forecast, fitted_forecast_graph, cross_val_graph


import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import *






class Forecast:

    """

    A class to bring together forecasting methods in normal_naive_models.py and more_models.py.
    The main function of this class is to perform and visualise cross validation of these models

    ----------
    Attributes
    ----------

    data: pandas.DataFrame - The data to forecast. Must have an index of dates which the forecast will then be indexed by. 

    target_col: str - The column which the data to be forecasted is located in.

    period: int -   The seasonal period.
                    This will also be used as the default value for the window of the mean forecast.

    model_dict: dict - A dictionary of models that will be cross validated.
                       The strings representing the models are keys,
                       and the models with the desired parameters are the values.
                       The model can be one of
                       {Naive_forecast(), Drift_forecast(), Mean_forecast(),
                       ETS_forecast(), ARIMA_forecast(), MSTL_forecast(), Prophet_forecast()}.

    cross_val_frame: pandas.DataFrame - A data-frame summarising the cross validation
                                        The rows have a multi-index, the first level is each model in the cross
                                        validation and the second level is the date-time index of self.data. The columns
                                        show the forecast values, the error from the observed data and the fold.

    best_model: str - The key from model_dict of the model that minimises
                      the chosen evaluation metric in the "select_best_model()" method.

                      

                      
                      
    -------
    Methods
    -------

    select_best_model(self, n_splits:int = 5, test_size:int = None,
                      models:dict = None, min_error_score:str = 'mean_squared_error') -

        Selects the model that minises the chosen evaluation metric and returns a dataframe
        with all of the values of the evaluation metrics for each model.
        Uses the inputted model or a dictionary of all models if no models are inputted.
        If a model dictionary is inputted this will then become the attribute model_dict.
        and returns a dataframe with all of the values of the evaluation metrics for each model.
        Sets the self.models attribute to the input


    plot_cross_validation(self, n_splits:int = 5, test_size:int = None, models:dict=None) - 
        
        Plots the cross validation forecasts calculated in the select_best_model() method, indicating where each fold starts and ends.
        Must have called Forecast.select_best_model() before calling this method


    plot_diagnostics(self, model = None) - 
        
        Outputs a figure including bar graph of the residuals, the ACF and a histogram of the residuals for a given model and its parameters

    
    forecast(self, horizon:int, model=None) - 
        
        Outputs a dataframe including forecast the prediction intervals of a given model and its parameters

        
    plot_forecast(self, horizon:int, model=None) - 
        
        Plots the forecast and prediction intervals of the observed data against a forecast using the given model and its parameters
        
    """


    def __init__(self, data:pd.DataFrame, target_col:str, period:int=1):
        
        """
        Parameters:
            data: pandas.DataFrame - Historical time series data with a date-time index.
            target_col: str - Column with historical data. If none will default to the first column.
            period: int - Seasonal period.
                          This value will also be used as the default value for the window of the mean forecast.
        
        """

        self.data = data

        self.period = period

        self.target_col = target_col



        #defining benchmark models to test if none are inputted
        self.model_dict = {'naive': Naive_forecast(period = period, pred_width=None),
                           'drift': Drift_forecast(pred_width = None),
                           'mean': Mean_forecast(window = period, pred_width = None),
                           'prophet': Prophet_forecast(pred_width = None),
                           'MSTL': MSTL_forecast(multi_period = period, pred_width = None)}
        



    def select_best_model(self, n_splits:int=5, test_size:int=None, models:dict=None, min_error_score:str = 'mean_squared_error') -> pd.DataFrame:

        """
        Cross validates the models specified.

        
        The model that minimises the mean squared error will be set as the attribute best_model.

        Outputs a data frame that shows the values of the mean absolute error, mean absolute percentage error,
        mean squared error and the max error for the forecast of each model and the observed data.

        Parameters:
            n_splits: int - Number of folds.
            test_size: int - Forecast horizon during each fold.
            models: dict - A dictionary with the models (str) as keys
                           and their respective forecast functions as values.
            min_error_score - Metric used to score the best model, one of
                              {'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error', 'max_error'}
        
        Returns:
            pandas.DataFrame: A summary data frame of the metric values for each forecast model.

        """

        #setting models to default if none are given
        if not models:

            models = self.model_dict


        else:
             #setting the models attribute to the input to use for other methods
             self.model_dict = models



        #performing cross validation on self.data

        self.cross_val_frame = cross_val(df=self.data,
                                         target_col=self.target_col,
                                         period=self.period,
                                         n_splits=n_splits,
                                         test_size=test_size,
                                         models=models)
                


        #setting the index of the data the same as the cross validation
        first_model = self.cross_val_frame.index[0][0]
        
        idx = self.cross_val_frame.loc[first_model].index
        
        obs_data = self.data[self.target_col][idx]


        #grouping the cross validation data and calculating the metrics for each of the model forecasts
        grouped_frame = self.cross_val_frame.groupby(by='model')

        output_frame = grouped_frame.agg(mean_absolute_error = ('forecast', lambda x: mean_absolute_error(obs_data,x)),
                                         mean_absolute_percentage_error = ('forecast', lambda x: mean_absolute_percentage_error(obs_data,x)),
                                         mean_squared_error = ('forecast', lambda x: mean_squared_error(obs_data,x)),
                                         max_error = ('forecast', lambda x: max_error(obs_data,x))
                                        )
        
        #setting the model that minimises the mean squared error to self.best_model
        self.best_model = output_frame[min_error_score].idxmin()
        


        return output_frame
    




    def plot_cross_validation(self, models = None) -> go.Figure:

        """ 
        Plots the cross validation data from the select_best_model() method and the observed data, indicating where each fold is located.
        Must have called Forecast.select_best_model() before calling this method.

        Parameters:
            models: dict - A dictionary with the models (str) as keys
                           and their respective forecast functions as values.
        Returns:
            go.Figure: A plot of the cross validation forecast and observed data.
        """

        fig = go.Figure()
        


        #finding the first first dates for each fold and plotting these as vertical lines
        fold_min_stats = self.cross_val_frame.copy().reset_index().groupby(by='fold').min()

        model_retrained = fold_min_stats.iloc[:,1].to_list()



        for retrained in model_retrained:

                fig.add_vline(retrained, line_width=1.5,
                                line_dash="dash",
                                line_color="green")


        #iterating throught the models and plotting them

        models = self.cross_val_frame.index.get_level_values('model').unique()


        for model in models:

                forecast = self.cross_val_frame.loc[model]['forecast']

                fig.add_trace(go.Scatter(x=forecast.index, y=forecast,
                                        name  = f'{model} forecast'))
                



        #plotting the observed data
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data[self.target_col],
                                name = 'Observed data',
                                line = dict(color = '#00789c')))
        


        fig.update_xaxes(title_text = 'Date')

        fig.update_yaxes(title_text = self.target_col)



        fig.update_layout(height = 800,
                        template = 'plotly_white',
                        legend=dict(orientation="h",  
                                    xanchor="center", 
                                    yanchor="top",  
                                    x=0.5,  
                                    y=-0.2))
        

        return fig


    
    


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

        Parameters:
            model: str - The model used to calculate the forecast with desired parameters, one of
                         {Naive_forecast(), Drift_forecast(), Mean_forecast(),
                          ETS_forecast(), ARIMA_forecast(), MSTL_forecast(), Prophet_forecast()}. 

        Returns:
            go.Figure: Subplots of the residuals, the ACF and a histogram of the residuals.
        """

        #defaulting to the best model if a model is not given
        if not model:

            model = self.model_dict[self.best_model]



        return resid_diagnostic(df = self.data,
                                target_col=self.target_col,
                                model = model)
    

    

    
    def forecast(self, horizon:int=None, model=None) -> pd.DataFrame:

        """ 

        Creates a summary data frame of the forecast using the model specified.
        The continued dates from the data are the index, and the forecast and 95% and 80%
        prediction intervals are columns.

        Parameters:
            horizon: int - The number of timesteps forecasted into the future.
                           Default value is None, in which case an in-sample forecast is calculated
            model: str - The model used to calculate the forecast with desired parameters, one of
                                {Naive_forecast(), Drift_forecast(), Mean_forecast(),
                                 ETS_forecast(), ARIMA_forecast(), MSTL_forecast(), Prophet_forecast()}.

        Returns:
            pandas.DataFrame: A forecast using the model specified.

        """
        #defaulting to the best model if a model is not given
        if not model:

            model = self.model_dict[self.best_model]



        return model_forecast(df = self.data,
                              target_col=self.target_col,
                              horizon = horizon,
                              model = model)
    


    def plot_forecast(self, horizon:int=None, model=None) -> go.Figure:

        """ 
        Creates a plot of the data, the forecast of the desired model
        and the 95% and 80% prediction intervals

        Parameters:
            horizon: int - The number of timesteps forecasted into the future.
                           Defaults to None, in which case a in-sample forecast will be calculated
            model: str - The model used to calculate the forecast with desired parameters, one of
                                {Naive_forecast(), Drift_forecast(), Mean_forecast(),
                                 ETS_forecast(), ARIMA_forecast(), MSTL_forecast(), Prophet_forecast()}. 

        Returns:
            go.Figure: A plot of the observed data, the forecast and 95% adn 80% prediction intervals.

        """


        #defaulting to the best model if a model is not given
        if not model:

            model = self.model_dict[self.best_model]


        if not horizon:
             
             fig = fitted_forecast_graph(df = self.data,
                                         target_col = self.target_col,
                                         model = model)
             

        else:

            fig = future_forecast(df = self.data,
                               target_col=self.target_col,
                               model = model,
                               horizon = horizon)
            

        return fig
import pandas as pd
from prophet import Prophet
from statsforecast.models import AutoETS, AutoARIMA, MSTL
from .normal_naive_models import Benchmark_forecast
from .bootstrap_naive_models import forecast_dates

class Prophet_forecast:

    def __init__(self, pred_width:list=[95,80], **kwargs):

        """
        Inputs:
            :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                             If none are needed, set to None.
            :param period: int - Seasonal period.
            :param kwargs - Facebook Prophet keyword arguments.
        """

        self.kwargs = kwargs
        self.pred_width = pred_width

        if not pred_width:

            self.model = Prophet(**self.kwargs)

        else:
            self.model = Prophet(**self.kwargs, interval_width=pred_width[0] / 100)

    def fit(self, df:pd.DataFrame, target_col:str):
        """
        Fits the model to the data ready to forecast

        Inputs:
            :param df: pandas.DataFrame - Historical time series data with date-time index.
            :param target_col: str - Column with historical data.
        Outputs:
            pandas.DataFrame: Facebook Prophet fitted forecast for df.
        """



        self.data = df

        data = {'ds':df.index, 'y':df[target_col].values}
        self.input_df = pd.DataFrame(data)

        self.model.fit(self.input_df)


    def predict(self, horizon:int = None) -> pd.DataFrame:

        """
        Creates a facebook prophet forecast with the desired predicton intervals.

        Inputs:
            :param horizon: int - Number of time steps to forecast,
                                  defaults to None and if so a fitted forecast will be calculated
        Outputs:
            pandas.DataFrame: Facebook Prophet forecast and prediction intervals for df.
        """

        if not horizon:

            forecast = self.model.predict()

            fitted_forecast = pd.DataFrame(forecast['yhat'].values, index = self.data.index, columns = ['fitted forecast'])

            return fitted_forecast

        else:
            
            future = self.model.make_future_dataframe(periods=horizon)

            forecast = self.model.predict(future)

            output_index = future['ds'].iloc[-horizon:]
            output_forecast = forecast['yhat'].iloc[-horizon:].values

            output_forecast = pd.DataFrame(output_forecast,
                                           index = output_index,
                                           columns = ['forecast'])
            if self.pred_width:

                output_forecast[[f'{self.pred_width[0]}% lower_pi',
                                 f'{self.pred_width[0]}% upper_pi']] = forecast[['yhat_lower','yhat_upper']].iloc[-horizon:].values
                
                for width in self.pred_width[1:]:
                    
                    model = Prophet(**self.kwargs, interval_width=width / 100)
                    model.fit(self.input_df)

                    forecast = model.predict(future)

                    output_forecast[[f'{width}% lower_pi',
                                 f'{width}% upper_pi']] = forecast[['yhat_lower','yhat_upper']].iloc[-horizon:].values

            
            return output_forecast
        





class ETS_forecast:


    def __init__(self, period:int=1, pred_width:list = [95,80], **kwargs):
        """
        Inputs:
            :param period: int - Seasonal period
            :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                             If none are needed, set to None
            :param kwargs - Keyword argumetns for the statsforecast AutoETS class
        """

        self.period = period
        self.pred_width = pred_width
        self.kwargs = kwargs
        self.model = AutoETS(season_length = self.period, **self.kwargs)


    def fit(self, df:pd.DataFrame, target_col:str):

        """
        Fits the inputted data to the ETS model.

        Inputs:
            :param df: pandas.DataFrame - Historical time series data with date-time index
            :param target_col: str - Column with historical data
        """

        self.model.fit(df[target_col].values)
        self.data = df


    def predict(self, horizon:int = None):
        """
        Uses the horizon and prediction interval width to create a forecast using ETS.
        Must fit to the data using ETS_forecast.fit() before using this method

        Inputs:
            :param horizon: int - Number of timesteps forecasted into the future.
                                  Defaults to None, in which case a fitted forecast is outputted
        Outputs:
            pd.DataFrame - A forecast or fitted forecast for the inputted data using the ETS model
        """

        if not horizon:
            
            forecast = pd.DataFrame(self.model.predict_in_sample())
            output_forecast = pd.DataFrame(index = self.data.index)
            output_forecast['fitted forecast'] = forecast['fitted'].values

            return output_forecast

        else:

            forecast = pd.DataFrame(self.model.predict(h=horizon,level=self.pred_width))

            #creating the output dataframe, renaming the columns and setting the index to the extended dates
            output_forecast = forecast_dates(self.data,horizon)
            output_forecast['forecast'] = forecast['mean'].values

            if self.pred_width:

                for width in self.pred_width:

                    output_forecast[[f'{width}% lower_pi', f'{width}% upper_pi']] = forecast[[f'lo-{width}',f'hi-{width}']].values

            
            return output_forecast
        


class ARIMA_forecast:
    

    def __init__(self, period:int=1, pred_width:list = [95,80], **kwargs):
        """
        Inputs:
            :param period: int - Seasonal period
            :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                             If none are needed, set to None
            :param kwargs - Keyword argumetns for the statsforecast AutoARIMA class
        """

        self.period = period
        self.pred_width = pred_width
        self.kwargs = kwargs
        self.model = AutoARIMA(season_length = self.period, **self.kwargs)


    def fit(self, df:pd.DataFrame, target_col:str):

        """
        Fits the inputted data to the ARIMA model.

        Inputs:
            :param df: pandas.DataFrame - Historical time series data with date-time index
            :param target_col: str - Column with historical data
        """

        self.model.fit(df[target_col].values)
        self.data = df


    def predict(self, horizon:int = None):
        """
        Uses the horizon and prediction interval width to create a forecast using ARIMA.
        Must fit to the data using ARIMA_forecast.fit() before using this method

        Inputs:
            :param horizon: int - Number of timesteps forecasted into the future.
                                  Defaults to None, in which case a fitted forecast is outputted
        Outputs:
            pd.DataFrame - A forecast or fitted forecast for the inputted data using the ARIMA model
        """

        if not horizon:
            
            forecast = pd.DataFrame(self.model.predict_in_sample())
            output_forecast = pd.DataFrame(index = self.data.index)
            output_forecast['fitted forecast'] = forecast['fitted'].values

            return output_forecast

        else:

            forecast = pd.DataFrame(self.model.predict(h=horizon,level=self.pred_width))

            #creating the output dataframe, renaming the columns and setting the index to the extended dates
            output_forecast = forecast_dates(self.data,horizon)
            output_forecast['forecast'] = forecast['mean'].values

            if self.pred_width:

                for width in self.pred_width:

                    output_forecast[[f'{width}% lower_pi', f'{width}% upper_pi']] = forecast[[f'lo-{width}',f'hi-{width}']].values

                return output_forecast
            
            return output_forecast
            

            

class MSTL_forecast:
    

    def __init__(self, multi_period, trend_forecaster = AutoETS(model = 'ZZN'), pred_width:list = [95,80], **kwargs):
        """
        Inputs:
            :param multi_period: int - Seasonal periods as an int or list
            :param trend_forecaster - A statsforecast.models class to forecat the trend,
                                      defaults to AutoETS
            :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                             If none are needed, set to None
            :param kwargs - Keyword argumetns for the statsforecast.models MSTL class
                            or statsmodels.tsa.seasonal STL class
        """

        self.multi_period = multi_period
        self.pred_width = pred_width
        self.kwargs = kwargs
        self.model = MSTL(season_length = self.multi_period, trend_forecaster = trend_forecaster, **self.kwargs)


    def fit(self, df:pd.DataFrame, target_col:str):

        """
        Fits the inputted data to the MSTL model.

        Inputs:
            :param df: pandas.DataFrame - Historical time series data with date-time index
            :param target_col: str - Column with historical data
        """

        self.model.fit(df[target_col].values)
        self.data = df


    def predict(self, horizon:int = None):
        """
        Uses the horizon and prediction interval width to create a forecast using ARIMA.
        Must fit to the data using ARIMA_forecast.fit() before using this method

        Inputs:
            :param horizon: int - Number of timesteps forecasted into the future.
                                  Defaults to None, in which case a fitted forecast is outputted
        Outputs:
            pd.DataFrame - A forecast or fitted forecast for the inputted data using the MSTL model
        """

        if not horizon:
            
            forecast = pd.DataFrame(self.model.predict_in_sample())
            output_forecast = pd.DataFrame(index = self.data.index)
            output_forecast['fitted forecast'] = forecast['fitted'].values

            return output_forecast

        else:

            forecast = pd.DataFrame(self.model.predict(h=horizon,level=self.pred_width))

            #creating the output dataframe, renaming the columns and setting the index to the extended dates
            output_forecast = forecast_dates(self.data,horizon)
            output_forecast['forecast'] = forecast['mean'].values

            if self.pred_width:

                for width in self.pred_width:

                    output_forecast[[f'{width}% lower_pi', f'{width}% upper_pi']] = forecast[[f'lo-{width}',f'hi-{width}']].values

                return output_forecast
            
            return output_forecast
        


def model_forecast(df:pd.DataFrame, target_col:str, model,  horizon:int=None) -> pd.DataFrame:
    """
    Creates forecast and prediction intervals of the desired model.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column with historical data.
        :param model: str - The model used to calculate the forecast with desired parameters, one of
                            {Benchmark_forecast(), ETS_forecast(), ARIMA_forecast(), MSTL_forecast(), Prophet_forecast()}
        :param horizon: int - Number of timesteps forecasted into the future.
                                  Defaults to None, in which case a fitted forecast is outputted 
    Output:
        pandas.DataFrame: A forecast for df using the chosen model.
    """

    model.fit(df = df, target_col=target_col)
    
    output_forecast = model.predict(horizon = horizon)

    return output_forecast







            



"""def MSTL_forecast(df:pd.DataFrame, target_col:str,horizon:int, multi_period=1, pred_width:list = [95,80], trend_forecaster:str = 'ETS' ,**kwargs) -> pd.DataFrame:
    
    Creates a MSTL forecast ising the statsmodels MSTL function.

    This decomposes the time series into trend, seasonal, and residual components,
    and forecasts the trend using the desired model.

    If no model is inputted, it defaults to using exponential smoothing to forecast the trend.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column to forecast.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param multi_period: int or list - Seasonal period/s.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
        :param trend_forecaster: str - The model (a key from model_dict) used to forecast the trend.
        :param **kwargs - Keyword arguments for trend_forecaster.
    Ouputs:
        pandas.DataFrame: MSTL forecast and prediction intervals for the trend of df.
    

    if multi_period == 1:
        return 'The time series must have periodicity to decompose'

    #decomposing df and finding the trend
    mstl = MSTL(df[target_col],periods=multi_period, **kwargs).fit()
    trend = pd.DataFrame(mstl.trend.values, index=df.index, columns=['trend'])

    forecaster = model_dict[trend_forecaster]
    output_forecast = forecaster(df = trend,
                                 target_col = 'trend',
                                 horizon = horizon,
                                 pred_width = pred_width,
                                 **kwargs)
        
    return output_forecast
    



def prophet_fit(df:pd.DataFrame, target_col:str,**kwargs) -> pd.DataFrame:
    
    
    Creates a fitted forecast using the facebook prophet model.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with historical data.
        :param kwargs - Facebook Prophet keyword arguments.
    Outputs:
        pandas.DataFrame: Facebook Prophet fitted forecast for df.
    

    data = {'ds':df.index, 'y':df[target_col].values}
    input_df = pd.DataFrame(data)

    model = Prophet()
    model.fit(input_df)
    forecast = model.predict()

    fitted_forecast = pd.DataFrame(forecast['yhat'].values, index = df.index, columns = ['fitted forecast'])

    return fitted_forecast

    
def prophet_forecast(df:pd.DataFrame, target_col:str, horizon:int, period:int = 1, pred_width:list = [95,80], **kwargs) -> pd.DataFrame:
    
    
    Creates a facebook prophet forecast with the desired predicton intervals.

    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of time steps to forecast.
        :param period: int - Seasonal period.
        :param kwargs - Facebook Prophet keyword arguments.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
    Outputs:
        pandas.DataFrame: Facebook Prophet forecast and prediction intervals for df.
    

    output_forecast = pd.DataFrame()

    #setting up the input dataframe for Prophet and storing this in output_forecast
    data = {'ds':df.index, 'y':df[target_col].values}
    input_df = pd.DataFrame(data)

    #entering the first prediction interval width manually to store the forecast if pred_width is not empty
    if pred_width != []:
        first_width = pred_width[0]
    else:
        first_width = 80

    #testing which keyword arguments are for prophet

    model = Prophet(**kwargs,interval_width=first_width / 100)
    model.fit(input_df)
    future = model.make_future_dataframe(periods=horizon)

    forecast = model.predict(future)

    output_forecast.index = future['ds'].iloc[-horizon:]
    output_forecast['forecast'] = forecast['yhat'].iloc[-horizon:].values

    if pred_width != []:

        output_forecast[[f'{first_width}% lower_pi',
                         f'{first_width}% upper_pi']] = forecast[['yhat_lower','yhat_upper']].iloc[-horizon:].values

    
    #storing each other prediction interval
    for width in pred_width[1:]:

        model = Prophet(**kwargs,interval_width=width/100)
        model.fit(input_df)
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)

        output_forecast[[f'{width}% lower_pi',
                         f'{width}% upper_pi']] = forecast[['yhat_lower','yhat_upper']].iloc[-horizon:].values

    return output_forecast


        


model_dict = {'naive':naive_pi, 'drift':drift_pi, 'mean':mean_pi,
              'ETS':ETS_forecast, 'ARIMA':ARIMA_forecast,
              'prophet':prophet_forecast, 'MSTL':MSTL_forecast}





def benchmark_fit(df:pd.DataFrame, target_col:str, model:str, period:int=1, **kwargs) -> pd.DataFrame:
    
    Inputs:
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column with historical data.
        :param model: str - Model to calculate the fitted forecast, one of the keys from model_dict.
        :param period: int - Seasonal period.
        :param **kwargs - Keyword arguments for the sktime functions AutoETS or AutoARIMA.
    Output:
        pandas.DataFrame: A fitted forecast for df of the specified model.
    

    fitted_forecast = pd.DataFrame()

    if model == 'naive':

        fitted_forecast = pd.DataFrame(index = df.index)
        fitted_forecast['fitted forecast'] = naive_error(df, target_col, period)['fitted forecast']

    elif model == 'drift':

        fitted_forecast = pd.DataFrame(index = df.index)
        fitted_forecast['fitted forecast'] = drift_error(df,target_col)['fitted forecast']

    elif model == 'mean':

        fitted_forecast = pd.DataFrame(index = df.index)
        fitted_forecast['fitted forecast'] = mean_error(df,target_col)['fitted forecast']

    elif model == 'ETS':

        fitted_forecast = ETS_fit(df, target_col, period, **kwargs)

    elif model == 'ARIMA':

        fitted_forecast = ARIMA_fit(df,target_col, period, **kwargs)

    elif model == 'prophet':
        fitted_forecast = prophet_fit(df,target_col, **kwargs)

    return fitted_forecast"""
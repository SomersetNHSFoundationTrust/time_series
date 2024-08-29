import pandas as pd

from prophet import Prophet

from statsforecast.models import AutoETS, AutoARIMA, MSTL

from .naive_method import forecast_dates






class Prophet_forecast:

    """

    Creates a forecast using facebook Prophet foreasting model

    """

    def __init__(self, pred_width:list=[95,80], **kwargs):

    
        """

        Parameters:
            pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                            If none are needed, set to None.
            period: int - Seasonal period.
            kwargs: Facebook Prophet keyword arguments.

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

        Parameters:
            df: pandas.DataFrame - Historical time series data with date-time index.
            target_col: str - Column with historical data.
        
        Returns:
            pandas.DataFrame: Facebook Prophet fitted forecast for df.
        """



        self.data = df


        data = {'ds':df.index, 'y':df[target_col].values}

        self.input_df = pd.DataFrame(data)


        self.model.fit(self.input_df)


    def predict(self, horizon:int = None) -> pd.DataFrame:

        """

        Creates a facebook prophet forecast with the desired predicton intervals.

        Parameters:
            horizon: int - Number of time steps to forecast,
                           defaults to None and if so a fitted forecast will be calculated
        
        Returns:
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

    """

    Creates a forecast and prediction intervals using AutoETS from statsforecast.models

    """


    def __init__(self, period:int=1, pred_width:list = [95,80], **kwargs):

        """

        Parameters:
            period: int - Seasonal period
            pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                             If none are needed, set to None
            kwargs: Keyword argumetns for the statsforecast AutoETS class
        """

        self.period = period
        
        self.pred_width = pred_width
        
        self.kwargs = kwargs
        
        self.model = AutoETS(season_length = self.period, **self.kwargs)




    def fit(self, df:pd.DataFrame, target_col:str):

        """

        Fits the inputted data to the ETS model.

        Parameters:
            df: pandas.DataFrame - Historical time series data with date-time index
            target_col: str - Column with historical data
        
        """

        self.model.fit(df[target_col].values)

        self.data = df





    def predict(self, horizon:int = None):
        """

        Uses the horizon and prediction interval width to create a forecast using ETS.
        Must fit to the data using ETS_forecast.fit() before using this method

        Parameters:
            horizon: int - Number of timesteps forecasted into the future.
                           Defaults to None, in which case a fitted forecast is outputted
        
        Returns:  
            pandas.DataFrame: A forecast or fitted forecast for the inputted data using the ETS model
        """



        if not horizon:
            

            forecast = pd.DataFrame(self.model.predict_in_sample())
           
            output_forecast = pd.DataFrame(index = self.data.index)
           
            output_forecast['fitted forecast'] = forecast['fitted'].values



            return output_forecast
        

        else:


            forecast = pd.DataFrame(self.model.predict(h=horizon,level=self.pred_width))



            #creating the output dataframe, renaming the columns and setting the index to the extended dates
            forecast_ds = forecast_dates(self.data,horizon)
            
            output_forecast = pd.DataFrame(forecast['mean'].values, index = forecast_ds, columns = ['forecast'])



            if self.pred_width:

                for width in self.pred_width:


                    output_forecast[[f'{width}% lower_pi', f'{width}% upper_pi']] = forecast[[f'lo-{width}',f'hi-{width}']].values

            

            return output_forecast
        


class ARIMA_forecast:
    
    """

    Creates a forecast and prediction intervals using AutoARIMA from statsforecast.models

    """
    

    def __init__(self, period:int=1, pred_width:list = [95,80], **kwargs):

        """

        Parameters:
            period: int - Seasonal period
            pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                            If none are needed, set to None
            kwargs: Keyword argumetns for the statsforecast AutoARIMA class
        """



        self.period = period
        
        self.pred_width = pred_width
        
        self.kwargs = kwargs
        
        self.model = AutoARIMA(season_length = self.period, **self.kwargs)




    def fit(self, df:pd.DataFrame, target_col:str):

        """

        Fits the inputted data to the ARIMA model.

        Parameters:
            df: pandas.DataFrame - Historical time series data with date-time index
            target_col: str - Column with historical data
        """



        self.model.fit(df[target_col].values)

        self.data = df



    def predict(self, horizon:int = None):

        """

        Uses the horizon and prediction interval width to create a forecast using ARIMA.
        Must fit to the data using ARIMA_forecast.fit() before using this method


        Parameters:
            horizon: int - Number of timesteps forecasted into the future.
                                  Defaults to None, in which case a fitted forecast is outputted
        Returns:
            
            pandas.DataFrame: A forecast or fitted forecast for the inputted data using the ARIMA model
        """

        if not horizon:
            

            forecast = pd.DataFrame(self.model.predict_in_sample())
           
            output_forecast = pd.DataFrame(index = self.data.index)
           

            output_forecast['fitted forecast'] = forecast['fitted'].values

            return output_forecast

        else:


            forecast = pd.DataFrame(self.model.predict(h=horizon,level=self.pred_width))


            #creating the output dataframe, renaming the columns and setting the index to the extended dates
            forecast_ds = forecast_dates(self.data,horizon)
            
            output_forecast = pd.DataFrame(forecast['mean'].values, index = forecast_ds, columns = ['forecast'])



            if self.pred_width:

                for width in self.pred_width:

                    

                    output_forecast[[f'{width}% lower_pi', f'{width}% upper_pi']] = forecast[[f'lo-{width}',f'hi-{width}']].values

            
            return output_forecast
            

            

class MSTL_forecast:
    
    """

    Creates a forecast and prediction intervals using MSTL from statsforecast.models

    """
    

    def __init__(self, multi_period, trend_forecaster = AutoETS(model = 'ZZN'), pred_width:list = [95,80], **kwargs):

        """

        Parameters:
            multi_period: int - Seasonal periods as an int or list
            trend_forecaster - A statsforecast.models class to forecat the trend,
                                    defaults to AutoETS
            pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
                                                            If none are needed, set to None
            kwargs: Keyword arguments for the statsforecast.models MSTL class
                        or statsmodels.tsa.seasonal STL class
        """



        self.multi_period = multi_period
        
        self.pred_width = pred_width
        
        self.kwargs = kwargs
        
        self.model = MSTL(season_length = self.multi_period, trend_forecaster = trend_forecaster, **self.kwargs)




    def fit(self, df:pd.DataFrame, target_col:str):

        """
        Fits the inputted data to the MSTL model.

        Parameters:
            df: pandas.DataFrame - Historical time series data with date-time index
            target_col: str - Column with historical data
        """


        self.model.fit(df[target_col].values)
        
        self.data = df




    def predict(self, horizon:int = None):
        """
        Uses the horizon and prediction interval width to create a forecast using ARIMA.
        Must fit to the data using ARIMA_forecast.fit() before using this method

        Parameters:
            horizon: int - Number of timesteps forecasted into the future.
                                  Defaults to None, in which case a fitted forecast is outputted
        
        Returns:    
            pandas.DataFrame: A forecast or fitted forecast for the inputted data using the MSTL model
        """



        if not horizon:
            

            forecast = pd.DataFrame(self.model.predict_in_sample())
            
            output_forecast = pd.DataFrame(index = self.data.index)
           
            output_forecast['fitted forecast'] = forecast['fitted'].values


            return output_forecast

        else:

            
            forecast = pd.DataFrame(self.model.predict(h=horizon,level=self.pred_width))



            #creating the output dataframe, renaming the columns and setting the index to the extended dates
            forecast_ds = forecast_dates(self.data,horizon)
            
            output_forecast = pd.DataFrame(forecast['mean'].values, index = forecast_ds, columns = ['forecast'])



            if self.pred_width:

                for width in self.pred_width:


                    output_forecast[[f'{width}% lower_pi', f'{width}% upper_pi']] = forecast[[f'lo-{width}',f'hi-{width}']].values



            
            return output_forecast
        





def model_forecast(df:pd.DataFrame, target_col:str, model,  horizon:int=None) -> pd.DataFrame:
    """

    Creates forecast and prediction intervals of the desired model.

    Parameters:
        df: pandas.DataFrame - Historical time series data.
        target_col: str - Column with historical data.
        model: str - The model used to calculate the forecast with desired parameters, one of
                        {Naive_forecast(), Drift_forecast(), Mean_forecast(),
                            ETS_forecast(), ARIMA_forecast(), MSTL_forecast(), Prophet_forecast()}. 
        horizon: int - Number of timesteps forecasted into the future.
                            Defaults to None, in which case a fitted forecast is outputted 
    
    Returns:
        pandas.DataFrame: A forecast for df using the chosen model.
    """




    model.fit(df = df, target_col=target_col)

    
    output_forecast = model.predict(horizon = horizon)


    return output_forecast
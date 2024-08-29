# Functions for train/test splits & cross validation
# We can output summary tables for key forecasting metrics (i.e., MAPE/MAE/...)


from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import *
import pandas as pd
import time

from .more_models import ETS_forecast, ARIMA_forecast, Prophet_forecast, MSTL_forecast
from .naive_method import Naive_forecast
from .drift_method import Drift_forecast
from .mean_method import Mean_forecast



#metrics to cross validate different forecasting methods
eval_metrics = [mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, max_error]


    
def cross_val(df:pd.DataFrame, target_col:str, period:int=1,
              n_splits:int=5, test_size:int=None,
              models:dict = None) -> pd.DataFrame:
    
    """

    Test forecasting method/s on observed data and records the error.

    Parameters:
         df: pandas.DataFrame - Univariate time series dataset.
         target_col:str - Column with historical data.
         period: int - Seasonal period.
         n_splits: int - Number of folds.
         test_size: int -  Forecast horizon during each fold.
         model: dict - A dictionary with the models (str) as keys and their respective classes as values.
                             If left None, it will cross validate all models
            
    Returns:
        pandas.DataFrame: A dataframe with a the models and dates as a multi-index and their respective forecast, errors and folds as columns
    """




    if not models:


        models = {'naive': Naive_forecast(period = period, pred_width=None),
                  'drift': Drift_forecast(pred_width = None),
                  'mean': Mean_forecast(window = period, pred_width = None),
                  'ETS': ETS_forecast(period=period),
                  'ARIMA': ARIMA_forecast(period=period),
                  'prophet': Prophet_forecast(pred_width=None),
                  'MSTL': MSTL_forecast(multi_period=period)}
        


    #defining a dictionary which will be the output of the cross validation
    output_dict = {}



    #iterating through the model dictionary
    for model in models:



        #splitting the time series to test
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        cross_val_idx = tscv.split(df[target_col])



        #creating a list to store each fold
        cv_summary = []



        #finding the new forecast model and fitting it to the data
        forecaster = models[model]


        #for tracking the time for each fold
        print(f'{model}:')


        for fold, (train, test) in enumerate(cross_val_idx):



            #measuring time elapsed for each fold
            start = time.time()



            #copying the test data in df to cv_output to compare to the forecast
            cv_output = df[target_col].copy().iloc[test].to_frame()


    
            #forecasting from training data
            forecaster.fit(df = df.copy().iloc[train],
                           target_col = target_col)
            

            
            forecast = forecaster.predict(horizon = len(test))



            #cannot refit the prophet model so we instantiate a new object
            if model == 'prophet':
                prophet_kwargs = models['prophet'].kwargs
                forecaster = Prophet_forecast(pred_width=None, **prophet_kwargs)



            cv_output['forecast'] = forecast['forecast'].values
            
            cv_output['fold'] = fold
            
            cv_output['error'] = cv_output[target_col] - cv_output['forecast']
            
            cv_summary.append(cv_output)
            


            end = time.time()

            print('fold {fold_no}: {time:.4f} seconds'.format(fold_no=fold, time=end-start))



        output_dict[model] = pd.concat(cv_summary)


    output_frame = pd.concat(output_dict, names = ['model',])



    return output_frame

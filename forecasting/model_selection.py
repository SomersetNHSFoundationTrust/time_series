# Functions for train/test splits & cross validation
# We can output summary tables for key forecasting metrics (i.e., MAPE/MAE/...)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import *
import pandas as pd
from .more_models import model_dict
import time

#metrics to cross validate different forecasting methods
eval_metrics = [mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, max_error]

    
def cross_val(df:pd.DataFrame, target_col:str, period:int=1,
              n_splits:int=5, test_size:int=None,
              models:dict = model_dict, time_taken:bool=False,**kwargs) -> pd.DataFrame:
    
    """
    Test forecasting method/s on observed data and records the error.

    Inputs:
        :param df: pandas.DataFrame - Univariate time series dataset.
        :param target_col:str - Column with historical data.
        :param period: int - Seasonal period.
        :param n_splits: int - Number of folds.
        :param test_size: int -  Forecast horizon during each fold.
        :param model: dict - A dictionary with the models (str) as keys and their respective functions as values.
        :param time_taken: bool - Toggle woether to print the time taken for each fold of each model
        :param **kwargs - Keyword arguments to be used for every model.
    Outputs:
        pandas.DataFrame: A dataframe with a the models and dates as a multi-index and their respective forecast, errors and folds as columns
    """

    #defining a dictionary which will be the output of the cross validation
    output_dict = {}

    #iterating through the model dictionary
    for model in models:

        #splitting the time series to test
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

        cross_val_idx = tscv.split(df[target_col])



        #creating a list to store each fold
        cv_summary = []

        forecaster = models[model]

        if time_taken:
            print(f'{model}:')


        for fold, (train, test) in enumerate(cross_val_idx):

            #measuring time elapsed for each fold
            start = time.time()

            #copying the test data in df to cv_output to compare to the forecast
            cv_output = df[target_col].copy().iloc[test].to_frame()
    
            #forecasting from training data
            forecast = forecaster(df = df.copy().iloc[train],
                                  target_col = target_col,
                                  horizon =  len(test),
                                  period = period,
                                  pred_width = [],
                                  **kwargs)

            cv_output['forecast'] = forecast['forecast'].values
            cv_output['fold'] = fold
            cv_output['error'] = cv_output[target_col] - cv_output['forecast']
            cv_summary.append(cv_output)
            
            end = time.time()

            if time_taken:
                print('fold {fold_no}: {time:.4f}'.format(fold_no=fold,time=end-start))

        output_dict[model] = pd.concat(cv_summary)


    output_frame = pd.concat(output_dict,names = ['model',])

    return output_frame



"""
def forecast_metrics(df:pd.DataFrame,target_col:str,period:int=1, n_splits:int=5, test_size:int=None, models:dict = model_dict,**kwargs) -> pd.DataFrame:
    
    
    Cross validation (k-fold) for time series data.

    Inputs:
        :param df: pandas.DataFrame - Univariate time series dataset.
        :param target_col: str - Column with historical data.
        :param periodL int - seasonal period.
        :param n_splits: int - Number of folds.
        :param test_size: int -  Forecast horizon during each fold.
        :param models: dict - a dictionary with the models (str) as keys and their respective functions as values.
        :param **kwargs - Keyword arguments to be used for every model.
    Outputs:
        pandas.DataFrame - A value of each evaluation metric for each model using cross_val.
    

    #cross validate all forecasting models
    train_test_dict = cross_val(df,target_col,period, n_splits, test_size, models,**kwargs)

    #creating a dataframe to store the metric scores for each method
    output_frame = pd.DataFrame(index = ['mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error', 'max_error'])

    #evaluating each of the methods against the observed data
    for model in models:

        #storing the forecast for this model
        cv = train_test_dict[model]['forecast']

        #storing the observed data to compare to the forecast
        obs_data = df[target_col][cv.index]

        eval_list = []

        for metric in eval_metrics:

            #creating a list of the metric score for each method
            eval = metric(y_true = obs_data, y_pred = cv)

            eval_list.append(eval)

        #adding this as a column to output_frame
        output_frame[model] = eval_list


    return output_frame.transpose()

    
def kwargs_in_func(func,kwargs) -> dict:

    
    Tests if any of the keyword arguments are in the function parameters

    Inputs:
        :param func - The function to test.
        :param kwargs - the keyword arguments to test.
    Outputs:
        dict - the subset of kwargs in the function parameters.
    

    #storing the parameters of func
    func_args = inspect.signature(func)

    output_dict = {}

    #iterating through kwargs to test if any parameters are in func_args
    for key in kwargs:

        if key in func_args:

            output_dict[key] = kwargs[key]

    #returning the a
    return output_dict
    


"""
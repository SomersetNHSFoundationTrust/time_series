# Graphs for both output forecasts & evaluation
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import normaltest, norm
from statsmodels.tsa.seasonal import MSTL
import numpy as np
from .bootstrap_naive_models import bs_benchmark_forecast
from .more_models import benchmark_fit, benchmark_forecast, model_dict
from .model_selection import forecast_metrics, cross_val
from sklearn.metrics import *



def resid_diagnostic(df:pd.DataFrame,target_col:str, model:str, period:int=1, **kwargs) -> go.Figure:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param df_target_col: str - Column with historical data.
        :param model: str - Model used to perform the residual diagnostics, one of the keys from model_dict.
        :param **kwargs - Keyword arguments for the sktime functions AutoETS or AutoARIMA.
    Outputs:
        go.Figure - Subplots of the residuals, the ACF and a histogram of the residuals.
    """

    plot_frame = pd.DataFrame()

    #calculating the residuals to analyse.
    fitted_forecast = benchmark_fit(df, target_col, model, period)
    plot_frame['error'] = df[target_col] - fitted_forecast['fitted forecast']

    #making subplots, the bar chart of the residuals taking up both columns of the first row
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2},None],[{}, {}]],
        subplot_titles=("Residual plot","Autocorrelation function", "Residual histogram"),
        vertical_spacing=0.18)
    


    #plotting the bar chart of the residuals
    fig.add_trace(go.Bar(x=plot_frame.index, y=plot_frame['error'],
                        marker_color='#00789c',
                        showlegend=False),
                    row=1, col=1)

    #adding a trace of the mean value of the residuals, positioning the legend underneath the plot
    fig.add_trace(go.Scatter(x=plot_frame.index,
                            y=[np.mean(plot_frame['error'])] * len(df),
                            name = 'Mean value',
                            line=dict(color='black')),
                            row = 1, col=1)

    fig.update_traces(row=1, col=1, legend='legend2')

    fig.update_layout({'legend2': dict(x=0.5,y=0.5,xanchor='center', yanchor="bottom")})




    #calculating the correlation coeficients for the lagged values to plot the ACF
    #and the bounds such that if the ACF is white noise, 95% of the bars should be withing these bounds
    cor_coeffs=[]

    for i in range(0,len(df)):
        plot_frame['shift'] = plot_frame['error'].shift(i+1)
        cor_coeffs.append(plot_frame['error'].corr(plot_frame['shift']))

    fig.add_trace(go.Bar(y=cor_coeffs,
                        marker_color = '#00789c',
                        showlegend=False),
                    row=2, col=1)
                    
    fig.add_trace(go.Scatter(y=[1.96 / np.sqrt(len(df))] * len(df),
                            line=dict(color = '#d1495b', dash = 'dash'),
                            name = '95% bound'),
                    row=2, col=1)
                    
    fig.add_trace(go.Scatter(y=[-1.96 / np.sqrt(len(df))] * len(df),
                            line=dict(color = '#d1495b', dash = 'dash'),
                            showlegend=False),
                    row=2, col=1)

    fig.update_traces(row=2,col=1,legend = 'legend3')

    fig.update_layout({'legend3' : dict(x=0.23, y=-0.1, xanchor = 'center', yanchor='bottom')})




    #plotting a histogram of the residuals
    fig.add_trace(go.Histogram(x=plot_frame['error'],
                            opacity=0.7,
                            marker_color='#66a182',
                            nbinsx=100,
                            showlegend=False),
                    row=2, col=2)


    #plotting a normal distribution with standard deviation the same as the residuals on the histogram
    sd = np.std(plot_frame['error'].dropna())
    x = np.linspace(start=plot_frame['error'].min()-1, stop=plot_frame['error'].max()+1, num=100)

    #finding the scales for the distribution to plot on the histogram
    bin_width = (df[target_col].values.max() - df[target_col].values.min()) / 100
    N = len(plot_frame['error'].dropna())

    fig.add_trace(go.Scatter(x=x,y=norm.pdf(x, loc=0, scale=sd ** 2) * N * bin_width,
                            name = 'Theoretical normally distributed residuals, standard deviation {:.2f}'.format(sd)),
                    row=2,col=2)
    
    #positioning the legend below the plot    
    fig.update_traces(row=2,col=2,legend='legend4')
    fig.update_layout({'legend4' : dict(x=0.77,y=-0.1, xanchor = 'center', yanchor='bottom')})




    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='Error from the forecast', row=1, col=1)

    fig.update_xaxes(title_text='Lag', row=2, col=1)
    fig.update_yaxes(title_text='Correlation coefficient', row=2, col=1)

    fig.update_xaxes(title_text='Error', row=2, col=2)
    fig.update_yaxes(title_text='Frequency', row=2, col=2)

    fig.update_layout(height=1000,
                    title_text="Residual Diagnostics",
                    template='plotly_white')

    #calculating the 2-sided chi-squared p-value for a normal hypothesis test on the residuals
    p_value = normaltest(plot_frame['error'].values).pvalue
    print('The 2-sided chi-squared probability for a normal hypotheis test on the residuals: {:.4f}'.format(p_value))

    return fig



def fitted_forecast_graph(df:pd.DataFrame,target_col:str,model:str,period:int=1, **kwargs) -> go.Figure:
    
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with Historical data.
        :param model: str - Model used to calculate the fitted forecast, one of the keys from model_dict.
        :param period: int - Seasonal period.
        :param **kwargs - Keyword arguments for the selected model.
    Ouputs:
        go.Figure - A plot of the observed data and the fit of the forecast selected.
    """

    fig=go.Figure()

    fig.add_trace(go.Scatter(x=df.index,y=df[target_col],
                             line=dict(color='#00789c'),
                             name='Observed data'))

    #calculating the fitted forecast for the selected model
    fitted_forecast = benchmark_fit(df,target_col, model, period, **kwargs)


    fig.add_trace(go.Scatter(x=df.index,y=fitted_forecast['fitted forecast'],
                        line=dict(color='#d1495b'),
                        name='Fitted forecast'))
    
    fig.update_layout(height=600,
                    title_text=f'Observed data and {model} fitted forecast',
                    legend=dict(orientation="h",  
                                xanchor="center", 
                                yanchor="top",  
                                x=0.5,  
                                y=-0.2))

    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text= target_col)

    
    
    return fig



def decomp(df:pd.DataFrame, target_col:str, period, **MSTLkwargs) -> tuple[pd.Series]:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with historical data.
        :param period: int ot list - Seasonal period.
        :param **MSTLkwargs - Keyword arguments for the statsmodels MSTL class.
    Ouputs:
        tuple: the observed data decomposed into the trend, seasonal and remainder data using statsmodels MSTL.
    """

    mstl = MSTL(df[target_col],periods = period, **MSTLkwargs).fit()
    trend = mstl.trend
    seasonal = mstl.seasonal
    remainder = mstl.resid

    return trend, seasonal, remainder



def decomp_plot(df:pd.DataFrame, target_col:str, period, **MSTLkwargs) -> go.Figure:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with historical data.
        :param period: int or list - Seasonal period.
        :param **MSTLkwargs - Keyword arguments for the statsmodels MSTL class.
    Ouputs:
        go.Figure - A plot of the observed data and its components, trend, seasonal and remainder.
    """

    #decomposing the time series, creating multiple plots for the seasonal data.
    (trend, seasonal, remainder) = decomp(df,target_col,period,**MSTLkwargs)

    seasonal = pd.DataFrame(seasonal, index = df.index)

    seasonal_components = len(seasonal.columns)
    no_rows = seasonal_components + 3
    

    fig = make_subplots(rows=no_rows,
                        cols=1,
                        subplot_titles=["Observed data","Trend"] + seasonal.columns.to_list() + ["Remainder"],
                        horizontal_spacing=0.01)

    fig.add_trace(go.Scatter(x=df.index, y=df[target_col],
                             name = 'Observed data'),
                    row=1, col=1)
    
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text=target_col, row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=trend.values,
                            showlegend=False),
                    row=2, col=1)
    
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text=target_col, row=2, col=1)
    
    #iterating through each seasonal component and creating a plot
    row_no = 3

    for column in seasonal.columns:
        
        fig.add_trace(go.Scatter(x=df.index, y=seasonal[column].values,
                                name = 'Observed data'),
                        row=row_no, col=1)
        
        fig.update_xaxes(title_text='Date', row=row_no, col=1)
        fig.update_yaxes(title_text=target_col, row=row_no, col=1)
        
        row_no += 1


    
    fig.add_trace(go.Scatter(x=df.index, y=remainder.values,
                             name = 'Observed data'),
                  row=no_rows, col=1)
    
    fig.update_xaxes(title_text='Date', row=no_rows, col=1)
    fig.update_yaxes(title_text=target_col, row=no_rows, col=1)


    fig.update_layout(height=1200,
                      showlegend=False, 
                      title_text="Decomposition of data", 
                      template='plotly_white',
                      legend=dict(orientation="h",  
                                xanchor="center", 
                                yanchor="top",  
                                x=0.5,  
                                y=-0.2))

    return fig



def seasonal_plot(df:pd.DataFrame,target_col:str, period:int) -> go.Figure:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with historical data.
        :param period: int - Seasonal period.
    Ouputs:
        go.Figure - A plot comparing all seasonal periods.
    """

    #creating the x-axis for the plot
    x_axis = [i for i in range(1,period+1)]

    fig = go.Figure()

    #plotting each season, checking each has the right length
    for season in range(len(df) // period):
        index = season * period

        fig.add_trace(go.Scatter(x=x_axis,
                                y=df[target_col].iloc[index:index+period-1],
                                name = f"{df.index[index]} - {df.index[index+period-1]}"))
    remainder = len(df) % period

    if remainder != 0:        
        fig.add_trace(go.Scatter(x=x_axis,
                                y=df[target_col].iloc[-remainder:],
                                name = f"{df.index[-remainder]} - {df.index[-1]}"))
        
    fig.update_layout(height=600,title_text='Seasonal data', template = 'plotly_white')
        
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text=target_col)
        
    return fig
    

def seasonal_change(df:pd.DataFrame, target_col:str, period:int) -> go.Figure:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with historical data.
        :param period: int - Seasonal period.
    Ouputs:
        go.Figure - A plot comparing each stage of each season.
    """
    
    #creating titles for each plot
    titles = []
    for i in range(1,period+1):
        titles.append(f"Stage {i}")

    fig = make_subplots(
            rows=1, cols=period,
            subplot_titles= titles,
            horizontal_spacing=0.043)

    #creating a plot for each stage in the season, checking if there is an incomplete season 
    remainder = len(df) % period

    for plot_no in range(0,period):

        if plot_no < remainder:
            indexes = [plot_no + period * i for i in range(len(df) // period+1)]
        
        else:
            indexes = [plot_no + period * i for i in range(len(df) // period)]

        x_axis = df.index[indexes]

        fig.add_trace(go.Scatter(x=x_axis,y=df[target_col].iloc[indexes],
                                 showlegend=False),
                    row=1, col=plot_no+1)
        
        fig.update_xaxes(title_text='Date',row=1, col=plot_no+1)
        fig.update_yaxes(title_text=target_col,row=1, col=plot_no+1)

    fig.update_layout(height=600,template='plotly_white',
                      title_text = 'Plots of each stage in every season')

    return fig


def future_forecast_data(df:pd.DataFrame, target_col:str, output_forecast:pd.DataFrame, fill=True) -> go.Figure:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with historical data.
        :param output_forecast: pandas.DataFrame - Data frame with the desired forecast and the lower and upper bounds of the prediction intervals as columns,
                                                   as outputted from the benchmark forecasts.
        :param fill: bool - Toggle whether to fill the space between each prediction interval.
    Ouputs:
        go.Figure - A plot of the oberserved data, the forecast and the prediction intervals.
    """
    
    fig=go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df[target_col],
                             line=dict(color='#00789c'), name = 'Observed Data'))
    
    #iterating through the columns for each prediction interval and plotting them with varying opacities
    #and filling the space between the predcition intervals if fill is True
    no_columns = len(output_forecast.columns)

    for index in range(1,no_columns,2):
        lower_pi_name = output_forecast.columns[index]
        upper_pi_name = output_forecast.columns[index+1]

        line_opacity = 1 - index / no_columns / 2

        fill_opacity = 0.5 - index / no_columns / 2

        fig.add_trace(go.Scatter(x=output_forecast.index, y=output_forecast[lower_pi_name],
                            name = f'{lower_pi_name[:2]}% Prediction interval',
                            line = dict(color=f'rgba(0,120,156,{line_opacity})')))
        
        if fill:


            fig.add_trace(go.Scatter(x=output_forecast.index, y=output_forecast[upper_pi_name],
                                    line = dict(color=f'rgba(0,120,156,{line_opacity})'),
                                    fill = 'tonexty',
                                    fillcolor=f'rgba(0,120,156,{fill_opacity})',
                                    showlegend=False))
            
        else:
             
             fig.add_trace(go.Scatter(x=output_forecast.index, y=output_forecast[upper_pi_name],
                                    line = dict(color=f'rgba(0,120,156,{line_opacity})'),
                                    showlegend=False))
        
    fig.add_trace(go.Scatter(x=output_forecast.index, y=output_forecast['forecast'],
                            name='Forecast',
                            line = dict(color='#d1495b')))


    
   
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text=target_col)

    fig.update_layout(height = 600,
                      template = 'plotly_white',
                      title_text='Forecast and prediction intervals',
                      legend=dict(orientation="h",  
                                xanchor="center", 
                                yanchor="top",  
                                x=0.5,  
                                y=-0.2))

    return fig



def future_forecast(df:pd.DataFrame,target_col:str, model:str, horizon:int, period:int=1, pred_width:list = [95,80], fill=True,**kwargs) -> go.Figure:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data.
        :param target_col: str - Column with historical data.
        :param model: str - The model to forecast, one of the keys from model_dict.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param period: int - Seasonal period.
        :param pred_width: list, 0 <= pred_width < 100 - List of widths of prediction intervals.
        :param fill: bool - Toggle whether to fill the space between each prediction interval.
        :param **kwargs - Keyword arguments for the chosen model.

    Output:
        go.Figure - A plot of the oberserved data, the forecast and the prediction intervals.
    """
    #calulating the forecast for the selected model and plotting this using the previous function
    output_forecast = benchmark_forecast(df,target_col,horizon,model,period,pred_width,**kwargs)

    fig = future_forecast_data(df,target_col, output_forecast,fill)

    return fig



def bootstrap_sim_graph(df:pd.DataFrame, target_col:str, horizon:int, model:str, repetitions:int=100,period:int=1, pred_width:list = [95,80]) -> go.Figure:
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with historical data.
        :param horizon: int - Number of timesteps forecasted into the future.
        :param model: str - One of {'naive','drift','mean'}, the model to simulate the forecast.
        :param repetitions: int - Number of bootstrap repetitions.
        :param period: int - Seasonal period.
        :param pred_width : list, 0 <= pred_width < 100 - List of widths of prediction intervals.
    Ouputs:
        go.Figure - A plot of all simulated bootstrapped forecasts and the prediction interval.
    """

    #storing each simulation, the forecast and the prediction intervals
    bs_fig = go.Figure()

    output_forecast, forecast_df = bs_benchmark_forecast(df, target_col, model, horizon, period,
                                                         repetitions, pred_width, simulations=True)
    
    #plotting each simulation
    for run in range(repetitions):
         
        bs_fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[f'run_{run}'],
                                    showlegend=False,
                                    line=dict(color='#9db2bf'),
                                    opacity=0.2))
        
    #finding the forecast and the prediction intervals
    fig = future_forecast_data(df, target_col, output_forecast,fill=False)


    #adding this figure to the simulations
    for trace in fig.data:

        bs_fig.add_trace(trace)
        
    
    bs_fig.add_trace(go.Scatter(x=df.index, y=df[target_col],
                                name='Observed data',
                                line=dict(color='#00789c')))
    
    
    bs_fig.update_xaxes(title_text = 'Date')
    bs_fig.update_yaxes(title_text = target_col)

    
    bs_fig.update_layout(title_text = 'Bootstrap simulations and prediction intervals',
                         template = 'plotly_white',
                         height=600,
                         legend=dict(orientation="h",  
                                xanchor="center", 
                                yanchor="top",  
                                x=0.5,  
                                y=-0.2))
    
    return bs_fig


def cross_val_graph(df:pd.DataFrame,target_col:str,model:str,period:int=1,n_splits:int=5,test_size:int=None,**kwargs) -> go.Figure:
    """
    Test forecasting method/s on observed data

    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col:str - Column with historical data.
        :param n_splits: int - Number of folds.
        :param test_size: int -  Forecast horizon during each fold.
        :param model: dict - The model to see the cross validation of, one of the keys from model_dict
        :param **kwargs - Keyword arguments for the selected model
    Outputs:
        go.Figure - A plot of each fold and it's respective forecast against the observed data
    """

    fig = go.Figure()

    #running cross validation for the selected model
    train_test_dict = cross_val(df,target_col,period,n_splits, test_size,{model:model_dict[model]},**kwargs)

    #plotting the observed data and forecast from the cross validation 
    fig.add_trace(go.Scatter(x=df.index, y=df[target_col],
                                name = 'Observed data',
                                line = dict(color = '#00789c')))

    fig.add_trace(go.Scatter(x=train_test_dict[model].index, y=train_test_dict[model]['forecast'],
                                name  = f'{model} forecast',
                                line = dict(color = '#d1495b')))
    
    fig.update_xaxes(title_text = 'Date')
    fig.update_yaxes(title_text = target_col)

    #finding the first first dates for each fold and plotting these as vertical lines
    fold_min_stats = train_test_dict[model].copy().reset_index().groupby(by='fold').min()
    model_retrained = fold_min_stats.iloc[:,0].to_list()

    for retrained in model_retrained:

        fig.add_vline(retrained, line_width=1.5,
                        line_dash="dash",
                        line_color="green")

    fig.update_layout(template = 'plotly_white',
                      legend=dict(orientation="h",  
                                xanchor="center", 
                                yanchor="top",  
                                x=0.5,  
                                y=-0.2))
    
    return fig


def metric_bar(df:pd.DataFrame,target_col:str, period=1, eval_metric:str='mean_squared_error', n_splits:int=5, test_size:int=None, model:dict = model_dict) -> go.Figure:
    
    """
    Inputs:
        :param df: pandas.DataFrame - Historical time series data with date-time index.
        :param target_col: str - Column with historical data.
        :param eval_metric - one of {'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error', 'max_error'},
                             the metric to evaluate each model by.
        :param n_splits: int - Number of folds.
        :param test_size: int -  Forecast horizon during each fold.
        :param model: dict - Dictionary with the models (str) as keys and their respective functions as values.
        **modelkwargs - Keyword arguments for the model chosen.

    Outputs:
        go.Figure - Bar graph comparing the value of the evaluation metric for each model.
    """

    #finding the value of the evaluation metric for each model
    eval_frame = forecast_metrics(df,target_col,period, n_splits, test_size, model)
    eval_scores = eval_frame[eval_metric]

    fig= go.Figure()

    #plotting these values
    fig = fig.add_trace(go.Bar(x = [key for key in model], y = eval_scores,
                 marker_color = '#00789c'))
    
    fig.update_layout(title_text = f'Values of the {eval_metric} for the methods below',
                      template = 'plotly_white')

    return fig
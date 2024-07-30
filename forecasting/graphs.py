# Graphs for both output forecasts & evaluation
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import normaltest
from statsmodels.tsa.seasonal import STL
import numpy as np
from .prediction_intervals import forecast_dates, naive_error, drift_error, mean_error, bs_forecast, bs_output



def resid_diagnostic(df:pd.DataFrame,target_col:str, method:str, period:int=1) -> go.Figure:
    #fix legend (underneath first plot)
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param df_target_col: str - Column with historical data
        :param fitted_forecast: list - One-step forecasts stored with a date-time index, as outputted by 'method'_error function
    Outputs:
        go.Figure - Subplots of the residuals, the ACF and a histogram of the residuals
    """

    plot_frame = pd.DataFrame()

    if method == 'naive':
        plot_frame['error'] = naive_error(df,target_col)['error']

    elif method == 'seasonal naive':
        plot_frame['error'] = naive_error(df,target_col, period)['error']

    elif method == 'drift':
        plot_frame['error'] = drift_error(df,target_col)['error']

    elif method == 'mean':
        plot_frame['error'] = mean_error(df,target_col)['error']

    cor_coeffs=[]

    #calculating the correlation coeficients for the lagged values for the ACF

    for i in range(0,len(df)):
        plot_frame['corr'] = df[target_col].shift(i+1)
        cor_coeffs.append(df[target_col].corr(plot_frame['corr']))

    #calculating the 2-sided chi-squared p-value for a normal hypothesis test on the residuals

    #plotting a bar chart of the residuals
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2},None],[{}, {}]],
        subplot_titles=("Residual plot","Autocorrelation funciton", "Residual histogram"))

    fig.add_trace(go.Bar(x=plot_frame.index, y=plot_frame['error'],
                        marker_color='#00789c',
                        showlegend=False),
                    row=1, col=1)
    
    fig.add_trace(go.Scatter(x=plot_frame.index,
                            y=[np.mean(plot_frame['error'])] * len(df),
                            name = 'Mean value',
                            line=dict(color='#d1495b',dash = 'dash')),
                row = 1, col=1)
    
    
     #plotting the ACF
    fig.add_trace(go.Bar(x=plot_frame.index, y=cor_coeffs,
                        marker_color = '#00789c',
                        showlegend=False),
                    row=2, col=1)
                    
    fig.add_trace(go.Scatter(x=plot_frame.index, y=[1.96 / np.sqrt(len(df))] * len(df),
                            line=dict(color = '#d1495b', dash = 'dash'),
                            showlegend=False,
                            name = '95% bound'),
                    row=2, col=1)
                    
    fig.add_trace(go.Scatter(x=plot_frame.index, y=[-1.96 / np.sqrt(len(df))] * len(df),
                            line=dict(color = '#d1495b', dash = 'dash'),
                            showlegend=False),
                    row=2, col=1)
    
    #plotting a histogram of the residuals
    fig.add_trace(go.Histogram(x=plot_frame['error'],
                            opacity=0.7,
                            marker_color='#66a182',
                            nbinsx=200,showlegend=False),
                    row=2, col=2)

    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='Error from the forecast', row=1, col=1)

    fig.update_xaxes(title_text='Lag', row=2, col=1)
    fig.update_yaxes(title_text='Correlation coefficient', row=2, col=1)

    fig.update_xaxes(title_text='Error from the forecast', row=2, col=2)
    fig.update_yaxes(title_text='Frequency', row=2, col=2)

    fig.update_layout(height=1000,
                    title_text="Residual Diagnostics",
                    template='plotly_white',
                    legend2=dict(yanchor="bottom",
                                orientation="h",
                                y=-0.1,
                                xanchor="left",
                                x=0.01),
                    legend4=dict(yanchor="bottom",
                                orientation="h",
                                y=-0.1,
                                xanchor="left",
                                x=0.01))

    return fig



def fitted_forecast_graph(df:pd.DataFrame,target_col:str,method:str,period:int=1) -> go.Figure:
    
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - The column of df the observed data is located
        :param method: str - one of {'naive','seasonal naive','drift','mean'} to see the forecast against the data
        :param period: int - seasonal period
    Ouputs:
        go.Figure - A plot of the observed data and the fit of the forecast selected
    """

    fig=go.Figure()

    fig.add_trace(go.Scatter(x=df.index,y=df[target_col],
                             line=dict(color='#00789c'),
                             name='Observed data'))

    #calculating the fitted forecast
    if method == 'naive':
        fitted_forecast = naive_error(df,target_col)['fitted forecast']

    elif method == 'seasonal naive':
        fitted_forecast = naive_error(df,target_col, period)['fitted forecast']

    elif method == 'drift':
        fitted_forecast = drift_error(df,target_col)['fitted forecast']

    elif method == 'mean':
        fitted_forecast = mean_error(df,target_col)['fitted forecast']


    fig.add_trace(go.Scatter(x=df.index,y=fitted_forecast,
                        line=dict(color='#d1495b'),
                        name='Fitted forecast'))
    
    fig.update_layout(height=600,
                    title_text=f'Observed data and fitted forecast with {method} method',
                    legend=dict(orientation="h",  
                                xanchor="center", 
                                yanchor="top",  
                                x=0.5,  
                                y=-0.2))

    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text= target_col)

    
    
    return fig



def decomp(df:pd.DataFrame, target_col:str, period:int, **STLkwargs) -> tuple:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - The column of df the observed data is located
        :param period: int - seasonal period
        :param **STLkwargs - keyword arguments for the statsmodels STL function
    Ouputs:
        tuple: the observed data decomposed into the trend, seasonal and remainder data using statsmodels STL
    """

    stl = STL(df[target_col],period = period, **STLkwargs).fit()
    trend = stl.trend
    seasonal = stl.seasonal
    remainder = stl.resid

    return trend, seasonal, remainder



def decomp_plot(df:pd.DataFrame, target_col:str, period:int, **STLkwargs) -> go.Figure:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - The column of df the observed data is located
        :param period: int - seasonal period
        :param **STLkwargs - keyword arguments for the statsmodels STL function
    Ouputs:
        go.Figure - A plot of the observed data and the fit of the forecast selected
    """

    (trend, seasonal, remainder) = decomp(df,target_col,period,**STLkwargs)

    fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=("Observed data","Trend", "Seasonal", "Remainder"))

    fig.add_trace(go.Scatter(x=df.index, y=df[target_col]),
                    row=1, col=1)

    fig.add_trace(go.Scatter(x=trend.index, y=trend.values),
                    row=2, col=1)
    
    fig.add_trace(go.Scatter(x=seasonal.index,y=seasonal.values),
                    row=3, col=1)
    
    fig.add_trace(go.Scatter(x=remainder.index, y=remainder.values),
                    row=4, col=1),

    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text=target_col, row=1, col=1)

    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text=target_col, row=2, col=1)

    fig.update_xaxes(title_text='Date', row=3, col=1)
    fig.update_yaxes(title_text=target_col, row=3, col=1)

    fig.update_xaxes(title_text='Date', row=4, col=1)
    fig.update_yaxes(title_text=target_col, row=4, col=1)

    fig.update_layout(height=2000,
                      showlegend=False, 
                      title_text="Decomposition of data", 
                      template='plotly_white')

    return fig



def seasonal_plot(df:pd.DataFrame,target_col:str, period:int) -> go.Figure:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - The column of df the observed data is located
        :param period: int - seasonal period
    Ouputs:
        go.Figure - A plot of all of the seasonal periods
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
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - The column of df the observed data is located
        :param period: int - seasonal period
    Ouputs:
        go.Figure - A plot of each stage of the period and how it changes over time
    """
    
    titles = []
    for i in range(1,period+1):
        titles.append(f"Stage {i}")

    fig = make_subplots(
            rows=1, cols=period,
            subplot_titles= titles,
            horizontal_spacing=0.043)

    #creating a plot for each stage in the season, and checking if there is an incomplete season 
    remainder = len(df) % period

    for plot_no in range(0,period):

        if plot_no < remainder:
            indexes = [plot_no + period * i for i in range(len(df) // period+1)]
        
        else:
            indexes = [plot_no + period * i for i in range(len(df) // period)]

        x_axis = df.index[indexes]

        fig.add_trace(go.Scatter(x=x_axis,y=df[target_col].iloc[indexes]),
                    row=1, col=plot_no+1)
        
        fig.update_xaxes(title_text='Date',row=1, col=plot_no+1)
        fig.update_yaxes(title_text=target_col,row=1, col=plot_no+1)

    fig.update_layout(height=600,showlegend=False,template='plotly_white',
                      title_text = 'Plots of each stage in every season')

    return fig


def future_forecast(df:pd.DataFrame, target_col:str, output_forecast:pd.DataFrame) -> go.Figure:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - The column of df the observed data is located
        :param output_forecast: pd.DataFrame - a data frame with the forecasted dates and the forecast,
                                        the lower and the upper bounds for the prediction intervals as columns,
                                        as outputted from prediction_intervals
    Ouputs:
        go.Figure - A plot of the oberserved data, the forecast and the prediction interval
    """
    
    fig=go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df[target_col],
                             line=dict(color='#00789c'), name = 'Observed Data'))
    
    #plotting the prediction intervals from pred_width and giving each the right opacity
    no_columns = len(output_forecast.columns)
    
    for index in range(1,no_columns,2):
        lower_pi_name = output_forecast.columns[index]
        upper_pi_name = output_forecast.columns[index+1]

        fill_opacity = 0.5 - index / no_columns / 2

        fig.add_trace(go.Scatter(x=output_forecast.index, y=output_forecast[lower_pi_name],
                            name = f'{lower_pi_name[:2]}% Prediction interval',
                            line = dict(color='#9db2bf')))
    
        fig.add_trace(go.Scatter(x=output_forecast.index, y=output_forecast[upper_pi_name],
                                line = dict(color='#9db2bf'),
                                fill = 'tonexty',
                                fillcolor=f'rgba(0,120,156,{fill_opacity})',
                                showlegend=False))
        
    fig.add_trace(go.Scatter(x=output_forecast.index, y=output_forecast['forecast'],
                            name='Forecast',
                            line = dict(color='#d1495b')))


    
   
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text=target_col)

    fig.update_layout(height = 600, template = 'plotly_white',title_text='Forecast and prediction intervals')

    return fig


def bootstrap_sim_graph(df:pd.DataFrame, target_col:str, horizon:int, method:str,repetitions:int,period:int=1, pred_width:float = 95) -> go.Figure:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - The column of df the observed data is located
        :param horizon: int - number of timesteps forecasted into the future
        :param model: str - one of {'naive','seasonal naive','drift','mean'}, the method to simulate the forecast
        :param repetitions: int - Number of bootstrap repetitions
        :param period: int - Seasonal period
        :param pred_width : float - 0 <= pred_width < 100 width of prediction interval
        :param show_simulations: bool - Toggle whether to see the bootstrapped simulations
    Ouputs:
        go.Figure - A plot of all simulated bootstrapped forecasts and the prediction interval
    """

    #finding the errors between the required fitted forecast and the data
    forecast_df = forecast_dates(df,horizon)
    error = pd.DataFrame()

    if method == 'naive':
        error = naive_error(df,target_col)['error']

    elif method == 'seasonal naive':
        error = naive_error(df,target_col,period)['error']

    elif method == 'drift':
        error = drift_error(df,target_col)['error']

    elif method == 'mean':
        error = mean_error(df,target_col)['error']


    bs_fig = go.Figure()

    #randomly sampling from these errors and graphing them
    for run in range(repetitions):
        forecast_df[f'run {run}'] = bs_forecast(df,target_col,horizon,error)

        bs_fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[f'run {run}'],
                                    showlegend=False,
                                    line=dict(color='#9db2bf'),
                                    opacity=0.25))
        
    
    bs_fig.add_trace(go.Scatter(x=df.index, y=df[target_col],
                                name='Observed data',
                                line=dict(color='#00789c')))
        
    pred_int = bs_output(forecast_df, pred_width)

    bs_fig.add_trace(go.Scatter(x=forecast_df.index, y=pred_int['forecast'],
                                name='Mean of bootstrapped forecasts',
                                line=dict(color='#d1495b')))
    
    #graphing the prediction intervals as darker than the simulations
    no_columns = len(pred_int.columns)
    
    for index in range(1,no_columns,2):

        lower_pi_name = pred_int.columns[index]
        upper_pi_name = pred_int.columns[index+1]

        bs_fig.add_trace(go.Scatter(x=pred_int.index, y=pred_int[lower_pi_name],
                                    name = f'{lower_pi_name[:2]}% Prediction interval',
                                    line = dict(color='#9db2bf')))
    
        bs_fig.add_trace(go.Scatter(x=pred_int.index, y=pred_int[upper_pi_name],
                                    line = dict(color='#9db2bf'),
                                    showlegend=False))
        
    
        
    bs_fig.update_xaxes(title_text='Date')
    bs_fig.update_yaxes(title_text=target_col)

    bs_fig.update_layout(height=600,
                      title_text='Bootstrapped precition interval',
                      template='plotly_white')
    
    return bs_fig
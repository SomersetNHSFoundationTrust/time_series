# Graphs for both output forecasts & evaluation
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import normaltest
from statsmodels.tsa.seasonal import STL
import numpy as np
from .bootstrap_naive_models import bs_benchmark_forecast
from .more_models import benchmark_fit, benchmark_forecast



def resid_diagnostic(df:pd.DataFrame,target_col:str, method:str, period:int=1, **kwargs) -> go.Figure:
    #fix legend (underneath first plot)
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param df_target_col: str - Column with historical data
        :param method: str - one of {'naive','drift','mean','ETS','ARIMA'} to see the forecast against the data
        :param **kwargs - Keyword arguments for the sktime functions AutoETS or AutoARIMA
    Outputs:
        go.Figure - Subplots of the residuals, the ACF and a histogram of the residuals
    """

    plot_frame = pd.DataFrame()

    fitted_forecast = benchmark_fit(df, target_col, method, period,**kwargs)
    plot_frame['error'] = df[target_col] - fitted_forecast['fitted forecast']

    cor_coeffs=[]

    #calculating the correlation coeficients for the lagged values for the ACF

    for i in range(0,len(df)):
        plot_frame['shift'] = plot_frame['error'].shift(i+1)
        cor_coeffs.append(plot_frame['error'].corr(plot_frame['shift']))


    #plotting a bar chart of the residuals
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2},None],[{}, {}]],
        subplot_titles=("Residual plot","Autocorrelation function", "Residual histogram"))

    fig.add_trace(go.Bar(x=plot_frame.index, y=plot_frame['error'],
                        marker_color='#00789c',
                        showlegend=False),
                    row=1, col=1)
    
    fig.add_trace(go.Scatter(x=plot_frame.index,
                            y=[np.mean(plot_frame['error'])] * len(df),
                            name = 'Mean value',
                            line=dict(color='black')), row = 1, col=1)
    
    
    #plotting the ACF
    fig.add_trace(go.Bar(x=plot_frame.index, y=cor_coeffs,
                        marker_color = '#00789c',
                        showlegend=False),
                    row=2, col=1)
                    
    fig.add_trace(go.Scatter(x=plot_frame.index, y=[1.96 / np.sqrt(len(df))] * len(df),
                            line=dict(color = '#d1495b', dash = 'dash'),
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

    fig.update_xaxes(title_text='Error', row=2, col=2)
    fig.update_yaxes(title_text='Frequency', row=2, col=2)

    fig.update_layout(height=1000,
                    title_text="Residual Diagnostics",
                    template='plotly_white',
                    legend=dict(orientation="h",  
                                xanchor="center", 
                                yanchor="top",  
                                x=0.5,  
                                y=-0.2))
    
    #calculating the 2-sided chi-squared p-value for a normal hypothesis test on the residuals
    p_value = normaltest(plot_frame['error']).pvalue
    print(f'The p-value for the 2-sided chi-squared test for a normal hypotheis test on the residuals is {p_value}')

    return fig



def fitted_forecast_graph(df:pd.DataFrame,target_col:str,method:str,period:int=1, **kwargs) -> go.Figure:
    
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - The column of df the observed data is located
        :param method: str - one of {'naive','drift','mean','ETS','ARIMA'} to see the forecast against the data
        :param period: int - seasonal period
        :param **kwargs - Keyword arguments for the sktime functions AutoETS or AutoARIMA
    Ouputs:
        go.Figure - A plot of the observed data and the fit of the forecast selected
    """

    fig=go.Figure()

    fig.add_trace(go.Scatter(x=df.index,y=df[target_col],
                             line=dict(color='#00789c'),
                             name='Observed data'))

    #calculating the fitted forecast
    fitted_forecast = benchmark_fit(df,target_col, method, period, **kwargs)


    fig.add_trace(go.Scatter(x=df.index,y=fitted_forecast['fitted forecast'],
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

    fig.add_trace(go.Scatter(x=df.index, y=df[target_col],
                             name = 'Observed data'),
                    row=1, col=1)

    fig.add_trace(go.Scatter(x=trend.index, y=trend.values,
                             showlegend=False),
                    row=2, col=1)
    
    fig.add_trace(go.Scatter(x=seasonal.index,y=seasonal.values,
                             name = 'Observed data'),
                    row=3, col=1)
    
    fig.add_trace(go.Scatter(x=remainder.index, y=remainder.values,
                             name = 'Observed data'),
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
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - The column of df the observed data is located
        :param output_forecast: pd.DataFrame - a data frame with the forecasted dates and the forecast,
                                        the lower and the upper bounds for the prediction intervals as columns,
                                        as outputted from prediction_intervals
        :param fill: bool - Toggle whether to fill the gap between each prediction interval
    Ouputs:
        go.Figure - A plot of the oberserved data, the forecast and the prediction interval
    """
    
    fig=go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df[target_col],
                             line=dict(color='#00789c'), name = 'Observed Data'))
    
    no_columns = len(output_forecast.columns)


    #plotting the prediction intervals using the columns of output_forecast,filling the space between if fill is True
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



def future_forecast(df:pd.DataFrame,target_col:str, method:str, horizon:int, period:int=1,
                    bootstrap:bool = False, repetitions:int = 100, pred_width:list = [95,80], fill=True,**kwargs) -> go.Figure:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data
        :param target_col: str - column with historical data
        :param method: str - one of {'naive','drift','mean','ETS','ARIMA'}, the method to simulate the forecast
        :param horizon: int - Number of timesteps forecasted into the future
        :param period: int - Seasonal period
        :param bootstrap: bool - toggle bootstrap or normal prediction interval
        :param repetitions: int - Number of bootstrap repetitions
        :param pred_width: list - 0 <= pred_width < 100 list of widths of prediction intervals
        :param fill: bool - Toggle whether to fill the gap between each prediction interval
        :param **kwargs - Keyword arguments for the sktime functions AutoETS or AutoARIMA

    Output:
        pandas.DataFrame: a bootstrapped or normal prediction interval for df
    """

    output_forecast = benchmark_forecast(df,target_col, method, horizon, period, bootstrap,repetitions, pred_width,**kwargs)

    fig = future_forecast_data(df,target_col, output_forecast,fill)

    return fig



def bootstrap_sim_graph(df:pd.DataFrame, target_col:str, horizon:int, method:str, repetitions:int=100,period:int=1, pred_width:list = [95,80]) -> go.Figure:
    """
    Inputs:
        :param df: pd.DataFrame - Historical time series data with date-time index
        :param target_col: str - The column of df the observed data is located
        :param horizon: int - number of timesteps forecasted into the future
        :param model: str - one of {'naive','drift','mean'}, the method to simulate the forecast
        :param repetitions: int - Number of bootstrap repetitions
        :param period: int - Seasonal period
        :param pred_width : list - 0 <= pred_width < 100 list of widths of prediction intervals
    Ouputs:
        go.Figure - A plot of all simulated bootstrapped forecasts and the prediction interval
    """

    #storing each simulation, the forecast and the prediction intervals
    bs_fig = go.Figure()

    output_forecast, forecast_df = bs_benchmark_forecast(df, target_col, method, horizon, period,
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
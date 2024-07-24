# Graphs for both output forecasts & evaluation
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import normaltest
from statsmodels.tsa.seasonal import STL



def resid_diagnostic(df:pd.DataFrame,df_target_col:str, fcst:pd.Series) -> go.Figure:
    """
    Inputs:
        df:pd.DataFrame - Historical time series data with date-time index
        df_target_col:str - column with historical data
        fcst:pd.Series - One-step forecasts stored with a date-time index
    Outputs:
        go.Figure - Subplots of the residuals, the ACF and a histogram of the residuals
    """

    plot_frame = pd.DataFrame(index = df.index)
    plot_frame['error'] = df[df_target_col] - fcst
    cor_coeffs=[]

    #calculating the correlation coeficients for the lagged values for the ACF

    for i in range(0,1820):
        plot_frame['corr'] = df['y'].shift(i+1)
        cor_coeffs.append(df[df_target_col].corr(plot_frame['corr']))

    #calculating the 2-sided chi-squared p-value for a normal hypothesis test on the residuals
    p_value = normaltest(plot_frame['error'].values).pvalue


    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2},None],[{}, {}]],
        subplot_titles=("Residual plot","Autocorrelation funciton", f"Residual histogram with p-value {p_value}"))

    fig.add_trace(go.Scatter(x=plot_frame.index, y=plot_frame['error']),
                    row=1, col=1)

    fig.add_trace(go.Bar(x=plot_frame.index, y=cor_coeffs),
                  row=2, col=1)
                  
    fig.add_trace(go.Scatter(x=plot_frame.index, y=1.96 / len(df)),
                  row=2, col=1)
                  
    fig.add_trace(go.Scatter(x=plot_frame.index, y=-1.96 / len(df)),
                    row=2, col=1)
    
    fig.add_trace(go.Histogram(x=plot_frame['error']),
                    row=2, col=2)

    fig.update_layout(showlegend=False, title_text="Residual Diagnostics")

    return fig


def forecast_with_data(df:pd.DataFrame,target_col:str,model:str,period:int) -> go.Figure:
    """
    Inputs:
        df: pd.DataFrame - Historical time series data with date-time index
        target_col: str - The column of df the observed data is located
        model: str - one of {'naive','seasonal naive','drift','mean'} to see the forecast against the data
        period: int - seasonal period
    Ouputs:
        go.Figure - A plot of the observed data and the fit of the forecast selected
    """
    fig = px.line(x=df.index,y=df[target_col])

    if model == 'naive':
        forecast = [df[target_col][-1]] * len(df)
        fig.add_trace(go.Scatter(x=df.index,y=forecast))

    elif model == 'seasonal naive':
        season = df[target_col][-period:].tolist()
        season_mult = len(df) // period + 1
        forecast = (season * season_mult)[-len(df) % period:]
        fig.add_trace(go.Scatter(x=df.index,y=forecast))
    
    elif model == 'drift':
        first_obs = df[target_col][0]
        latest_obs = df[target_col][-1]
        slope = (latest_obs-first_obs) / len(df)
        forecast = [first_obs + i * slope for i in range(len(df))]
        fig.add_trace(go.Scatter(x=df.index,y=forecast))
    
    elif model == 'mean':
        mean = sum(df['y']) / len(df)
        forecast = [mean] * len(df)
        fig.add_trace(x=df.index,y=forecast)
    
    return fig



def decomp(df:pd.DataFrame, target_col:str, period:int, **STLkwargs) -> tuple:
    """
    Inputs:
        df: pd.DataFrame - Historical time series data with date-time index
        target_col: str - The column of df the observed data is located
        period: int - seasonal period
        **STLkwargs - keyword arguments for the statsmodels STL function
    Ouputs:
        tuple: the observed data decomposed into the trend, seasonal and remainder data using statsmodels STL
    """
    stl = STL(df[target_col],period = period, **STLkwargs).fit()
    trend = stl.trend()
    seasonal = stl.seasonal()
    remainder = stl.resid()

    return trend, seasonal, remainder



def decomp_graph(df:pd.DataFrame, target_col:str, period:int, **STLkwargs) -> go.Figure:
    """
    Inputs:
        df: pd.DataFrame - Historical time series data with date-time index
        target_col: str - The column of df the observed data is located
        period: int - seasonal period
        **STLkwargs - keyword arguments for the statsmodels STL function
    Ouputs:
        go.Figure - A plot of the observed data and the fit of the forecast selected
    """

    (trend, seasonal, remainder) = decomp(df,target_col,period,**STLkwargs)

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(f"{target_col}","Trend", "Seasonal", "Remainder"))
    
    fig.add_trace(go.Scatter(x=df.index, y=df),
                    row=1, col=1)

    fig.add_trace(go.Scatter(x=trend.index, y=trend.values),
                    row=2, col=1)
    fig.add_trace(go.Scatter(x=seasonal.index,y=seasonal.values),
                    row=3, col=1)
    fig.add_trace(go.Scatter(x=remainder.index, y=remainder.values),
                    row=4, col=1),

    fig.update_layout(showlegend=False, title_text="Decomposition of data")

    return fig



def seasonal_plot(df:pd.DataFrame,target_col:str, period:int) -> go.Figure:
    """
    Inputs:
        df: pd.DataFrame - Historical time series data with date-time index
        target_col: str - The column of df the observed data is located
        period: int - seasonal period
    Ouputs:
        go.Figure - A plot of all of the seasonal periods
    """

    x_axis = [i for i in range(1,period+1)]

    fig = go.Figure()

    for season in range(len(df) // period):
        index = season * period

        fig.add_trace(go.Scatter(x=x_axis,
                                y=df['y'].iloc[index:index+period-1],
                                name = f"{df.index[index]} - {df.index[index+period-1]}"))
    remainder = len(df) % period
    
    if remainder != 0:        
        fig.add_trace(go.Scatter(x=x_axis,
                                y=df['y'].iloc[-remainder:],
                                name = f"{df.index[-remainder]} - {df.index[-1]}"))
        
    return fig
    

def seasonal_change(df:pd.DataFrame, target_col:str, period:int) -> go.Figure:
    """
    Inputs:
        df: pd.DataFrame - Historical time series data with date-time index
        target_col: str - The column of df the observed data is located
        period: int - seasonal period
    Ouputs:
        go.Figure - A plot of each stage of the period and how it changes over time
    """
    
    titles = []
    for i in range(1,period+1):
        titles.append(f"Stage {i}")

    fig = make_subplots(
            rows=1, cols=7,
            subplot_titles= titles)

    #creating a plot for each stage in the season, and checking if there is an incomplete season 
    remainder = len(df) % period
 
    for plot_no in range(0,period):

        if plot_no < remainder:
            indexes = [plot_no + period * i for i in range(len(df) // 7+1)]
            x_axis = df.index[indexes]
        
        else:
            indexes = [plot_no + period * i for i in range(len(df) // 7 )]
            x_axis = df.index[indexes]

        fig.add_trace(go.Scatter(x=x_axis,y=df[target_col].iloc[indexes]),
                    row=1, col=plot_no+1)
    
    fig.update_traces(showlegend=False)

    return fig

def data_with_forecast(df:pd.DataFrame, target_col:str, output_forecast:pd.DataFrame) -> go.Figure:
    """
    Inputs:
        df: pd.DataFrame - Historical time series data with date-time index
        target_col: str - The column of df the observed data is located
        output_forecast: pd.DataFrame - a data frame with the forecasted dates and the forecast, the lower and the upper bounds for the prediction interval as columns
                                        as outputted from prediction_intervals
    Ouputs:
        go.Figure - A plot of the oberserved data, the forecast and the prediction interval
    """
    
    fig = px.line(x=df.index, y=df[target_col])
    fig.add_trace(go.Scatter(x=output_forecast.index, y=output_forecast['forecast']))
    fig.add_trace(go.Scatter(x=output_forecast.index, y=output_forecast['lower_pi']))
    fig.add_trace(go.Scatter(x=output_forecast.index, y=output_forecast['upper_pi']))

    return fig






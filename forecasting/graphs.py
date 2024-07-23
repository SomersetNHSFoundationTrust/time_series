# Graphs for both output forecasts & evaluation
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

def resid_diagnostic(df:pd.DataFrame,df_target_col:str, fcst:pd.Series):
    """
    Inputs:
        df:pd.DataFrame - Historical time series data with date-time index
        fcst:pd.Series - One-step forecasts stored with a date-time index
    Outputs:
        Subplots of the residuals, the ACF and a histogram of the residuals
    """

    plot_frame = pd.DataFrame(index = df.index)
    plot_frame['error'] = df[df_target_col] - fcst
    cor_coeffs=[]

    #calculating the correlation coeficients for the lagged values for the ACF

    for i in range(0,1820):
        plot_frame['corr'] = df['y'].shift(i+1)
        cor_coeffs.append(df[df_target_col].corr(plot_frame['corr']))

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2},None],[{}, {}]],
        subplot_titles=("Residual plot","Autocorrelation funciton", "Residual histogram"))

    fig.add_trace(go.Scatter(x=plot_frame.index, y=plot_frame['error']),
                    row=1, col=1)

    fig.add_trace(go.Bar(x=plot_frame.index, y=cor_coeffs),
                    row=2, col=1)
    fig.add_trace(go.Histogram(x=plot_frame['error']),
                    row=2, col=2)

    fig.update_layout(showlegend=False, title_text="")

    return fig





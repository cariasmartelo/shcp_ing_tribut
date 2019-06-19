'''
SHCP UPIT Forecasting Public Revenue
Produce graphs, test augmented dickey fuller for stationarity, make transormations.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plot_series(df, cols=None, title=None, subtitle=None, legend=None, save_to=None, 
                figsize=(12, 8)):
    '''
    Plot columns of df indicated using matplotlib. Title and Subtitle are used for
    the title.
    Inputs:
        df: DataFrame
        cols: [str]
        title: str
        subtitle: str
        legend: [str]
    Output
        Plot
        .png if indicated
    '''
    if isinstance(df, pd.core.series.Series):
        df = df.to_frame()
    if not cols:
        cols = df.columns
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[cols])
    if not legend is False:
        if not legend:
            legend = cols
        ax.legend(legend)
    if title:
        if subtitle:
            ax.set_title('{}\n{}'.format(title, subtitle))
        else:
            ax.set_title('{}'.format(title))
    ax.grid()
    if save_to:
        plt.savefig(save_to)
    plt.show()
    plt.close()


def test_stationarity(df, col):
    """

    Test stationarity using moving average statistics and Dickey-Fuller test
    Source: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    Source2: Tamara Louie, PyData LA, October 2018

    """
    
    # Determing rolling statistics
    df2 = df.loc[df[col].notna(),].copy()
    df2['rolmean'] = df2[col].rolling(window = 12, center = False).mean()
    df2['rolstd'] = df2[col].rolling(window = 12, center = False).std()
    
    # Plot rolling statistics:
    plot_series(df2, [col, 'rolmean', 'rolstd'],
                'Rolling Mean & Standard Deviation for ', 
                subtitle='{}'.format(col),
                legend=['Original', 'Rolling Mean', 'Rolling Std'],
                figsize=(5, 4))
    plt.show()
    plt.close()
    
    # Perform Dickey-Fuller test:
    # Null Hypothesis (H_0): time series is not stationary
    # Alternate Hypothesis (H_1): time series is stationary
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(df2[col], 
                      autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic',
                                  'p-value',
                                  '# Lags Used',
                                  'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    print (dfoutput)

def transformation(df, col, transform, window=None):
    '''
    Transform series indicates by transformation indicated. Window used for moving avg.
    Input:
        df: DF
        col: str
        transform: str
    Output:
        Series
    '''

    if transform == 'diff':
        return df[col].diff()

    if transform == 'log':
        return df[col].apply(lambda x: np.log(x))

    if transform == 'log_diff':
        return transformation(df, col, 'log').diff()

    if transform == 'moving_avg':
        return df[col].rolling(window = window, center = False).mean()

    if transform == 'moving_average_diff':
        return transformation(df, col, 'moving_avg', window).diff()

    if transform == 'log_moving_avg':
        return transformation(df, col, 'log').rolling(window = window,
                                                 center = False).mean

    if transform == 'log_moving_avg_diff':
        return transformation(df, col, 'log').rolling(window = window,
                                                 center = False).mean().diff()
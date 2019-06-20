'''
SHCP UPIT Forecasting Public Revenue
Produce graphs, test augmented dickey fuller for stationarity, make transormations.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plot_series(df, cols=None, title=None, subtitle=None, legend=None, save_to=None, 
                min_date=None, max_date=None, ticks='auto', ticks_freq=1, figsize=(12, 8),
                legend_out=False):
    '''
    Plot columns of df indicated using matplotlib. Title and Subtitle are used for
    the title.
    Inputs:
        df: DataFrame or Serie
        cols: [str]
        title: str
        subtitle: str
        legend: [str]
        save_to: 'str'
        min_date: str or datetime
        max_date: str or datetime
        ticks: {'auto', 'yearly', 'monthly'}
    Output
        Plot
        png if indicated
    '''
    if isinstance(df, pd.core.series.Series):
        df = df.to_frame()
    if not cols:
        cols = df.columns
    if min_date:
        df = df.loc[df.index >= min_date]
    if max_date:
        df = df.loc[df.index <= max_date]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[cols])
    if not ticks == 'auto':
        if ticks == 'yearly':
            loc = mdates.YearLocator(ticks_freq)
            ax.xaxis.set_major_locator(loc)
            for tick in ax.get_xticklabels():
                tick.set_rotation(40)
        elif ticks == 'monthly':
            loc = mdates.MonthLocator(interval = ticks_freq)
            date_fmt = mdates.DateFormatter('%Y-%m')
            ax.xaxis.set_major_formatter(date_fmt)
            ax.xaxis.set_major_locator(loc)
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
    ax.grid()
    if not legend is False:
        if not legend:
            legend = cols
        if legend_out:
            ax.legend(legend, bbox_to_anchor=(1.04,1), loc="upper left")
        else:
            ax.legend(legend)
    if title:
        if subtitle:
            ax.set_title('{}\n{}'.format(title, subtitle))
        else:
            ax.set_title('{}'.format(title))
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

def revert_transformation(transformed, applied_transformation, initial_value=None,
                          initial_date=None):
    '''
    revert the transformation of a series to its original value.
    Inputs:
        serie: Series
        applied_transformation: str {''}
    '''
    assert ((applied_transformation.endswith('diff') and initial_value) or
            (not applied_transformation.endswith('diff') and (not initial_value))),\
            'Must give initial value if transformation applied was differences'

    serie = transformed.copy()
    if initial_value:
        serie.loc[pd.to_datetime(initial_date)] = initial_value
        serie = serie.sort_index()

    if applied_transformation == 'diff':
        return serie.cumsum()

    if applied_transformation == 'log':
        return np.exp(serie)

    if applied_transformation == 'log_diff':
        serie.iloc[0] = np.log(serie.iloc[0])
        serie = serie.cumsum()
        return np.exp(serie)

def plot_acf_pacf(series, lags, col=None, save_to=None):
    '''
    Plot ACF and PACF of serie.
    Inputs:
    serie: Pandas Serie or DF
    lags: int
    col: str
    '''
    if isinstance(series, pd.core.frame.DataFrame):
        series = series[col]

    fig, ax = plt.subplots(nrows=2)
    plot_acf(series, lags=lags, ax= ax[0])
    ax[0].set_xticks(np.arange(0, lags+1))
    plot_pacf(series, lags=lags, ax=ax[1])
    ax[1].set_xticks(np.arange(0, lags+1))
    if save_to:
        plt.savefig(save_to)
    plt.show()

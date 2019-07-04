'''
SHCP UPIT Forecasting Public Revenue
Produce graphs, test augmented dickey fuller for stationarity, make transormations.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from pandas.plotting import register_matplotlib_converters
from cycler import cycler
register_matplotlib_converters()

def plot_series(df, cols=None, title=None, subtitle=None, legend=None, save_to=None, 
                min_date=None, max_date=None, ticks='auto', ticks_freq=1, figsize=(15, 8),
                legend_out=False, dpi=200, footnote=None, hline=None):
    '''
    La función plot_series sirve para hacer una gráfica de los datos de un DataFrame.
    Inputs:
        - df: DataFrame or Serie
        - cols: [str] (Nombres de las columnas a graficar) Default: Todo el DF
        - title: str (Título del gráfico). Default: None
        - subtitle: str (Subtítulo del gráfico). Default: None
        - legend: [str] (Legendas a utilizar) Default: Las legendas serán los nombres de las variables). 
        - save_to: 'str' (Filepath para guardar imagen) Defaults: None (No se guarda)
        - min_date: str or datetime. (Fecha a partir de la cual se verá la gráfica). Default: None (Se incluyen todas las fechas)
        - max_date: str or datetime. (Fecha máxima se verá la gráfica). Default: None (Se incluyen todas las fechas)
        - ticks: {'auto', 'yearly', 'monthly'}. (Que el eje X marque las lineas divisorias en meses, anños, etc) Default: Automatico
        - ticks_freq: Int, (Cada cuantos meses o anños mostrar lineas) Default: automatico
        - figsize: tuple, (Tamaño en pixeles de la figura) Default: (15, 10)
        - legend_out: Bool (Ver legenda afuera o adentro de la figura) Default: Adentro
    Output:
        - Gráfica
        - Guarda la figura si indicado.
    '''
    custom_cycler = (cycler(color=['#333f50', '#691A30', '#7f7f7f',
                                   'xkcd:khaki', 'darkgreen', 'darkblue',
                                   'crimson', 'gold']))


    if isinstance(df, pd.core.series.Series):
        df = df.to_frame()
    if not cols:
        cols = df.columns
    if min_date:
        df = df.loc[df.index >= min_date]
    if max_date:
        df = df.loc[df.index <= max_date]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_prop_cycle(custom_cycler) 
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
        elif ticks == 'quarterly':
            loc = mdates.MonthLocator(bymonth=range(1, 13, 3))
            date_fmt = mdates.DateFormatter('%Y-%m')
            ax.xaxis.set_major_formatter(date_fmt)
            ax.xaxis.set_major_locator(loc)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
    ax.set_prop_cycle(custom_cycler) 
    if (df[cols].mean() < 1000).all():
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
    else:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
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
            ax.set_title('{}\n{}'.format(title, subtitle), fontsize=14)
        else:
            ax.set_title('{}'.format(title), fontsize=14)
    if footnote:
        plt.annotate(footnote, (0,0), (0, -50), xycoords='axes fraction',
                     textcoords='offset points', va='top')
    if save_to:
        plt.savefig(save_to, bbox_inches="tight")
    if not hline is None:
        ax.axhline(y=hline, linewidth=.8, color='k')


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

def plot_acf_pacf(series, lags, series_name, col=None, save_to=None):
    '''
    Plot ACF and PACF of serie.
    Inputs:
    serie: Pandas Serie or DF
    lags: int
    col: str
    '''
    if isinstance(series, pd.core.frame.DataFrame):
        series = series[col]

    fig, ax = plt.subplots(nrows=2, figsize=(14, 8))
    plot_acf(series, lags=lags, ax= ax[0])
    ax[0].set_xticks(np.arange(0, lags+1))
    plot_pacf(series, lags=lags, ax=ax[1])
    ax[1].set_xticks(np.arange(0, lags+1))
    if save_to:
        plt.savefig(save_to)
    fig.suptitle('Autocorrelación y autocorrelación parcial de \n{}'
                     .format(" ".join(series_name.split('_')).capitalize()),
                 fontsize=14)

    plt.show()

def cross_tab(df, cols, years, ratios=False, cols_for_tot=None, for_plot=False,
              notna=True, absolute_change=False):
    '''
    Make cross tab of columns by month
    df: DF
    cols: [str]
    years: [int]
    ratios: Bool, Indicates if the variable are ratios.
    cols_for_tot: [[str]] If the variables are ratios, it need the original variables
                  to calculate the ratios again and get the totals.
    for_plot: Bool, If True, the cross tab is formatted with months as columns.
    notna: Bool: Do not include months with NA values
    absolute_change:False, Compute the change as absolute or percentage.
    '''
    meses = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
             7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre',
             12: 'Diciembre'}
    df2 = df.copy()
    df2['fecha'] = df2.index
    df2['anio'] = df2['fecha'].map(lambda x: x.year)
    df2['mes'] = df2['fecha'].map(lambda x: x.month)
    df2['mes_n'] = df2['mes'].map(meses)
    meses_orden = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio',\
               'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre', 'Total']

    if for_plot:
        # Esre produce un cuadro muy sencillo, solo para despues hacer grafica
        assert len(cols) == 1, "If for plot, only give one col"
        df2 = df2.loc[df2['anio'].isin(years)]
        cross_tab = df2.pivot(
        index='anio', columns='mes_n', values=cols[0])
        col_order = [month for month in meses_orden if month in cross_tab.columns]
        cross_tab = cross_tab[col_order]
        if notna:
            cross_tab = cross_tab.loc[cross_tab.notna().all(1)]
        return cross_tab

    else:
        # Este produce un cuadro para analizar. La parte compleja es construir
        # los totales cuando se esta tabulando una variable en porcentajes.
        # Para ello, se usan las columnas indicadas en col_for_tot. Estas son
        # las columnas en niveles que se usaron para construir los porcentajes.
        # Se usan para calcular los totales ne niveles y luedo los totales
        # en porcentajes.
        df2 = df2.loc[df2['anio'].isin(years)]
        if not ratios:
            cross_tab = df2.pivot_table(
                index='mes_n', columns='anio', values=cols, margins=True,
                margins_name='Total')
            to_drop = [(x, 'Total') for x in cols]
            cross_tab.drop(columns=to_drop, inplace=True)
        else:
            cross_tab = df2.pivot_table(
                index='mes_n', columns='anio', values=cols)
        col_order = [month for month in meses_orden if month in cross_tab.index]
        cross_tab = cross_tab.loc[col_order]
        # Eliminar los NA's
        if notna:
            cross_tab = cross_tab.loc[cross_tab.notna().all(1)]
        months = list(cross_tab.index)
        # cols_for_tot es una lista de listas, cada sub lista es una lista de dos
        # elementos, el primero el numerados y el segundo el denominador para
        # construir los %

        # Crear lista de totales
        if ratios:
            totals_list = []
            for columns in cols_for_tot:
                totals = df2.loc[df2['mes_n'].isin(months)]
                totals = totals.pivot_table(index='mes_n', columns='anio',
                                            values=columns)
                # Obtener el total de los meses incluidos en el año
                totals = totals.sum()
                # Obtener el total en prcentaje
                totals = (totals[columns[0]] / totals[columns[1]]) * 100
                totals = list(totals.values)
                totals_list += totals
            totals = pd.Series(totals_list).rename('Total').to_frame()
            # Cambiar el índice para poder concatenar con cross tabs.
            iterables = [cols, years]
            index = pd.MultiIndex.from_product(iterables)
            totals = totals.set_index(index).transpose()
            cross_tab = pd.concat([cross_tab, totals])

        for col in cols:
            if not absolute_change:
                cross_tab[(col, 'Cambio porcentual(%)')] = (
                    (cross_tab[(col, years[-1])] / 
                     cross_tab[(col, years[0])]) - 1) * 100
            else:
                cross_tab[(col, 'Cambio (puntos %)')] = (
                    cross_tab[(col, years[-1])] - 
                     cross_tab[(col, years[0])])


        cross_tab = cross_tab[cols]
        def formating(x):
            if x > 100:
                return '{:,.0f}'.format(x)
            else: 
                return '{:,.2f}'.format(x)

        return cross_tab.applymap(formating).style

def seasonally_adjust(serie):
    '''
    Función para desestacionalizar serie usando el método X13 del Censo de USA
    con la implementación en R. La función crea una función en R y la llama
    con la serie indicada.
    Inputs:
        Pandas Serie
    Output:
        Pandas Serie
    '''
    rstring="""
    function(timeseries){
    library(seasonal)
    ts = ts(timeseries, start=c(2014,1), end=c(2019,5), frequency=12)
    m <- seas(ts)
    final(m)
    }
    """
    rfunc = robjects.r(rstring)
    serie_sa = rfunc(serie)
    serie_sa = pd.Series(serie_sa, index=serie.index)
    return serie_sa


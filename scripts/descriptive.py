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
                legend_out=False, dpi=None, footnote=None, hline=None):
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

    # Establecer los colores de la gráfica. Este custom cycler es el ciclo de colores
    # que sigue la gráica.
    custom_cycler = (cycler(color=['#333f50', '#691A30', '#7f7f7f',
                                   'xkcd:khaki', 'darkgreen', 'darkblue',
                                   'crimson', 'gold']))

    # Si en vez de pasar un DataFrame, pasaron una serie, lo convertimos a DataFrame.
    if isinstance(df, pd.core.series.Series):
        df = df.to_frame()
    # Si no se especificaron columnas, se grafican todas las columnas.
    if not cols:
        cols = df.columns
    # Si se especificó fecha mínima para graficar, o fecha máxima, filtramos el
    # DataFrame pasa quedarons solo con ese rango
    if min_date:
        df = df.loc[df.index >= min_date]
    if max_date:
        df = df.loc[df.index <= max_date]

    # Creamos un eje (ax) y una figura, del tamaño especificado en figsize y con la
    # resolución especificada en dpi.
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Indicamos el ciclo de colores que usaremos.
    ax.set_prop_cycle(custom_cycler) 

    # Ploteamos las columnas del df que queremos plotear. Con df[cols] seleccionamos
    # del df, las columnas que estan especificadas en la lista cols. 
    ax.plot(df[cols])

    # Vamos a formatear los 'ticks', las marcas del eje x. Si vemos la definición de la
    # función, veremos que si no se especificó nada en ticks, el valor default es
    # 'auto'

    # Si ticks NO es auto:
    if not ticks == 'auto':
        if ticks == 'yearly':
            # Si ticks son yearly, creamos el objeto loc igual al year locator con la frecuencia
            # especificada en ticks_freq (Si no se especifica ticks_freq, el default es 1) lo que 
            # significa que las marcas del eje serán cada añ0
            loc = mdates.YearLocator(ticks_freq)
            ax.xaxis.set_major_locator(loc)
            # Rotamos los ticks para que quepan
            for tick in ax.get_xticklabels():
                tick.set_rotation(40)
        # Repetios lo mismo si se especifica 'monthly' o si se especifica 'quarterly'
        elif ticks == 'monthly':
            loc = mdates.MonthLocator(interval = ticks_freq)
            # A diferencia de año, especificamos el formato para que sea año-mes
            date_fmt = mdates.DateFormatter('%Y-%m')
            ax.xaxis.set_major_formatter(date_fmt)
            ax.xaxis.set_major_locator(loc)
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
        elif ticks == 'quarterly':
            # Si el usuario quere ver trimestres, vamos a ver todos los
            # meses que son [1, 4, 7 10], lo cual se logra con
            # range(1, 13, 3)
            loc = mdates.MonthLocator(bymonth=range(1, 13, 3))
            date_fmt = mdates.DateFormatter('%Y-%m')
            ax.xaxis.set_major_formatter(date_fmt)
            ax.xaxis.set_major_locator(loc)
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
    # Vamos a formatear el eje y. Si el promedio de los valores son menores a mil, 
    # queremos ver cifras a dos decimales. Si son mayores a mil, 
    # no queremos ver decimales. Se usó 1000 porque lo que estamos graficando
    # es porcentual o son miles de millones de pesos, no hay un intermedio
    # para preocuparnos.
    if (df[cols].mean() < 1000).all():
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
    else:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    # Activamos las lineas divisorias.
    ax.grid()
    # La legenda: Si el usuario no especifica nada, se usa como legenda los nombres
    # de las variables. Si el usuario especifica False, no se pone legenda. Si el
    # usuario especifica una lista con la legenda, se usa esa legenda. 

    # Entonces, vamos a modificar el objeto que se llama legenda solo si el usuario
    # no especifica nada. Después, vamos a decirle a la gráfica que la legenda es igual
    # a ese objeto que se llama legenda.

    # Si el usuario no especificó nada, asignamos a la variable legend igual a la
    # lista de columnas.
    if not legend is False:
        if not legend:
            legend = cols
        # Si el usuario especifica legend_out=True, la legenda se coloca afuera
        # del gráfico. Sirve cuando la legenda es muy estorbosa.
        if legend_out:
            ax.legend(legend, bbox_to_anchor=(1.04,1), loc="upper left")
        else:
            ax.legend(legend)
    # Si el usuario especifica un título, vamos a crearle título a la gráfica.
    if title:
        # Si el usuario especifica título y subtítulo, los colocamos a ambos en el
        # título. \n sirve para poner el subtítulo por debajo. Si solo se especifica
        # título, solo ponemos el título.
        if subtitle:
            ax.set_title('{}\n{}'.format(title, subtitle), fontsize=14)
        else:
            ax.set_title('{}'.format(title), fontsize=14)
    # Si el usuario especifica un pie de gráfica, lo colocamos usando el método
    # annotate.
    if footnote:
        plt.annotate(footnote, (0,0), (0, -50), xycoords='axes fraction',
                     textcoords='offset points', va='top')
    # Si el usuario especifica una linea horizontal, la colocamos donde se especifica.
    # por ejemplo, si hline=0, se pone una linea horizonta el y=0
    if not hline is None:
        ax.axhline(y=hline, linewidth=.8, color='k')
    # Si el usuario especifica una ruta para guardar la imagen, la guardamos. Si la ruta no
    # existe, habrá error.
    if save_to:
        plt.savefig(save_to, bbox_inches="tight")

    # Mostramos el gráfico
    plt.show()
    # Cerramos el gráfico.
    plt.close()


def test_stationarity(serie):
    """
    Función para hacer prueba de estacionariedad, usamos moving average y prueba 
    Dickey-Fuller test.
    Source: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    Source2: Tamara Louie, PyData LA, October 2018
    Inputs:
        Pandas Series o DF de una sola variable
    """
    
    # Determing rolling statistics
    # Creamos una serie nueva que no tenga NA's
    serie2 = serie[serie.notna()]
    # SI la variable es un DF, lo convertimos en series.
    if isinstance(serie2, pd.core.frame.DataFrame):
        serie2 = serie2.iloc[:,0]

    # Creamos un DataFrame con la serie nueva. Vamos a incluir en el DataFrame
    # la media móvil y la desviación estándar móvil. 
    df2 = serie2.to_frame()
    # Aquí creamos la columna 'rollmean y rolstd' usando la función rolling, usamos 12
    # periodos. 
    df2['rolmean'] = serie2.rolling(window = 12, center = False).mean()
    df2['rolstd'] = serie2.rolling(window = 12, center = False).std()
    
    # Graficamos la variable usando la función plot_series que definimos en este mismo
    # script.
    plot_series(df2,
                title='Rolling Mean & Standard Deviation for ', 
                subtitle='{}'.format(serie2.name),
                legend=['Original', 'Rolling Mean', 'Rolling Std'],
                figsize=(5, 4), dpi=100)
    plt.show()
    plt.close()
    
    # Perform Dickey-Fuller test:
    # Null Hypothesis (H_0): time series is not stationary
    # Alternate Hypothesis (H_1): time series is stationary
    print('Results of Dickey-Fuller Test:')
    # Creamos un objeto con los resultados de adfuller, que es una función que hace
    # la prueba de estacionariedad.
    dftest = adfuller(serie2, 
                      autolag='AIC')
    # Obtenemos los primeros cuatro resultados que son
    # el estadístico t, el pvalue, el numero de lags y el numero de observacines.
    # Y creamos un Pandas Series con eso.
    dfoutput = pd.Series(dftest[0:4], 
                         index = ['Test Statistic',
                                  'p-value',
                                  '# Lags Used',
                                  'Number of Observations Used'])

    # El quinto elemento de dftest es un diccionario con los critical values.
    # Usamos el diccionario para imprimir cuales serían los critical values
    # al 1%, 5% y 10%. Iteramos sobre el diccionario.

    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    print (dfoutput)

def transformation(serie, transform, lags=1):
    '''
    Transform series indicated by transformation indicated. Window used for moving avg.
    Input:
        df: DF
        col: str
        transform: str: {'diff', 'log', 'log_diff', 'moving_avg', 'moving_average_diff',
                         'log_moving_avg', 'log_moving_avg_diff'}
        lags: int
    Output:
        Series
    '''
    serie_copy = serie.copy()
    if isinstance(serie, pd.core.frame.DataFrame):
        serie_copy = serie.iloc[:,0].copy()

    # Si en transform se indica 'diff', la función regresa la serie con el método diff
    # aplicado. el método diff calcula la diferencia con la observación inmediata anterior,
    # pero si se especifica un numero dentro, como diff(12), calcula la diferencia con
    # el rezago 12.
    if transform == 'diff':
        return serie_copy.diff(lags)
    # Si se especifica 'log', usamos la función de numpy para calcular logaritmos
    # y la aplicamos a la serie/
    if transform == 'log':
        if min(serie_copy)  <= 0:
            serie_copy = serie_copy + abs(min(serie_copy)) + 1
        return np.log(serie_copy)

    # Si se especifica log_diff, llamamos esta función para que calcule el logaritmo,
    # y después aplicamos la diferencia. Llamar a la función que estás creando dentro de
    # esa función es seguir el método recursivo.
    if transform == 'log_diff':
        return transformation(serie_copy, 'log').diff(lags)
    # Si se especifica moving average, el usuario también tiene que especificar la ventana.
    # Aplicamos el método rolling, con el tamano especificado en window.
    if transform == 'moving_avg':
        return serie_copy.rolling(window = lags, center = False).mean()
    # Si se especifica moving average_diff, llamamos la función para moving average y al
    # resultado le aplicamos las diferencias.
    if transform == 'moving_average_diff':
        return transformation(serie_copy, 'moving_avg', window).diff(lags)

    # Si se especifica log_moving_average, llamamos la función para log, y obtenemos
    # el moving average.
    if transform == 'log_moving_avg':
        return transformation(serie_copy, 'log').rolling(window = lags,
                                                 center = False).mean
    # Si especifica 'log_moving_average_diff', obtenemos el log, obtenemos la media movil,
    # y obtenemos la diferencia.
    if transform == 'log_moving_avg_diff':
        return transformation(serie_copy, 'log').rolling(window = lags,
                                                 center = False).mean().diff(lags)

def revert_transformation(transformed, applied_transformation, initial_value=None,
                          initial_date=None):
    '''
    Revert the transformation of a series to its original value. 
    Inputs:
        transformed: Series
        applied_transformation: {'diff', 'log', 'log_diff'}
        initial_value
    '''
    # assert sirve para comprobar que algo sea cierto, y si no lo es, imprimir
    # el error que uno indique. Usamos assert para comprobar que si el usuario
    # indica que la transformación fue 'diff', también pase el valor inicial.
    # Para revertir una serie a la que se le aplicó diferencias, necesitamos el
    # valor inicial y a partir de ese valor obtener los siguientes valores
    # haciendo sumas acumuladas.
    assert ((applied_transformation.endswith('diff') and initial_value) or
            (not applied_transformation.endswith('diff') and (not initial_value))),\
            'Must give initial value if transformation applied was differences'
    # Obtenemos una copia de la serie que se pasó. Esto para no modificar la original.
    serie = transformed.copy()
    # Si se pasó un valor inciail, incluimos este valor en la fecha que se pase como
    # fecha inicial
    if initial_value:
        # Insertamos el valor incial con la fecha indicada.
        serie.loc[pd.to_datetime(initial_date)] = initial_value
        # Reordenamos la serie en orden ascendente.
        serie = serie.sort_index()

    # Si la serie se transformó a diferencias, aplicamos el método cumsum, que
    # va haciendo una suma acumulada.
    if applied_transformation == 'diff':
        return serie.cumsum()

    # Si la serie se transformó a log, la exponenciamos.
    if applied_transformation == 'log':
        return np.exp(serie)

    # Si la serie se transformó a log_diff, transformamos el valor
    # inicial que insertamos a logaritmo, aplicamos cumsum y luego
    # exponensiamos la serie.
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
    # Si pasamos un DataFrame en vez de una serie, obtenemos la serie del DataFrame
    if isinstance(series, pd.core.frame.DataFrame):
        series = series[col]

    # Creamos 1 ejes y una figura.
    fig, ax = plt.subplots(nrows=2, figsize=(14, 8))
    # Llamamos la funcion plot_acf de statsmodels.graphics, indicamos el numero de lags,
    # y lo ponemos en el primer eje: ax[0]
    plot_acf(series, lags=lags, ax= ax[0])
    # Modificamos los ticks para que vayan de 0 al lag.
    ax[0].set_xticks(np.arange(0, lags+1))
    # Llamamos la funcion plot_pacf de statsmodels.graphics, indicamos el numero de lags,
    # y lo ponemos en el segundo eje: ax[1]
    plot_pacf(series, lags=lags, ax=ax[1])
    # Modificamos los ticks para que vayan de cero al lag.
    ax[1].set_xticks(np.arange(0, lags+1))
    # Si se especificó una ruta para guardar, guardamos la imagen. Si la ruta no existe,
    # dará error.
    if save_to:
        plt.savefig(save_to)

    # Creamos un título de toda la figura. Usamos el nombre de la serie,
    # la separamos por los guiones bajos, la juntamos usando espacios y capitalizamos.
    fig.suptitle('Autocorrelación y autocorrelación parcial de \n{}'
                     .format(" ".join(series_name.split('_')).capitalize()),
                 fontsize=14)
    # mostramos gráfica.
    plt.show()

def cross_tab(df, cols, years, for_plot=False, notna=True, absolute_change=False, 
             perc_change=False, cumsum=False, style=True, titles=None):
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
    # Creamos diccionario que usaremos para que la tabla tenga meses en vez de numeros.
    meses = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
             7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre',
             12: 'Diciembre'}
    # Copiamos el df para no modificarlo.
    df2 = df.copy()
    # Eliminamos aquellas filas que son NA para todos
    df2 = df2[df2[cols].notna().any(1)]
    #  Creamos una nueva columna en el nuevo DF que se llamará fecha que es igual
    # al índice del DF, pues el índice del DF es fecha
    df2['fecha'] = df2.index
    # Creamos columna de anio usando la columna fecha. Esto lo hacemos aprovechando que
    # la columna fecha no es un string, es un objeto datetime. Los objetos datetime
    # son muy útiles. Si haces .year en un objeto datetime, obtienes el año. 
    # .map(lambda x: x.year) significa que para cada valor de la columna fecha, 
    # queremos obtener el año. lambda sirve para definir funciones pequeñas
    # que vas a usar en momentos especficos. 
    df2['anio'] = df2['fecha'].map(lambda x: x.year)
    # Hacemos lo mismo para el mes
    df2['mes'] = df2['fecha'].map(lambda x: x.month)
    # Creamos una columna que se llama mes_n, que es el mes pero en letra. Usamos
    # el diccionario que definimos anteriormente para ello.
    df2['mes_n'] = df2['mes'].map(meses)

    # Creamos una lista que usaremos para ordenar la tabla cuando la tengamos por meses.
    meses_orden = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio',\
               'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre', 'Total']
    if notna:
        meses_completos =\
            df2['mes_n'].unique()[df2.loc[df2['anio'].isin(years), 'mes_n'].value_counts() == len(years)]
        df2 = df2[df2['mes_n'].isin(meses_completos)]

    if for_plot:
        # Si el usuario especifica que la tabla la quiere solo para crear una gráfica,
        # entonces la tabla tendrå los meses como columnas y los años como índice.
        # La tambla tendrá poco formato. 
        # Esre produce un cuadro muy sencillo, solo para despues hacer grafica

        # Comprobamos que solo se haya pasado una columna, que es un requisito del
        # tipo de tabla for_plot
        assert len(cols) == 1, "If for plot, only give one col"
        # Nos quedamos únicamente con los años que estén en la lista de años que el
        # usuario pase.
        df2 = df2.loc[df2['anio'].isin(years)]
        # Utilizamos el metodo pivot, similar a las pivot_table de excel, que
        # organizan y agregan un DataFrame según el campo que se pase como
        # columna, el campo que se pase como eindice, y los valores. En este caso,
        # los valores son las columna que se pasó dentro de la lista de columnas.
        cross_tab = df2.pivot(
        index='anio', columns='mes_n', values=cols[0])
        # Vamos a usar la lista de meses en orden para obtener la lista de los meses
        # que estan en las columnas del cross tab en orden. Es decir, si solo
        # está Enero y Febrero, solo necesitamos decirle que orgene Enero y Febrero.
        # Si indicamos todos los meses, será error.
        col_order = [month for month in meses_orden if month in cross_tab.columns]
        # Reordenamos la tabla cross tab
        cross_tab = cross_tab[col_order]
        # Si el usuario especifica notna=True, solo nos quedaremos con los meses
        # que tengan valores para todos los años que se especifiquen. Es decir, si 
        # queremos hacer un cross tab para 2018 y 2019 y estamos en Junio de 2019, 
        # si el usuario especifica notna=True, solo veremos el cross tab de enero a
        # junio.
        if notna:
            cross_tab = cross_tab.loc[cross_tab.notna().all(1)]
        return cross_tab

    else:
        # Este produce un cuadro para analizar.

        # Creamos DF copia con unicamente los añoa que el usuario especifico.
        df2 = df2.loc[df2['anio'].isin(years)]
        cross_tab = df2.pivot_table(
            index='mes_n', columns='anio', values=cols)
        # Margins crea totales para filas y totales para columnas. No queremos el totales
        # para filas, lo eliminamos. Creamos una lista con todos los totales a eliminar.
        # Cross tab tendrá columnas estilo multiindice, pues entrá una columna por cada
        # variable que se especifique en cols, por cada año que se especifique en years.
        # Esto hace un poco mas complejo trabajar con el DF. Por eso, la lista de columnas
        # a eliminar será por ejemplo [('iva', 'Total'), ('isr', 'Total')]
        col_order = [month for month in meses_orden if month in cross_tab.index]
        cross_tab = cross_tab.loc[col_order]
        # Eliminar los NA's
        if notna:
            cross_tab = cross_tab.loc[cross_tab.notna().all(1)]
        months = list(cross_tab.index)
        if not titles:
            titles = cols
        iterables = [titles, years]
        # Creamos indice
        index = pd.MultiIndex.from_product(iterables)
        cross_tab.columns = index
        # añadimos cumsum
        if cumsum:
            cross_tab_cumsum = cross_tab.cumsum()
            iterables = [[title + ' acumulado' for title in titles],\
                         years]
            index = pd.MultiIndex.from_product(iterables)
            cross_tab_cumsum.columns = index
            cross_tab = cross_tab.merge(cross_tab_cumsum, left_index=True, right_index=True)

        if absolute_change:
            for title in titles:
                cross_tab[title, 'Diferencia absoluta'] = (cross_tab[title, years[-1]] - 
                                                           cross_tab[title, years[0]])
        if perc_change:
            for title in titles:
                cross_tab[title, 'Diferencia porcentual'] = ((cross_tab[title, years[-1]] - 
                                                              cross_tab[title, years[0]]) / 
                                                              cross_tab[title, years[0]]) * 100
        if cumsum:
            if absolute_change:
                for title in titles:
                    cross_tab[title + ' acumulado', 'Diferencia absoluta'] =\
                        (cross_tab[title + ' acumulado', years[-1]] -\
                            cross_tab[title + ' acumulado', years[0]])
            if perc_change:
                for title in titles:
                 cross_tab[title + ' acumulado', 'Diferencia porcentual'] =\
                    ((cross_tab[title + ' acumulado', years[-1]] -
                        cross_tab[title + ' acumulado', years[0]]) / 
                            cross_tab[title + ' acumulado', years[0]]) * 100
        order_of_columns = []
        for title in titles:
            order_of_columns.append(title)
            if cumsum:
                order_of_columns.append(title + ' acumulado')

        cross_tab = cross_tab[order_of_columns]

        # Finalmente, si los valores son menores a 100, queremos ver dos decimanes, de lo
        # contrario, queremos ver cero decimales. Creamos una función que formatea un float
        def formating(x):
            if x > 100:
                return '{:,.0f}'.format(x)
            else: 
                return '{:,.2f}'.format(x)
        # Aplicamos la funcion  de formato al cross tab y regresamos.
        if style:
            return cross_tab.applymap(formating).style
        else:
            return cross_tab

def cross_tab_lif(df, cols, notna=True, absolute_change=False, perc_change=False,
                  cumsum=False, style=True, title=None):
    '''
    Make cross tab to compare with LIF
    '''
    meses = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
             7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre',
             12: 'Diciembre'}
    # Copiamos el df para no modificarlo.
    df2 = df[cols].copy()
    # Eliminamos aquellas filas que son NA para todos
    df2 = df2[df2[cols].notna().any(1)]
    #  Creamos una nueva columna en el nuevo DF que se llamará fecha que es igual
    # al índice del DF, pues el índice del DF es fecha
    df2['fecha'] = df2.index
    # Creamos columna de mes usando la columna fecha. Esto lo hacemos aprovechando que
    # la columna fecha no es un string, es un objeto datetime. Los objetos datetime
    # son muy útiles. Si haces .month en un objeto datetime, obtienes el mes. 
    # .map(lambda x: x.month) significa que para cada valor de la columna fecha, 
    # queremos obtener el año. lambda sirve para definir funciones pequeñas
    # que vas a usar en momentos especficos. 
    df2['mes'] = df2['fecha'].map(lambda x: x.month)
    # Creamos una columna que se llama mes_n, que es el mes pero en letra. Usamos
    # el diccionario que definimos anteriormente para ello.
    df2['mes_n'] = df2['mes'].map(meses)
    # Eliminamos alguna fila que sea unicamente de valores NA
    df2 = df2[df2.notna().any(1)]
    # Si el usuario indica notna, nos quedamos solo con los meses que no son NA para todas las variables.
    if notna:
        df2 = df2[df2.notna().all(1)]
    # Establecemos el indice como el nombre de los meses
    df2.index = df2['mes_n']
    # Nos quedamos unicamoente con las columnas que indica el usuario
    df2 = df2[cols]
    # Si el usuario quiere ver valores acumulados, nos quedamos unicamente con ellos.
    # Si el usuario especifica un titulo, vamos a modificar los nombres de las columnas y las
    # convertiremos en un indice jerárquico, con supertitulo igual al titulo, y subtitulos
    # ioguales a 'Presupuestado', 'Observado', y 'Diferencia absoluta' y 'DIferencia relativa'
    # si es el caso.
    if title:
        iterables = [[title], ['Presupuestado', 'Observado']]
        # Creamos indice
        index = pd.MultiIndex.from_product(iterables)
        df2.columns = index
    if cumsum:
        df2_cumsum = df2.cumsum()
        iterables = [[title + ' acumulado'], ['Presupuestado', 'Observado']]
        index = pd.MultiIndex.from_product(iterables)
        df2_cumsum.columns = index
        df2 = df2.merge(df2_cumsum, left_index=True, right_index=True)

    # Creamos las columnas de diferencia absoluta y relativa si es necesario
    if absolute_change:
        df2[title, 'Diferencia absoluta'] = df2[title, 'Observado'] - df2[title, 'Presupuestado']
    if perc_change:
        df2[title, 'Diferencia porcentual'] = ((df2[title, 'Observado'] - df2[title, 'Presupuestado']) / 
                                                df2[title, 'Presupuestado']) * 100
    if cumsum:
        if absolute_change:
            df2[title + ' acumulado', 'Diferencia absoluta'] = (df2[title + ' acumulado', 'Observado'] - 
                                                 df2[title + ' acumulado', 'Presupuestado'])
        if perc_change:
             df2[title + ' acumulado', 'Diferencia porcentual'] = ((df2[title + ' acumulado', 'Observado'] - 
                                                    df2[title + ' acumulado', 'Presupuestado']) / 
                                                    df2[title + ' acumulado', 'Presupuestado']) * 100
        df2 = df2[[title, title + ' acumulado']]
    # Definimos funcion para dar formato
    def formating(x):
        if x > 100:
            return '{:,.0f}'.format(x)
        else: 
            return '{:,.2f}'.format(x)
    # Aplicamos la funcion  de formato al df y regresamos.
    if style:
        return df2.applymap(formating).style
    else:
        return df2




    # if not cumsum:
    #     cross_tab = df2.pivot_table(
    #         index='mes_n', columns=cols, margins=True,
    #         margins_name='Total', aggfunc='sum')
    #     return cross_tab


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


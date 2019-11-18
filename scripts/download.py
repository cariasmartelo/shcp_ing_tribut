'''
SHCP UPIT Forecasting Public Revenue
Download public revenue data, clean and load
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import os
import requests
import config
import zipfile
import io
#Functions ordered by importance in use
# FOlder donde guardaremos descargas.
downloads_folder = '../inputs/downloads/'


def get_files_inegi(inpc_2018=False, pibr_2013=False, pibr_2013_sa=False,
                    igae=False, igae_sa=False, igae_prim=False, igae_secun=False,
                    igae_terc=False, confianza_consumidor=False,
                    indic_mens_consumo=False, indic_adelant=False, pea=False,
                    pobl_ocupada=False, asegurados_imss=False,  imai=False,
                    imai_mineria=False, imai_construccion=False, imai_manufacturas=False,
                    imai_egergia_gas_agua_gas=False, emec_menor_total=False,
                    emec_menor_aba_ali_beb_tab=False, emec_menor_aba_ali=False,
                    emec_menor_hie_tab=True, emec_menor_antad=False,
                    emer_menor_text_vest_calz=False, emec_menor_pape_espar_otros=False,
                    emec_menor_domesticos=False, emec_menor_vehic=False, importaciones=False,
                    exportaciones=False):
    '''
    Run functions to download csv files.
    Inputs:
        Bools
    Output:
        Saves csv
    '''
    ############## INEGI

    # Si el usuario indica inpc_2018 = True, llamamos la función download_inegi con
    # 'inpc_2018' como argumento.
    if inpc_2018:
        download_inegi('inpc_2018')
    # Si el usuario indica pibr_2013 = True, llamamos la función download_inegi con
    # 'pibr_2013' como argumento.
    if pibr_2013:
        download_inegi('pibr_2013')
    # Si el usuario indica pibr_2013_Sa = True, llamamos la función download_inegi con
    # 'pibr_2013_sa' como argumento.
    if pibr_2013_sa:
        download_inegi('pibr_2013_sa')
    if igae:
        download_inegi('igae')
    if igae_sa:
        download_inegi('igae_sa')
    if igae_prim:
        download_inegi('igae_prim')
    if igae_secun:
        download_inegi('igae_secun')
    if igae_terc:
        download_inegi('igae_terc')
    if confianza_consumidor:
        download_inegi('confianza_consumidor')
    if indic_mens_consumo:
        download_inegi('indic_mens_consumo')
    if indic_adelant:
        download_inegi('indic_adelant')
    if pea:
        download_inegi('pea')
    if pobl_ocupada:
        download_inegi('pobl_ocupada')
    if asegurados_imss:
        download_inegi('asegurados_imss')
    if imai:
        download_inegi('imai')
    if imai_mineria:
        download_inegi('imai_mineria')
    if imai_egergia_gas_agua_gas:
        download_inegi('imai_egergia_gas_agua_gas')
    if imai_construccion:
        download_inegi('imai_construccion')
    if imai_manufacturas:
        download_inegi('imai_manufacturas')
    if emec_menor_total:
        download_inegi('emec_menor_total')
    if emec_menor_aba_ali_beb_tab:
        download_inegi('emec_menor_aba_ali_beb_tab')
    if emec_menor_aba_ali:
        download_inegi('emec_menor_aba_ali')
    if emec_menor_hie_tab:
        download_inegi('emec_menor_hie_tab')
    if emec_menor_antad:
        download_inegi('emec_menor_antad')
    if emer_menor_text_vest_calz:
        download_inegi('emer_menor_text_vest_calz')
    if emec_menor_pape_espar_otros:
        download_inegi('emec_menor_pape_espar_otros')
    if emec_menor_domesticos:
        download_inegi('emec_menor_domesticos')
    if emec_menor_vehic:
        download_inegi('emec_menor_vehic')
    if importaciones:
        download_inegi('importaciones')
    if exportaciones:
        download_inegi('exportaciones')

def get_files_datos_abiertos(fiscal_current=False, fiscal_hist=False):
    '''
    Run functions to download csv files.
    Inputs:
        Bools
    Output:
        Saves csv
    '''
    # Si el usuario indica fiscal_current = True, llamamos la función download_fiscal_data con
    # 'fiscal_current' como argumento.
    if fiscal_current:
        download_fiscal_data()
    # Si el usuario indica fiscal_hist = True, llamamos la función download_fiscal_data con
    # 'fiscal_hist' como argumento.
    if fiscal_hist:
        download_fiscal_data(current=False)

def get_files_banxico(tc_diario=False, tc_mensual=False, indice_tc_real=False,
                      tasa_cetes_28_diario=False, tasa_cetes_91_diario=False,
                      tasa_cetes_28_mensual=False, tasa_cetes_91_mensual=False):
    if tc_diario:
        download_banxico('tc_diario')
    if tc_mensual:
        download_banxico('tc_mensual')
    if indice_tc_real:
        download_banxico('indice_tc_real')
    if tasa_cetes_28_diario:
        download_banxico('tasa_cetes_28_diario')
    if tasa_cetes_91_diario:
        download_banxico('tasa_cetes_91_diario')
    if tasa_cetes_28_mensual:
        download_banxico('tasa_cetes_28_mensual')
    if tasa_cetes_91_mensual:
        download_banxico('tasa_cetes_91_mensual')


def get_files_fed(pibr_us_2012=False, pibr_us_2012_sa=False, ind_prod_ind_us_sa=False,
                  ind_prod_ind_us=False, tbill_3meses_mensual=False,
                  tbill_3meses_diario=False, cons_price_index_us=False,
                  cons_price_index_us_sa=False, trade_weighted_exchange_rate=False,
                  commodity_price_index=False):
    ############## FED
    if ind_prod_ind_us_sa:
        download_fed('ind_prod_ind_us_sa')
    if ind_prod_ind_us:
        download_fed('ind_prod_ind_us')
    if tbill_3meses_mensual:
        download_fed('tbill_3meses_mensual')
    if tbill_3meses_diario: 
        download_fed('tbill_3meses_diario')
    if cons_price_index_us:
        download_fed('cons_price_index_us')
    if cons_price_index_us_sa:
        download_fed('cons_price_index_us_sa')
    if pibr_us_2012:
        download_fed('pibr_us_2012')
    if pibr_us_2012_sa:
        download_fed('pibr_us_2012_sa')
    if trade_weighted_exchange_rate:
        download_fed('trade_weighted_exchange_rate')
    if commodity_price_index:
        download_fed('commodity_price_index')


def download_inegi(indicator, filepath=None):
    '''
    Download data from INEGI given an indicator, using the names from config.py
    Indicator:
        str
    Output:
        DF
    '''
    # EL path donde guardaremos la descarga es en una carpeta 'inputs' en la carpeta madre
    # de donde estamos.
    filepath = downloads_folder
    # Vamos a crear la carpeta si no existe. Usamos la libreria os de Python. Ver mas info:
    # https://docs.python.org/3/library/os.html
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    # El nombre del csv será el nombre del indicador.csv. Creamos la ruta del archivo que es
    # la carpeta que creamos + el nombre dle archivo usando la libreria os. 
    csv_path = os.path.join(filepath, indicator + '.csv')

    # Creamos la URL de descarga usando el archivo config. La url de INEGI se forma de cinco
    # elementos. Una URL inicial, despues una clave especifica de cada indicador, despues
    # otros ele entos que indican la fuente (hasta ahora la fuente siempre es el Banco de
    # información económica), después un token personal y después una terminación que indica
    # el tipo de archivo que queremos descargar. Para mas info: 
    # https://www.inegi.org.mx/servicios/api_indicadores.html
    url = config.INEGI['INEGI_URL'] + config.INEGI[indicator] + config.INEGI['INEGI_BIE']\
        + config.INEGI['INEGI_TOKEN'] + config.INEGI['INEGI_JSON']
    # Usamos la libreria requests para obtener el elemendo que se descarga del URL indicado.
    # Los objetos requests tienen varios atributos. Uno de ellos es status, que indica si se
    # pudo acceder al URL. Para obtener el json, hacemos response.json(). Para mas info:
    # https://2.python-requests.org/en/master/
    response = requests.get(url)
    # Creamos un DataFrame con el json que se descargo. El JSON entra a Python como un diccionario,
    # tenemos que filtrar el diccionario para obtener lo que está dentro de la clave 'Series', que
    # es una lista. De esa lista, obtenemos el primer elemento a hacer [0], que es un diccionario.
    # De ese diccionario obtenemos lo que está en la clave 'OBSERVATIONS'
    df = pd.DataFrame(response.json()['Series'][0]['OBSERVATIONS'])
    # Filtramos el DF para quedarnos unicamente con las columnas TIME_PERIOD y OBS_VALUE
    df = df[['TIME_PERIOD', 'OBS_VALUE']]
    # Renombramos TIME_PERIOD a fecha y OBS_VALUE al nombre del indicador.
    df.rename(columns={'TIME_PERIOD':'fecha', 
                       'OBS_VALUE': indicator},
                       inplace=True)
    # Pasamos el indicador a una variable numerica.
    df[indicator] = pd.to_numeric(df[indicator])
    # Exportamos el df a un csv, en el folder 
    df.to_csv(csv_path, index=False)
    # Obtenemos la decha de la última observación para imprimir una notificación.
    last_value = df['fecha'].max()
    # Imprimimos lo que descargamos, en donde lo descargamos y el valor de la última fecha.
    print('Downloaded {} in {}, last value: {}'.format(
        indicator, csv_path, last_value))

def download_banxico(indicator, filepath=None):
    '''
    Download data from BANXICO given an indicator, using the names from config.py
    Indicator:
        str
    Output:
        DF
    '''
    # EL path donde guardaremos la descarga es en una carpeta 'inputs' en la carpeta madre
    # de donde estamos.
    filepath = downloads_folder
    # Vamos a crear la carpeta si no existe. Usamos la libreria os de Python. Ver mas info:
    # https://docs.python.org/3/library/os.html
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    # El nombre del csv será el nombre del indicador.csv. Creamos la ruta del archivo que es
    # la carpeta que creamos + el nombre dle archivo usando la libreria os. 
    csv_path = os.path.join(filepath, indicator + '.csv')


    # Creamos la URL de descarga usando el archivo config. La url de BANXICO se forma de tres
    # elementos. Una URL inicial, despues una clave especifica de cada indicador, despues
    # un elemento que indica si se quiere el ultimo dato o toda la serie.
    # El token personal se envia como header.
    # https://www.banxico.org.mx/SieAPIRest/service/v1/
    url = config.BANXICO['BANXICO_URL'] + config.BANXICO[indicator] \
        + config.BANXICO['BANXICO_terminacion']
    # Creamos el objeto headers, en donde colocamos nuestro token y que tenemos que enviar
    # con el request.
    headers = {'Bmx-Token': config.BANXICO['BANXICO_token']}
    # Usamos la libreria requests para obtener el elemendo que se descarga del URL indicado.
    # Los objetos requests tienen varios atributos. Uno de ellos es status, que indica si se
    # pudo acceder al URL. Para obtener el json, hacemos response.json(). Para mas info:
    # https://2.python-requests.org/en/master/
    response = requests.get(url, headers=headers)
    # Creamos un DataFrame con el json que se descargo. El JSON entra a Python como un diccionario,
    # tenemos que filtrar el diccionario para obtener lo que está dentro de la clave 'Bmx',
    # dentro de ['series;, que es una lista. De esa lista, obtenemos el primer elemento a hacer [0],
    # que es un diccionario. De ese diccionario obtenemos lo que está en la clave 'datos'
    df = pd.DataFrame(response.json()['bmx']['series'][0]['datos'])
    # Renombramos dato por el nombre del indicador.
    df.rename(columns={'dato': indicator},  inplace=True)
    # Pasamos el indicador a una variable numerica. Indicamos que si un valor no se puede
    # convertir a numerico, que se asigne como NaN, con coerce.
    df[indicator] = pd.to_numeric(df[indicator], errors='coerce')
    # Modificamos la columna fecha a que sea datetime. Indicamos que el formato es '%d/%m/%Y'
    df['fecha'] = pd.to_datetime(df['fecha'], format = '%d/%m/%Y')
    # reordenamos y guardamos a csv
    df = df[['fecha', indicator]]
    # Guardamos a csv
    df.to_csv(csv_path, index=False)
    # Obtenemos la decha de la última observación para imprimir una notificación.
    last_value = df['fecha'].max()
    # Imprimimos lo que descargamos, en donde lo descargamos y el valor de la última fecha.
    print('Downloaded {} in {}, last value: {}'.format(
          indicator, csv_path, last_value))


def download_fed(indicator, filepath=None):
    '''
    Download data from Federal Reserve Bank of Saint Luois given an indicator, using the names
    from config.py
    Indicator:
        str
    Output:
        DF
    '''
    # EL path donde guardaremos la descarga es en una carpeta 'inputs' en la carpeta madre
    # de donde estamos.
    filepath = downloads_folder
    # Vamos a crear la carpeta si no existe. Usamos la libreria os de Python. Ver mas info:
    # https://docs.python.org/3/library/os.html
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    # El nombre del csv será el nombre del indicador.csv. Creamos la ruta del archivo que es
    # la carpeta que creamos + el nombre dle archivo usando la libreria os. 
    csv_path = os.path.join(filepath, indicator + '.csv')


    # Creamos la URL de descarga usando el archivo config. La url de FED se forma de cuatro
    # elementos. Una URL inicial,una clave especifica de cada indicador, un token personal,
    # y el formato de archivo que queremos. Hay otras opciones, como fecha iniial, agregacion,
    # etc que no vamos a usar.
    # El token, el id de la serie y el tipo de archivo se envia como params.
    # https://research.stlouisfed.org/docs/api/fred/
    url = config.FED['FED_URL'] 
    # Creamos el objeto params, en donde colocamos nuestro token y que tenemos que enviar
    # con el request.
    params = {'series_id': config.FED[indicator],
              'api_key': config.FED['FED_token'],
              'file_type': config.FED['FED_file_type']}
    # Usamos la libreria requests para obtener el elemendo que se descarga del URL indicado.
    # Los objetos requests tienen varios atributos. Uno de ellos es status, que indica si se
    # pudo acceder al URL. Para obtener el json, hacemos response.json(). Para mas info:
    # https://2.python-requests.org/en/master/
    response = requests.get(url, params=params)
    # Creamos un DataFrame con el json que se descargo. El JSON entra a Python como un diccionario,
    # tenemos que filtrar el diccionario para obtener lo que está dentro de la clave 'observations',
    # que es una lista de diccionarios. Transformamos esa lista de diccionarios a DF'
    df = pd.DataFrame(response.json()['observations'])
    df = df[['date', 'value']]
    # Renombramos dato por el nombre del indicador.
    df.rename(columns={'date': 'fecha', 'value': indicator},  inplace=True)
    # Pasamos el indicador a una variable numerica. Indicamos que si un valor no se puede
    # convertir a numerico, que se asigne como NaN, con coerce.
    df[indicator] = pd.to_numeric(df[indicator], errors='coerce')
    # Modificamos la columna fecha a que sea datetime. 
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Guardamos a csv
    df.to_csv(csv_path, index=False)
    # Obtenemos la decha de la última observación para imprimir una notificación.
    last_value = df['fecha'].max()
    # Imprimimos lo que descargamos, en donde lo descargamos y el valor de la última fecha.
    print('Downloaded {} in {}, last value: {}'.format(
          indicator, csv_path, last_value))

def download_fiscal_data(current=True):
    '''
    Downloads current fiscal data from datos abiertos and saves it to
    csv. It downloads data from 2011 to current if current. Otherwise,
    from 1990 to 2010. It also extracts the relevant data using the function
    get_taxes_from_csv_of_ingresos_fiscales_netos() and saves a clean csv in 
    the downloads folder.
    Inputs:
        current: str
    '''
    # Hacemos el mismo procedimiento que con INEGI para la carpeta con los datos.
    filepath = downloads_folder
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    # Seleccionamos la llave que usaremos para obtener el url del archivo config dependiendo
    # de si se quieren descargar datos actuales o históricos.
    if current:
        key = 'INGRESO_GASTO_SHCP_actual'
    else:
        key = 'INGRESO_GASTO_SHCP_hist'
    # Obtenemos el url del archivo config. Este es un url de datosabiertos que descarga un zipfile.
    url = config.INGRESOS_FISCALES[key]
    # Usamos la libreria requests para descargar el zipfile del url de datosabiertos.
    r = requests.get(url)
    # Obtenemos el archivo zip de la descarga que guardamos en el objeto r. Usamos la libreria
    # io. Ver mas info: https://docs.python.org/3/library/io.html
    z = zipfile.ZipFile(io.BytesIO(r.content))
    # Extraemos todo el contenido del archivo zip en la carpeta creada.
    z.extractall(filepath)
    # Imprimimos notificacion de descarga hecha.
    print('Downloaded {} in '.format(key) + downloads_folder)
    # Ahora importamos ambos csv, obtenemos los impuestos que buscamos, los vamos a concatenar y vamos a guardar un
    # únicamente con los impuestos que queremos. Tendremos tres csv: Dos de toda la base de estadísticas oportunas,
    # (uno historico y uno actual), y uno de los ingresos tributarios.
    fiscal_hist = get_taxes_from_csv_of_ingresos_fiscales_netos(
        downloads_folder + 'ingreso_gasto_finan_hist.csv')

    if current:
        # Creamos un DF con get_taxes_from_csv_of_ingresos_fiscales_netos datods actuales usando la funcion
        # load_discal_data
        fiscal_current = get_taxes_from_csv_of_ingresos_fiscales_netos(
            downloads_folder + 'ingreso_gasto_finan.csv')
        # Creamos un DF con todos los datos haciendo concat.
        fiscal_total = pd.concat([fiscal_hist, fiscal_current], sort=False)
        # Guardamos el DF a un csv
        fiscal_total.to_csv(downloads_folder + 'ingresos_tributarios_netos.csv', index=True)
        print('Saved ingresos_tributarios_netos.csv in '.format(downloads_folder))
        return fiscal_total

def get_taxes_from_csv_of_ingresos_fiscales_netos(fiscal_csv):
    '''
    Loads any of the two csv files
    inputs:
        fiscal_csv: str
    '''
    # Creamos un DF con el csv que descargamos de datos abiertos.
    df = pd.read_csv(fiscal_csv, encoding='latin-1')
    # usamos el archivo config para obtener las claves relevantes que necesitamos. Esto porque
    # de DatosAbiertos se descarga un archivo con TODOS los ingresos fiscales, que incluyen 
    # muchisimas variables. Solo seleccionaremos algunos.
    relevant_keys_d = config.INGRESOS_FISCALES['RELEVANT_KEYS_SHCP']
    # Creamos un diccionario que servirá para seleciconar las variables que queremos y para
    # renombrarlas.
    cols_to_keep_d = {'CICLO': 'year', 'MES': 'month', 'MONTO': 'monto',
                     'CLAVE_DE_CONCEPTO': 'clave_de_concepto', 'NOMBRE':'nombre'}
    # Nos quedamos solo con las observaciones cuya clave de concepto esté dentro de las claves
    # relevantes, y con las columnas indicadas en cols_to_keep_d
    df = df.loc[df['CLAVE_DE_CONCEPTO'].isin(relevant_keys_d), [k for k in cols_to_keep_d]]
    # renombramos las columnas usando el diccionario cols_to_keep_d
    df.rename(columns=cols_to_keep_d, inplace=True)
    # Creamos columna day
    df['day'] = 1
    # Creamos un diccionaro de meses-numero de mes para crear columna month
    months = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
              'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11,
              'Diciembre': 12}
    # Creamos columna month usando map y el diccionario. CUando uno hace .map y el diccionario,
    # los valores de la columna que se hace map se convierten al respectivo valor del diccionario.
    df['month'] = df['month'].map(months)
    # Creamos columna fecha tipo datetime
    df['fecha'] = pd.to_datetime(df[['year', 'month', 'day']])
    # Indicamos que la columna fecha sera el indice.
    df.set_index(df['fecha'], inplace=True)
    # Hacemos un pivot table para que los a ingreoss pasen a ser columnas y la columna monto
    # pase a ser lso valores.
    df = df.pivot(columns='clave_de_concepto', values='monto')
    # usamos el diccionario de config para renombrar las columnas.
    df.rename(columns=relevant_keys_d, inplace=True)
    # TOdo esta en miles de pesos. Dividimos para convertir a millones.
    df = df.div(1000)
    # Añadimos apellido a cada variables de _(mdp)
    df = df.add_suffix('_(mdp)')
    # eliminamos el nombre de las columnas (No el nombre de las variables). El nombre que Python
    # asigno al grupo de columnas.
    del df.columns.name
    # Retornamos DF

    return df

def load_ingresos_fiscales(excel_brutos='../inputs/ingresos_tributarios_desglosados.xlsx',
                           csv_netos=downloads_folder + 'ingresos_tributarios_netos.csv',
                           ajustes=True, ajustes_xlsx='../inputs/ajustes.xlsx'):
    '''
    Load ingresos netos e ingresos brutos y retornar un df conjunto
    '''
    # Cargamos ingresos brutos
    ingresos_brutos = load_ingresos_fiscales_brutos(excel_brutos)
    # Cargamos ingresos netos
    ingresos_netos = load_ingresos_fiscales_netos(csv_netos, ajustes, ajustes_xlsx)
    # De los ingresos brutos, vamos a quedarnos con las columnas que no son netos.
    # Para ello, vamos a hacer un loop un poco sobre las columnas de ingresos brutos,
    # quedandonos solo con las que no tenga neto en ellas.
    columns_brutos = ingresos_brutos.columns
    columns_to_merge = []
    for col in columns_brutos:
        if not 'neto' in col:
            columns_to_merge.append(col)
    ingresos_brutos = ingresos_brutos[columns_to_merge]
    ingresos_totales = pd.merge(ingresos_netos, ingresos_brutos, left_index=True,
                                right_index=True, how='outer')
    return ingresos_totales

def load_calendario_lif(excel_calendario_lif='../inputs/calendario_lif.xlsx'):
    '''
    Cargar calendario LIF en valores reales
    '''
    calendario_lif = pd.read_excel(excel_calendario_lif, index_col='fecha')
    calendario_lif.index = pd.to_datetime(calendario_lif.index)
    calendario_lif = calendario_lif.add_suffix('_presupuestado')
    calendario_lif['ingresos_no_tributatios_presupuestado'] = \
    calendario_lif[['derechos_presupuestado', 'aprovechamientos_presupuestado',\
                    'transferencias_fmp_presupuestado', 'otros no tributarios_presupuestado']].sum(axis=1)
    inpc = load_inpc()
    # Hacemos un merge entre el INPC y la base actual. Tanto el INPC como la base de ingresos
    # fiscales tiene indice de fecha, entonces el merge se hace con ledt_index y right_index
    calendario_lif = calendario_lif.merge(inpc, left_index=True, right_index=True)
    # Creamos un DF con todas las columnas en valores reales.
    calendario_lif_real = calendario_lif.div(calendario_lif['inpc'], axis=0) * 100
    # Eliminamos inpc del nuevo DF
    calendario_lif_real.drop(['inpc'], axis=1, inplace=True)
    # Añadimos _r al nombre de todas las variables del nuevo DF
    calendario_lif_real = calendario_lif_real.add_suffix('_r')
    # Concatenamos la base nominal y la base real. POdriamos hacer merge usando el indice, pero
    # sabemos que tienen las mismas dimesiones y que los valores estan alineados.
    calendario_lif = pd.concat([calendario_lif, calendario_lif_real], axis=1)

    return calendario_lif

def load_ingresos_fiscales_brutos(excel_file='../inputs/ingresos_tributarios_desglosados.xlsx'):
    '''
    Loads ingresos fiscales brutos from csv. Not historic data or data from datos abiertos,
    is data from SAT.
    Output:
        DF
    '''
    ## Leer CSV de ingresos tributarios desglosados en un DataFrame
    df = pd.read_excel(excel_file)
    # Crear una columna 'dia' con valor de 1. Esto para crear una fecha tipo datetime
    df['day'] = 1
    # Renombramos las columnas anio y mes
    df.rename(columns={'anio': 'year', 'mes': 'month'}, inplace=True)
    # Creamos columna fecha usando las columnas year, month y day. 
    df['fecha'] = pd.to_datetime(df[['year', 'month', 'day']])
    # Seleccionamos la columna fecha como el índice.
    df.index = df['fecha']
    # Eliminamos todas las columnas que no necesitamos. 
    df.drop(['year', 'mes_n', 'month', 'day', 'fecha', 'tiempo', 'rfc_u'], axis=1, inplace=True)
    # Renombramos iva_regu como iva_reg, para consistencia.
    df.rename(columns={'iva_regu': 'iva_reg'}, inplace=True)
    # Para cada uno de los impuestos, crearemos columna de suma de devoluciones y compensaciones
    # y de suma de devoluciones, compensaciones y regulaciones.
    for tax in ['iva', 'isr', 'ieps']:
        df[tax + '_dev_comp'] = df[tax + '_comp'] + df[tax + '_dev']
        df[tax + '_dev_comp_reg'] = df[tax + '_dev_comp'] + df[tax + '_reg']
    # Dividimos todo entre 1,000,000 para tener valores en MDP.
    df = df.div(1000000)
    # Añadimos _(mdp) al nombre de todas las variables
    df = df.add_suffix('_(mdp)')
    # Cargamos el INPC, lo usaremos para crear variables reales.
    inpc = load_inpc()
    # Hacemos un merge entre el INPC y la base actual. Tanto el INPC como la base de ingresos
    # fiscales tiene indice de fecha, entonces el merge se hace con ledt_index y right_index
    df = df.merge(inpc, left_index=True, right_index=True)
    # Creamos un DF con todas las columnas en valores reales.
    df_real = df.div(df['inpc'], axis=0) * 100
    # Eliminamos inpc del nuevo DF
    df_real.drop(['inpc'], axis=1, inplace=True)
    # Añadimos _r al nombre de todas las variables del nuevo DF
    df_real = df_real.add_suffix('_r')
    # Concatenamos la base nominal y la base real. POdriamos hacer merge usando el indice, pero
    # sabemos que tienen las mismas dimesiones y que los valores estan alineados.
    df = pd.concat([df, df_real], axis=1)
    # Creamos columnas por impuesto del % del bruto. Hacemos loop para cada impuesto y para
    # cada sub concepto. 
    for tax in ['iva', 'isr', 'ieps']:
        for expense in ['comp', 'dev', 'reg', 'neto', 'dev_comp', 'dev_comp_reg']:
            df['_'.join([tax, expense, r'%bruto'])] = (
                df['_'.join([tax, expense, '(mdp)', 'r'])]
                    / df['_'.join([tax, 'bruto',  '(mdp)', 'r'])]) * 100
    # Indicamos a Pandas que la frecuencia de los datos es mensual.
    df = df.asfreq(freq='MS')

    # Retornamos el DF
    return df


def load_ingresos_fiscales_netos(csv_file=downloads_folder + 'ingresos_tributarios_netos.csv',
                                 ajustes=True, ajustes_xlsx='../inputs/ajustes.xlsx'):
    '''
    Loads both historic and current fiscal data. Returns DF
    Output:
        DF
    '''
    # Cargamos el csv de ingresos fiscales que guardamos al descargar los datos fiscales
    fiscal_total = pd.read_csv(csv_file)
    # Convertimos la columna fecha a datetime
    fiscal_total['fecha'] = pd.to_datetime(fiscal_total['fecha'])
    # Seleccionamos la columna fecha como indice
    fiscal_total.index = fiscal_total['fecha']
    # Eliminamos la columna fecha ya que es indice
    fiscal_total.drop('fecha', axis=1, inplace=True)
    # renombramos
    rename = {
        'ingresos_gobierno_federal_neto_(mdp)': 'ing_gob_fed_neto_(mdp)',
        'ingresos_tributarios_neto_(mdp)': 'ing_trib_neto_(mdp)',
        'ingresos_no_tributarios_neto_(mdp)': 'ing_no_trib_neto_(mdp)',
     }
    fiscal_total.rename(columns=rename, inplace=True)
    # Si el usuario indica ajustes, importamos la hoja de excel con los ajustes
    if ajustes:
        # Leemos el archivo de excel
        ajustes = ajustes = pd.read_excel(ajustes_xlsx, sheet_name='ajustes')
        # Convertimos la columna fecha a datetime
        ajustes['fecha'] = pd.to_datetime(ajustes['fecha'])
        # Establecemos la columna fecha como el índice
        ajustes.index = ajustes['fecha']
        # Eliminamos la columna fecha ya que es indice
        ajustes.drop('fecha', axis=1, inplace=True)
        # Nos quedamos unicamente con las observaciones que están también en fiscal total. Es decir, 
        # con las mismas fechas.
        ajustes = ajustes.loc[fiscal_total.index]
        # Obtengo lista de columnas que vamso a ajsutar agregando '_(mdp)' al nombre de las columnas
        # en ajustes
        cols_to_ajust = [col + '_(mdp)' for col in ajustes.columns]
        # Cambio el nombre de las columnas de ajustes por los nombres com '_(mdp)'
        ajustes.columns = cols_to_ajust
        # Hacemos la corrección. Sumo ambos dataframes (Las columnas que concuerdan)
        fiscal_total[cols_to_ajust] = fiscal_total[cols_to_ajust] + ajustes

    # Creamos ingresoso tributarios sin gasolina
    fiscal_total['ing_trib_sin_gasol_neto_(mdp)'] = \
        fiscal_total['ing_trib_neto_(mdp)'] - fiscal_total['ieps_gasolina_neto_(mdp)']
    # Cargamos el inpc
    inpc = load_inpc()
    # Hacemos merge del DF total con el inpc. Usamos left y right index=True porque
    # ambos tienen fecha como índice.
    fiscal_total = fiscal_total.merge(inpc, left_index=True, right_index=True)

    # Creamos un DF con los valores reales dividiendo todo el DF de ingresos fiscales.
    fiscal_real = fiscal_total.div(fiscal_total['inpc'], axis=0) * 100
    # Eliminamos la columna inpc del DF on valores reales.
    fiscal_real.drop('inpc', axis=1, inplace=True)
    # Añadimos _r a todas las columnas del DF real 
    fiscal_real = fiscal_real.add_suffix('_r')
    # Concatenamos el DF nominal y el DF real. axis=1 indica que queremos concatenar
    # columnas, no filas.
    fiscal_total = pd.concat([fiscal_total, fiscal_real], axis=1)

    # Indicamos a Pandas que la frequencia es mensual.
    fiscal_total = fiscal_total.asfreq(freq='MS')
    # retornamos el DF.
    return fiscal_total

def load_inegi_indic(varname, csv_file=None):
    '''
    Esta función sirve apra cargar uno de los muchos indicadores del INEGI a un DataFrame.
    Se debe pasar como imput el nombre del indicador, establecido en config.py en el diccionario
    INEGI/
    Inputs:
        varname: string
        csv_file: str
    Output:
        DF
    '''
    if not csv_file:
        csv_file =downloads_folder + varname + '.csv'
    # Cargamos el csv a un DF
    df = pd.read_csv(csv_file)
    # Transformamos la columna fecha a un objeto datetime. Hasta este momento, la columna fecha
    # era un string tipo '2019-01-01'. Pandas lee eso y puede convertirlo a un objeto fecha.
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Indicamos que el indice va a ser la fecha.
    df.set_index(df['fecha'], inplace=True)
    # Eliminamos la columna fecha. Pues solo la queremos en el indice.
    df.drop('fecha', axis=1, inplace=True)
    # Indicamos que los valores son mensuales
    df = df.asfreq(freq='MS')
    # Retornamos el df
    return df

def load_banxico(varname, monthly=True, csv_file=None):
    '''
    Esta función sirve apra cargar uno de los muchos indicadores del INEGI a un DataFrame.
    Se debe pasar como imput el nombre del indicador, establecido en config.py en el diccionario
    INEGI/
    Inputs:
        varname: string
        csv_file: str
    Output:
        DF
    '''
    if not csv_file:
        csv_file =downloads_folder + varname + '.csv'
    # Cargamos el csv a un DF
    df = pd.read_csv(csv_file)
    # Transformamos la columna fecha a un objeto datetime. Hasta este momento, la columna fecha
    # era un string tipo '2019-01-01'. Pandas lee eso y puede convertirlo a un objeto fecha.
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Indicamos que el indice va a ser la fecha.
    df.set_index(df['fecha'], inplace=True)
    # Eliminamos la columna fecha. Pues solo la queremos en el indice.
    df.drop('fecha', axis=1, inplace=True)
    # Si el usuario pasa el argumento monthly=True, todos los df de banxico se convierten a
    # mensuales. Si los datos ya son mensuales, no pasa nada. SI los datos son diariosn, 
    # se convierten a mensual usando la media.
    if monthly:
        df = df.resample('MS').mean()
        df = df.asfreq(freq='MS')
    # Retornamos el df
    return df


def load_fed(varname, monthly=True, csv_file=None):
    '''
    Esta función sirve apra cargar uno de los muchos indicadores del INEGI a un DataFrame.
    Se debe pasar como imput el nombre del indicador, establecido en config.py en el diccionario
    INEGI/
    Inputs:
        varname: string
        csv_file: str
    Output:
        DF
    '''
    if not csv_file:
        csv_file = downloads_folder + varname + '.csv'
    # Cargamos el csv a un DF
    df = pd.read_csv(csv_file)
    # Transformamos la columna fecha a un objeto datetime. Hasta este momento, la columna fecha
    # era un string tipo '2019-01-01'. Pandas lee eso y puede convertirlo a un objeto fecha.
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Indicamos que el indice va a ser la fecha.
    df.set_index(df['fecha'], inplace=True)
    # Eliminamos la columna fecha. Pues solo la queremos en el indice.
    df.drop('fecha', axis=1, inplace=True)
    # Si el usuario pasa el argumento monthly=True, todos los df de banxico se convierten a
    # mensuales. Si los datos ya son mensuales, no pasa nada. SI los datos son diariosn, 
    # se convierten a mensual usando la media.
    if monthly:
        if varname.startswith('pibr_us'):
            date_range = pd.date_range(
                df.index.min(), df.index.max() + relativedelta(months = 3), freq='QS')
            df = df.reindex(date_range)
            df = df.resample('MS').pad()
        else:
            df = df.resample('MS').mean()
        df = df.asfreq(freq='MS')
    # Retornamos el df
    return df


def load_inpc(csv_file = downloads_folder + 'inpc_2018.csv'):
    '''
    Load INPC as DF.
    Inputs:
        csv_file: str
    Output:
        DF
    '''
    # Cargamos el csv a un DF
    df = pd.read_csv(csv_file)
    # Creamos columna fecha tipo datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Creamos columna de año.
    df['year'] = df['fecha'].map(lambda x: x.year)
    # obtenemos ultimo año
    last_year = df['year'].max()
    # Obtenemos inpc promeido ultimo año
    avg_inpc_last_year = df.loc[df['year'] == last_year, 'inpc_2018'].mean()
    # Seleccionamos la columna fecha como indice
    df.set_index(df['fecha'], inplace=True)
    # Eliminamos columna fecha y year
    df.drop(['fecha', 'year'], axis=1, inplace=True)
    # Convirtiendo el INPC a la base del último dato. Esot lo hacemos dividiendo
    # todos los valores entre el valor promedio del INPC del ultimo año
    # avg_inpc_last_year = df.loc[df.index.max(), 'inpc_2018']

    df = df.div(avg_inpc_last_year) * 100
    # Renombramos 'inpc_2018' a 'inpc'
    df = df.rename(columns={'inpc_2018': 'inpc'})
    # Retornamos el DF.
    return df

def load_pib_r(csv_file=None, sa=False, monthly=False):
    '''
    Load PIB_r as DF. It is in (MDP). If sa, loads seasonally adjusted.
    Inputs:
        csv_file: str
    Output:
        DF
    '''
    # SImilar a con el igae, depende si se llama la función con sa=True o no, vamos a
    # construir la ruta del csv que vamos a cargar.
    if not csv_file:
        if not sa:
            varname = 'pibr_2013'
        else:
            varname = 'pibr_2013_sa'

        csv_file = downloads_folder + varname + '.csv'

    # Cargamos el csv a un DataFrame
    df = pd.read_csv(csv_file)
    # Convertimos la columna fecha a datetime. OJO: Fecha es formato 'Y/Q'
    # Por ejemplo '2015-02-01' indica el segundo trimestre del 2015. 
    # Tenemos que convertir esto a una fecha '2015-04-01', que marque correctamente
    # los meses.
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Creamos una columna year a partir de la fecha. .map indica que vamos a hacer una operacion
    # en cada uno de los valores de la columna. lambda x: x.year indica que de cada elemento
    # queremos obtener el attributo year (x.year).
    # ver: https://www.w3schools.com/python/python_lambda.asp para funciones lambda
    # ver https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.map.html
    # para .map.
    df['year'] = df['fecha'].map(lambda x: x.year)
    # Creamos una columna month igual a year. Modificamos para que refleje el mes real.
    # Para pasar de quarter a mes hay que multiplicar (3*(Q-1) + 1)
    df['month'] = df['fecha'].map(lambda x: 3*(x.month -1) + 1)
    # Cremos columna day, la necesitamos para volver a construir columna fecha de forma correcta.
    df['day'] = 1
    # Remplazamos columna fecha creando un objeto datetime con los valores de year, month  y day
    # que creamos.
    df['fecha'] = pd.to_datetime(df[['year', 'month', 'day']])
    # Seleccionamos la columna fecha como indice.
    df.set_index(df['fecha'], inplace=True)
    # Nos quedamos unicamente con pibr_2013 o pibr_2013_sa
    df = df[varname]
    # Indicamos que la frequencia es mensual.
    df = df.asfreq(freq='QS')
    # cargamos el inpc
    inpc = load_inpc()
    # Obtenemos valor promedio del inpc en 2013.
    inpc_2013_mean = inpc.loc[pd.date_range('2013-01-01', '2013-12-01', freq='MS')].mean()
    # Lo convertimos a float (Ahorita es Pandas Series de un solo valor)
    inpc_2013_mean = float(inpc_2013_mean)
    # La nueva variable será pibr_2019 o pibr_2019_sa, usamos el nombre de la variable
    # actual para obtenerlo
    varname_2019 = varname.replace('13', '19')
    # Convertimos la serie a DF
    df = df.to_frame()
    # Creamos la nueva variable dividiendo el valor 2013 entre el promedio del inpc 2013.
    df[varname_2019] = (df[varname] / inpc_2013_mean) * 100
    # Si el usuario no quiere valores mensuales, ya está listo el DF
    if not monthly:
        return df
    # De lo contrario, insertamos un valor exra NA para un extra quarter. Esto va a servir para poblar
    # los valores mensuales. relativedelta indica que queremos un valor para un indice
    # 3 meses después del úitimo índice. Cuando poblemos los valores mensuales,
    # Pandas va a poblar los meses que esten entre el minimo y el maximo mes.
    # Como queremos todos los meses del último quarter, necesitamos añadir un trimestre más.
    new_quarter = df.index.max() + relativedelta(months = 3)
    # Asignamos el valor de ese nuevo indice como  Nan
    df.loc[new_quarter] = np.NaN
    # Resample DF into months, using forward fill. .pad() es equivalente a ffill. ffill significa
    # que usaremos el valor no NA anterior para poblar los nuevos valores. Así, el pib de enero
    # se replicará a febrero y a marzo.
    df = (df.resample('MS').pad())
    # Indicamos que la frequencia es mensual.
    df = df.asfreq(freq='MS')
    # Retornamos el DF
    return df

def load_balanza_comercial(varname, csv_file=None):
    '''
    Load importaciones or exportaciones. Tenemos que convertir a pesos, por eso la funcion especial
    '''
    if not csv_file:
        csv_file =downloads_folder + varname + '.csv'
    # Cargamos el csv a un DF
    df = pd.read_csv(csv_file)
    # Transformamos la columna fecha a un objeto datetime. Hasta este momento, la columna fecha
    # era un string tipo '2019-01-01'. Pandas lee eso y puede convertirlo a un objeto fecha.
    df['fecha'] = pd.to_datetime(df['fecha'])
    # Indicamos que el indice va a ser la fecha.
    df.set_index(df['fecha'], inplace=True)
    # Eliminamos la columna fecha. Pues solo la queremos en el indice.
    df.drop('fecha', axis=1, inplace=True)
    # Cargamos TC
    tc = load_banxico('tc_mensual')
    # Hacemos merge con TC
    df = df.merge(tc, left_index=True, right_index=True)
    # Convertimos a pesos
    df[varname] = df[varname] * df['tc_mensual']
    # Adios TC
    df.drop('tc_mensual', axis=1, inplace=True)
    # Hola INPC
    inpc = load_inpc()
    df = df.merge(inpc, left_index=True, right_index=True)
    # Creamos un DF con todas las columnas en valores reales.
    df_real = df.div(df['inpc'], axis=0) * 100
    # Eliminamos inpc del nuevo DF
    df_real.drop('inpc', axis=1, inplace=True)
    # Añadimos _r al nombre de todas las variables del nuevo DF
    df_real = df_real.add_suffix('_r')
    # Concatenamos la base nominal y la base real. POdriamos hacer merge usando el indice, pero
    # sabemos que tienen las mismas dimesiones y que los valores estan alineados.
    df = pd.concat([df, df_real], axis=1)
    # Eliminamos inpc
    df.drop('inpc', axis=1, inplace=True)
    # Retornamos el df
    return df

def extract_from_cuadro_preliminar(excel_file):
    '''
    Extract data from excel file of cuadro contribuciones.
    Inputs:
        excel_file: str
    Output:
        dict
    '''
    index_dict = {
    'total de contribuciones': 'ingresos_gobierno_federal',
    'suma tributarios': 'ingresos_tributarios',
    'impuesto sobre la renta':'isr',
    'impuesto al valor agregado': 'iva',
    'impuesto especial sobre producción y servicios': 'ieps',
    'impuesto sobre automóviles nuevos': 'isan',
    'impuesto al comercio exterior': 'importaciones',
    'accesorios': 'accesorios_contribuciones', 
    'impuesto extracción de hidrocarburos': 'impuesto_extraccion_hidrocarburos',
    'suma no tributarios': 'ingresos_no_tributarios'}

    # Creamos diccionario vacio donde insertaremos valores nuevos
    new_vals = pd.DataFrame(index=range(0, 17))
    # Vamos a obtener valores. Importamos la hoja 'Contribuciones Federales' del excel
    # 'Cuadros carpeta preliminar mes'
    sheet_names = {'brutos': 'Contribuciones_Federales Bruta',
                   'netos': 'Contribuciones_Federales'}
    for income_type, sheet in sheet_names.items():
        contribuciones = pd.read_excel(
            excel_file, sheet_name=sheet, header=4)
        # Nos quedamos solo con las columnas de conceptos (importada como Unnamed 0) y con la que tiene los valores
        # titulada 2019
        contribuciones = contribuciones[['Unnamed: 0', 2019]]
        # Renombramos las columnas para que se llamen concept y el tipo
        contribuciones.columns = ['concept', income_type]
        # Eliminamos las filas de valores nulos
        contribuciones = contribuciones.loc[contribuciones.notna().all(1)]
        # MOdificamos el nombre de los conceptos, quitando espacios y convirtiendo a minusculas
        contribuciones['concept'] = contribuciones['concept'].map(lambda x: x.strip().lower())
        # Nos quedamos solo con los valores relevantes
        contribuciones = contribuciones.iloc[0:17]
        # Indicamos el indice
        contribuciones.index = contribuciones['concept']
        # Eliminamos columna concept
        contribuciones.drop('concept', axis=1, inplace=True)
        # Agregamos la serie al diccionario, primero aggregando el indice
        new_vals.index = contribuciones.index
        # luego agregabdo la columna
        new_vals[income_type] = contribuciones
    # Ahora añadimos las devoluciones y compensaciones
    dev_comp = pd.read_excel(excel_file, sheet_name='DCR´s', header=4)
    dev_comp = dev_comp.iloc[:, 0:7]
    # Nos quedamos con las primeras 7 columnas
    dev_comp = dev_comp.iloc[:, 0:7]
    # renombramos las columnas
    dev_comp.columns = ['concept', 'dev_t-1', 'dev', 'comp_t-1', 'comp', 'regu_t-1', 'regu']
    # Nos quedamos con las columnas de concepto y de los gastos de este año
    dev_comp = dev_comp.loc[dev_comp.notna().all(1), ['concept', 'dev', 'comp', 'regu']]
    # MOdificamos el nombre de los conceptos, quitando espacios y convirtiendo a minusculas
    dev_comp['concept'] = dev_comp['concept'].map(lambda x: x.strip().lower())
    # Indicamos el índice
    dev_comp.index = dev_comp['concept']
    # Eliminamos columna concept
    dev_comp.drop('concept', axis=1, inplace=True)
    # Hacemos merge
    new_vals = new_vals.merge(dev_comp, left_index=True, right_index=True)
    # modificamos indice
    new_vals.index = new_vals.index.map(index_dict)
    # Nos quedamos con los que queremos
    new_vals = new_vals.loc[new_vals.index.notna()]

    return new_vals


def extract_from_cuadro_isr_iva_ieps(excel_file):
    '''
    Extract data from excel file of cuadro contribuciones.
    Inputs:
        excel_file: str
    Output:
        dict
    '''
    index_map = {
    'total': 'ieps',
    'bebidas alcoholicas': 'ieps_ba',
    'cervezas y bebidas refrescantes': 'ieps_cerveza',
    'tabacos labrados':  'ieps_tabaco',
    'juegos con apuestas y sorteos': 'ieps_jys',
    'redes públicas de telecomunicaciones': 'ieps_telecom',
    'bebidas energetizantes': 'ieps_be',
    'bebidas saborizadas': 'ieps_bsaborizadas',
    'alimentos no básicos con alta densidad calórica': 'ieps_aadc',
    'plaguicidas': 'ieps_plaguicidas',
    'combustibles fósiles': 'ieps_combustibles_fosiles',
    'otros petroliferos': 'ieps_otros_petroliferos',
    'hidrocarburos': 'ieps_hidrocarburos',
    'retenciones por terceros': 'ieps_otras_retenciones'}

    ieps = pd.read_excel(excel_file, sheet_name='IEPS', header=4)
    ieps.columns = ['concept', 'ieps_t-1', 'ieps_t', 'participacion',\
                    'variacion_abs' , 'variacion_perc']
    ieps = ieps.loc[ieps.notna().all(1), ['concept', 'ieps_t']]
    ieps = ieps.iloc[:14]
    ieps['concept'] = ieps['concept'].map(lambda x: x.strip().lower())
    ieps.index = ieps['concept']
    # Eliminamos columna concept
    ieps.drop('concept', axis=1, inplace=True)
    # Cambiamos nombre de indice
    ieps.index = ieps.index.map(index_map)

    return ieps

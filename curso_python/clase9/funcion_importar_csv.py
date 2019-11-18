import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from cycler import cycler


def importar_ingresos(ingresos_csv='ingresos_tributarios_netos.csv',
                      inpc_csv='inpc_2018.csv'):
    '''
    Funcion para importar ingresos fiscales
    Inputs:
        csv_file: String
    Output:
        DataFrame
    '''
    ingresos = pd.read_csv(ingresos_csv)
    ingresos['fecha'] = pd.to_datetime(ingresos['fecha'])
    ingresos.index = ingresos['fecha']
    ingresos.drop('fecha', axis=1, inplace=True)

    inpc = pd.read_csv(inpc_csv)
    inpc['fecha'] = pd.to_datetime(inpc['fecha'])
    inpc.index = inpc['fecha']
    inpc.drop('fecha', axis=1, inplace=True)

    ingresos = pd.merge(ingresos, inpc, left_index=True,
                        right_index=True)
    ingresos_reales = ingresos.div(ingresos['inpc_2018'], axis=0) * 100
    ingresos_reales.drop('inpc_2018', axis=1, inplace=True)
    ingresos_reales = ingresos_reales.add_suffix('_r')

    ingresos = pd.concat([ingresos, ingresos_reales], axis=1)

    return ingresos



def descargar_inegi(indicador, token):
    '''
    Funcion para descargar datos de INEGI
    Inputs:
        indicador: string
    Output:
        DF y csv
    '''
    inegi_dict = {
        'pib': '493621',
        'igae': '496150',
        'exportaciones': '127598'
    }
    clave = inegi_dict[indicador]
    url = ('https://www.inegi.org.mx/app/api/indicadores/'
            'desarrolladores/jsonxml/INDICATOR/{}/es/0700/'
            'false/BIE/2.0/{}'
            'fe?type=json'.format(clave, token))
    r = requests.get(url)
    df = pd.DataFrame(r.json()['Series'][0]['OBSERVATIONS'])
    df = df[['OBS_VALUE', 'TIME_PERIOD']]
    df.rename(columns={'OBS_VALUE': indicador,
                        'TIME_PERIOD': 'fecha'},
                inplace=True)
    df.to_csv('{}.csv'.format(indicador), index=False)
    print('Guardamos en CSV')
    df['fecha'] = pd.to_datetime(df['fecha'])
    df.index = df['fecha']
    df.drop('fecha', axis=1, inplace=True)
    df[indicador] = pd.to_numeric(df[indicador])
    return df


def descargar_banxico(indicador, token):
    '''
    Funcion para descargar datos de Banxico
    Inputs:
        indicador: string
        token: string
    Output:
        Pandas DF
        CSV
    '''
    banxico_dict = {
        'tc_diario': 'SF43718',
        'cetes_28': 'SF43936',
        't_bill_3_meses': 'SI563'
    }
    clave = banxico_dict[indicador]
    url = ('https://www.banxico.org.mx/SieAPIRest/service/v1/'
           'series/{}/datos'.format(clave))
    header = {'Bmx-Token': token}
    r = requests.get(url, headers=header)
    df = pd.DataFrame(r.json()['bmx']['series'][0]['datos'])

    df.rename(columns={'dato': indicador}, inplace=True)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df.index = df['fecha']
    df.drop('fecha', axis=1, inplace=True)
    df[indicador] = pd.to_numeric(df[indicador], errors='coerce')
    return df


def plot_ingresos_tributarios(df, columnas=None, titulo=None, grid=True, 
                              legenda=None, calidad=100,
                              tamaño=(12, 6), fecha_minima=None):
    '''
    Graficas ingresos tributarios
    '''
    custom_cycler = (cycler(color=['#333f50', '#691A30', '#7f7f7f',
                                   'xkcd:khaki', 'darkgreen', 'darkblue',
                                   'crimson', 'gold']))
    if not columnas:
        columnas = df.columns
    if not legenda:
        legenda = columnas
    if fecha_minima:
        df = df.loc[fecha_minima:]
    fig, ax = plt.subplots(figsize=tamaño, dpi=calidad)
    ax.set_prop_cycle(custom_cycler) 
    ax.plot(df[columnas])
    ax.legend=legenda
    if grid:
        ax.grid()
    if titulo:
        ax.set_title(titulo)

    plt.show()












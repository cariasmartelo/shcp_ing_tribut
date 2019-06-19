'''
SHCP UPIT Forecasting Public Revenue
Download public revenue data, clean and load
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import requests
import config
import zipfile
import io

#Functions ordered by importance in use
def load_fiscal_income():
    '''
    Loads both historic and current fiscal data. Returns DF
    Output:
        DF
    '''
    fiscal_hist = load_fiscal_data('../inputs/ingreso_gasto_finan_hist.csv')
    fiscal_current = load_fiscal_data('../inputs/ingreso_gasto_finan.csv')
    fiscal_total = pd.concat([fiscal_hist, fiscal_current])
    inpc = load_inpc()
    fiscal_total = fiscal_total.merge(inpc, left_index=True, right_index=True)
    fiscal_real = fiscal_total.div(fiscal_total['inpc_2018'], axis=0) * 100
    fiscal_real.drop('inpc_2018', axis=1, inplace=True)
    fiscal_real = fiscal_real.add_suffix('_r')
    fiscal_total = pd.concat([fiscal_total, fiscal_real], axis=1)

    return fiscal_total

def get_files(inpc=False, fiscal_current=False, fiscal_hist=False):
    '''
    Run functions to download csv files. If update only, it does not
    download historic fiscal data.
    Inputs:
        update:only:Bool
    Output:
        Saves csv
    '''
    if inpc:
        download_inpc()
    if fiscal_current:
        download_fiscal_data()
    if fiscal_hist:
        download_fiscal_data(current=False)

def download_inpc(csv_file='../inputs/inpc.csv'):
    '''
    Download INPC from INEGI.
    Inputs:
        csv_file: str
    Output:
        Saves csv
    '''
    url0, url1, url2 = config.INEGI['INEGI_URL']
    url = url0 + config.INEGI['INEGI_INPC'] + url1 + \
          config.INEGI['INEGI_TOKEN'] + url2
    response = requests.get(url)
    df = pd.DataFrame(response.json()['Series'][0]['OBSERVATIONS'])
    df = df[['TIME_PERIOD', 'OBS_VALUE']]
    df.rename(columns={'TIME_PERIOD':'fecha', 
                       'OBS_VALUE': 'inpc_2018'},
                       inplace=True)
    df['inpc_2018'] = pd.to_numeric(df['inpc_2018'])
    df.to_csv(csv_file, index=False)
    last_value = df['fecha'].max()
    print('Downloaded INPC in {}, last value: {}'.format(csv_file, last_value))

def download_fiscal_data(current=True):
    '''
    Downloads current fiscal data from datos abiertos and saves it to
    csv. It downloads data from 2011 to current.
    Inputs:
        current: str
    '''
    if current:
        key = 'INGRESO_GASTO_SHCP_actual'
    else:
        key = 'INGRESO_GASTO_SHCP_hist'
    url = config.INGRESOS_FISCALES[key]
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall('../inputs/')
    print('Downloaded {} in ../inputs/'.format(key))

def load_inpc(csv_file ='../inputs/inpc.csv'):
    '''
    Load INPC as DF.
    Inputs:
        csv_file: str
    Output:
        DF
    '''
    df = pd.read_csv(csv_file)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df.set_index(df['fecha'], inplace=True)
    df.drop('fecha', axis=1, inplace=True)
    df = df.loc['1990-01-01':]

    return df

def load_fiscal_data(fiscal_csv):
    '''
    Loads any of the two csv files
    inputs:
        fiscal_csv: str
    '''
    df = pd.read_csv(fiscal_csv, encoding='latin-1')
    relevant_keys_d = config.INGRESOS_FISCALES['RELEVANT_KEYS_SHCP']
    cols_to_keep_d = {'CICLO': 'year', 'MES': 'month', 'MONTO': 'monto',
                     'CLAVE_DE_CONCEPTO': 'clave_de_concepto', 'NOMBRE':'nombre'}
    df = df.loc[df['CLAVE_DE_CONCEPTO'].isin(relevant_keys_d), [k for k in cols_to_keep_d]]
    df.rename(columns=cols_to_keep_d, inplace=True)
    df['day'] = 1
    months = {'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
              'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11,
              'Diciembre': 12}
    df['month'] = df['month'].map(months)
    df['fecha'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.set_index(df['fecha'], inplace=True)
    df = df.pivot(columns='clave_de_concepto', values='monto')
    df.rename(columns=relevant_keys_d, inplace=True)
    df = df.div(1000000)
    df = df.add_suffix('_(mmdp)')
    del df.columns.name

    return df

def load_revenue_data(file, columns=None, convert_to_csv=False):
    #DEPRECATED BECUSE FOUND FULL DATA DOWNLOADABLE
    '''
    Load BDD ingresos Delgado Formato.xlx and create some columns.
    Data with Public Revenue Data from 1990 to 2019. If convert_to_csv
    the data is loaded and saved as a csvm, this to convert for first time
    the excel file to csv.
    Inputs:
        file: string
        columns = [string]
    Output:
        Pand
    '''
    if not columns:
        columns = ['ingresos_tributarios_neto', 'fecha']

    # If convert to csv is specified, then the function assumes that is being
    # passed the xlsx, and saves it to a csv.
    if convert_to_csv:
        df = pd.read_excel(file, skiprows=[0])
        file = '../inputs/ingresos_fiscales_historicos.csv'
        df.rename(columns={
            ' Fecha(Mensual) ': 'fecha', 
            'XAB - Ingresos presupuestarios': 'ingresos_presupuestarios',
            'XAB2210 - Ingresos tributarios no petroleros': 'ingresos_tributarios_no_petroleros_neto',
            'XDB34 - ISR Total': 'isr_neto', 'XAB1120 - IVA': 'iva_neto',
            'XAB1130 - IEPS': 'ieps_neto', 
            'XBB10 - Ingresos tributarios': 'ingresos_tributarios_neto',
            'XAB30 - Ingresos no tributarios': 'ingresos_no_tributarios_neto'
            }, inplace=True)
        df = df.iloc[:-12]
        df.to_csv(file)

    df = pd.read_csv(file, usecols=columns)
    df.set_index(pd.to_datetime(df['fecha']), inplace=True)
    df.drop('fecha', axis=1, inplace=True)
    df['ingresos_tributarios_r']

    return df

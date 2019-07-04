'''
SHCP UPIT Forecasting Public Revenue
Key and params of model
'''
INEGI={
    'INEGI_URL': [('https://www.inegi.org.mx/app/api/indicadores/desarrolladores/'
                   'jsonxml/INDICATOR/'), '/es/00000/false/BIE/2.0/', '?type=json'],
    'INEGI_TOKEN': 'b95e6c18-9de9-393d-e550-080d2a5e37fe',
    'inpc_2018': '628194',
    'pibr_2013': '493621',
    'pibr_2013_sa': '493911',
    'igae': '496150',
    'igae_sa': '496216'
    }

INGRESOS_FISCALES= {
    'INGRESO_GASTO_SHCP_hist': 'https://www.secciones.hacienda.gob.mx/work/models/estadisticas_oportunas/datos_abiertos_eopf/ingreso_gasto_finan_hist.zip',
    'INGRESO_GASTO_SHCP_actual': 'https://www.secciones.hacienda.gob.mx/work/models/estadisticas_oportunas/datos_abiertos_eopf/ingreso_gasto_finan.zip',
    'RELEVANT_KEYS_SHCP': {'XAB': 'ingresos_sector_publico_neto',
                           'XBB':  'ingresos_gobierno_federal_neto',
                           'XDB34': 'isr_neto',
                           'XAB1120': 'iva_neto',
                           'XAB1130': 'ieps_neto',
                           'XBB10': 'ingresos_tributarios_neto',
                           'XAB30': 'ingresos_no_tributarios_neto'}
    }

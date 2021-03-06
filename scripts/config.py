'''
SHCP UPIT Forecasting Public Revenue
Key and params of model
'''
INEGI = {
    'INEGI_URL': 'https://www.inegi.org.mx/app/api/indicadores/desarrolladores/'
                   'jsonxml/INDICATOR/',
	'INEGI_BIE': '/es/00000/false/BIE/2.0/', 
    'INEGI_JSON': '?type=json',
    'INEGI_TOKEN': 'b95e6c18-9de9-393d-e550-080d2a5e37fe',
    'inpc_2018': '628194',
    'pibr_2013': '493621',
    'pibr_2013_sa': '493911',
    'igae': '496150',
    'igae_sa': '496216',
    'igae_prim': '496151',
    'igae_secun': '496152',
    'igae_terc': '496157',
    'confianza_consumidor': '63017',
    'indic_mens_consumo': '497613',
    'indic_adelant': '436141',
    'pea': '444620',
    'pobl_ocupada': '444622',
    'asegurados_imss': '215744',
    'imai': '496326',
    'imai_mineria': '496327',
    'imai_egergia_gas_agua_gas': '496331',
    'imai_construccion': '496334',
    'imai_manufacturas': '496338',
    'emec_menor_total': '655451',
    'emec_menor_aba_ali_beb_tab': '655452',
    'emec_menor_aba_ali': '655453',
    'emec_menor_hie_tab': '655454',
    'emec_menor_antad': '655455',
    'emer_menor_text_vest_calz': '655458',
    'emec_menor_pape_espar_otros': '655463',
    'emec_menor_domesticos': '655468',
    'emec_menor_vehic': '655474',
    'importaciones': '33226',
    'exportaciones': '33223'
}

INGRESOS_FISCALES = {
    'INGRESO_GASTO_SHCP_hist': 'https://www.secciones.hacienda.gob.mx/work/models/estadisticas_oportunas/datos_abiertos_eopf/ingreso_gasto_finan_hist.zip',
    'INGRESO_GASTO_SHCP_actual': 'https://www.secciones.hacienda.gob.mx/work/models/estadisticas_oportunas/datos_abiertos_eopf/ingreso_gasto_finan.zip',
    'RELEVANT_KEYS_SHCP': {'XAB': 'ingresos_sector_publico_neto',
                           'XBB':  'ingresos_gobierno_federal_neto',
                           'XBB10': 'ingresos_tributarios_neto',
                           'XBB20' : 'ingresos_no_tributarios_neto',
                           'XBB11' : 'isr_neto',
                           'XBB12' : 'iva_neto',
                           'XAB1130' : 'ieps_neto',
                           'XAB2213' : 'ieps_sin_gas_neto',
                           'XNA0117' : 'ieps_gasolina_neto',
                           'XNA0141' : 'isan_neto',
                           'XBB14' : 'importaciones_neto',
                           'XOA0801': 'impuesto_extraccion_hidrocarburos_neto',
                           'XNA0149': 'accesorios_contribuciones_neto',
                           'XNA0204' : 'otros_ingresos_tributarios_neto',
                           'XNA0226' : 'ieps_combustibles_fosiles_neto',
                           'XDB32' : 'ieps_hidrocarburos_neto',
                           'XDB33' : 'ieps_otros_petroliferos_neto',
                           'XNA0125' : 'ieps_tabaco_neto',
                           'XNA0126' : 'ieps_ba_neto',
                           'XNA0130' : 'ieps_otras_retenciones_neto',
                           'XNA0127' : 'ieps_cerveza_neto',
                           'XNA0211' : 'ieps_jys_neto',
                           'XNA0128' : 'ieps_telecom_neto',
                           'XNA0221' : 'ieps_be_neto',
                           'XNA0223' : 'ieps_bsaborizadas_neto',
                           'XNA0224' : 'ieps_aadc_neto',
                           'XNA0225' : 'ieps_plaguicidas_neto',
                           'XNA0209' : 'ietu_neto',
                           'XNA0210' : 'ide_neto',
                           'XNA0139' : 'exportaciones_neto'}
    }

BANXICO = {
    'BANXICO_URL': 'https://www.banxico.org.mx/SieAPIRest/service/v1/series/',
    'BANXICO_terminacion': '/datos/',
    'BANXICO_token': 'c774655e024da7bf5ce6d8d531ea12d19b6d44869e9ef35815052ae586dbf070',
    'tc_diario': 'SF43718',
    'tc_mensual': 'SF17908',
    'indice_tc_real': 'SR28',
    'tasa_cetes_28_diario': 'SF45470',
    'tasa_cetes_28_mensual': 'SF282',
    'tasa_cetes_91_diario': 'SF45471',
    'tasa_cetes_91_mensual': 'SF3338',
    'libor_3meses_mensual': 'SI561',
    'tbill_3meses_mensual': 'SI563',
    'tbill_6meses_mensual': 'SI564',
    'ingresos_presupuestarios': 'SG259',
    'tipo_de_cambio': 'http://www.banxico.org.mx/tipcamb/tipCamIHAction.do'
    }

FED = {
    'FED_URL': 'https://api.stlouisfed.org/fred/series/observations',
    'FED_file_type': 'json',
    'FED_token': '84e756065b7f9b52686517f129e0f772',
    'pibr_us_2012_sa': 'GDPC1',
    'pibr_us_2012': 'ND000334Q',
    'ind_prod_ind_us_sa': 'INDPRO',
    'ind_prod_ind_us': 'IPB50001N',
    'tbill_3meses_mensual': 'TB3MS',
    'tbill_3meses_diario': 'DTB3',
    'cons_price_index_us': 'CPIAUCNS',
    'cons_price_index_us_sa': 'CPIAUCSL',
    'trade_weighted_exchange_rate': 'TWEXBMTH',
    'commodity_price_index': 'PALLFNFINDEXM'
    }

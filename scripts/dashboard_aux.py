'''
Auxiliar para Dashboard
'''
columns = {
    'neto': ['ingresos_sector_publico_neto_(mdp)',
             'ing_gob_fed_neto_(mdp)_r',
             'ing_trib_neto_(mdp)_r',
             'ing_trib_sin_gasol_neto_(mdp)_r',
             'ing_no_trib_neto_(mdp)_r',
             'ieps_gasolina_neto_(mdp)_r',
             'isr_neto_(mdp)_r',
             'iva_neto_(mdp)_r',
             'ieps_sin_gas_neto_(mdp)_r',
             'ieps_gasolina_neto_(mdp)_r',
             'importaciones_neto_(mdp)_r',
             'otros_ingresos_tributarios_neto_(mdp)_r'],
    'bruto': ['ing_gob_fed_bruto_(mdp)_r',
              'ing_trib_bruto_(mdp)_r',
              'ing_no_trib_bruto_(mdp)_r',
              'isr_bruto_(mdp)_r',
              'iva_bruto_(mdp)_r',
              'ieps_bruto_(mdp)_r']}


cols_ingresos_gob_federal_neto = ['ing_gob_fed_neto_(mdp)_r',
                                  'ing_trib_neto_(mdp)_r',
                                  'ing_trib_sin_gasol_neto_(mdp)_r',\
                                  'ing_no_trib_neto_(mdp)_r',\
                                  'ieps_gasolina_neto_(mdp)_r']

cols_ingresos_tributarios_principales_netos = \
                            ['ing_trib_neto_(mdp)_r',
                             'isr_neto_(mdp)_r',
                             'iva_neto_(mdp)_r',
                             'ieps_sin_gas_neto_(mdp)_r',\
                             'ieps_gasolina_neto_(mdp)_r',
                             'importaciones_neto_(mdp)_r',
                             'otros_ingresos_tributarios_neto_(mdp)_r']

cols_ingresos_gob_federal_bruto = ['ing_gob_fed_bruto_(mdp)_r',
                                   'ing_trib_bruto_(mdp)_r',
                                   'ing_no_trib_bruto_(mdp)_r']

cols_ingresos_tributarios_principales_brutos = [
                             'ing_trib_bruto_(mdp)_r',
                             'isr_bruto_(mdp)_r',
                             'iva_bruto_(mdp)_r',\
                             'ieps_bruto_(mdp)_r']

cols_ingresos_tributarios_relevantes = \
                            ['ingresos_tributarios_sin_gasol_neto_(mdp)_r',
                             'isr_neto_(mdp)_r', 'iva_neto_(mdp)_r', 'ieps_sin_gas_neto_(mdp)_r',
                             'importaciones_neto_(mdp)_r']

cols_ingresos_tributarios_totales = \
                            ['ingresos_tributarios_sin_gasol_neto_(mdp)_r',
                             'isr_neto_(mdp)_r', 'iva_neto_(mdp)_r', 'ieps_sin_gas_neto_(mdp)_r',\
                             'ieps_gasolina_neto_(mdp)_r', 'importaciones_neto_(mdp)_r',\
                             'impuesto_extraccion_hidrocarburos_neto_(mdp)_r', 'isan_neto_(mdp)_r',\
                             'accesorios_contribuciones_neto_(mdp)_r', 'otros_ingresos_tributarios_neto_(mdp)_r']

graph_names = {'ingresos_sector_publico_neto_(mdp)': 'Ingresos sector público',
               'ing_gob_fed_neto_(mdp)_r': 'Ingresos Gobierno Federal',
               'ing_gob_fed_bruto_(mdp)_r': 'Ingresos Gobierno Federal',
               'ing_trib_neto_(mdp)_r': 'Ingresos tributarios',
               'ing_trib_bruto_(mdp)_r': 'Ingresos tributarios',
               'ing_trib_sin_gasol_neto_(mdp)_r': 'Ingresos tributarios sin IEPS gasolina',
               'ing_no_trib_neto_(mdp)_r': 'Ingresos no tributarios',
               'ing_no_trib_bruto_(mdp)_r': 'Ingresos no tributarios',
               'ieps_gasolina_neto_(mdp)_r': 'IEPS a las gasolinas',
               'ieps_bruto_(mdp)_r': 'IEPS',
               'isr_neto_(mdp)_r': 'ISR',
               'isr_bruto_(mdp)_r': 'ISR',
               'iva_neto_(mdp)_r': 'IVA',
               'iva_bruto_(mdp)_r': 'IVA',
               'ieps_sin_gas_neto_(mdp)_r': 'IEPS sin gasolinas',
               'importaciones_neto_(mdp)_r': 'Impuesto a las importaciones',
               'otros_ingresos_tributarios_neto_(mdp)_r': 'Otros ingresos tributarios'}
dev_comp_reg = {
    'compensaciones': {
        'IVA': 'iva_comp_(mdp)_r',
        'ISR': 'isr_comp_(mdp)_r',
        'IEPS': 'ieps_comp_(mdp)_r',
    },
    'devoluciones': {
        'IVA': 'iva_dev_(mdp)_r',
        'ISR': 'isr_dev_(mdp)_r',
        'IEPS': 'ieps_dev_(mdp)_r',
    },
    'regularizaciones': {
        'IVA': 'iva_reg_(mdp)_r',
        'ISR': 'isr_reg_(mdp)_r',
        'IEPS': 'ieps_reg_(mdp)_r',
    }
}


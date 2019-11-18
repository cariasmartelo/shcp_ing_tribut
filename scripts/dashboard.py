import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
import dash_table
import plotly.graph_objs as go
import download
import pandas as pd
import descriptive
import base64
import io

import dashboard_aux


cols_ingresos_gob_federal_neto = ['ing_gob_fed_neto_(mdp)_r',
                                  'ing_trib_sin_gasol_neto_(mdp)_r',\
                                  'ing_no_trib_neto_(mdp)_r',\
                                  'ieps_gasolina_neto_(mdp)_r']
cols_ingresos_tributarios_principales_netos = \
                            ['ing_trib_neto_(mdp)_r',
                             'isr_neto_(mdp)_r', 'iva_neto_(mdp)_r', 'ieps_sin_gas_neto_(mdp)_r',\
                             'ieps_gasolina_neto_(mdp)_r', 'importaciones_neto_(mdp)_r',
                             'otros_ingresos_tributarios_neto_(mdp)_r']
cols_ingresos_tributarios_principales_brutos = \
                            ['ing_trib_bruto_(mdp)_r',
                             'isr_bruto_(mdp)_r', 'iva_bruto_(mdp)_r',\
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

graph_names = {'ing_gob_fed_neto_(mdp)_r': 'Ingresos Gobierno Federal',
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

ingresos_totales = download.load_ingresos_fiscales(
    excel_brutos='../inputs/ingresos_tributarios_desglosados_updated.xlsx')

calendario_lif = download.load_calendario_lif()
# creamos lista de variables que vamos a comparar
observados_a_comparar = ['ing_gob_fed_neto_(mdp)',\
                         'ing_no_trib_neto_(mdp)_r',\
                         'ing_trib_neto_(mdp)_r',\
                         'isr_neto_(mdp)_r',\
                         'iva_neto_(mdp)_r',\
                         'ieps_neto_(mdp)_r',
                         'importaciones_neto_(mdp)_r']
presupuestados_a_comparar = ['ingresos_gobierno_federal_presupuestado_r',\
                             'ingresos_no_tributatios_presupuestado_r',\
                             'ingresos_tributarios_presupuestado_r',\
                             'isr_presupuestado_r',\
                             'iva_presupuestado_r',\
                             'ieps_presupuestado_r',
                             'importaciones_presupuestado_r']
# Hacemos un DF conjunto para que sea mas facil graficar y obtener estadísticas comparativas
df_comparacion = pd.merge(
                        calendario_lif[presupuestados_a_comparar],
                        ingresos_totales[observados_a_comparar],
                        left_index=True, right_index=True,
                        how='left')
df_comparacion = df_comparacion.loc['2019-01-01':]
# Vamos a graficar varias sub graficas. Para ello vamos a usar las listas de columnas.
# Creamos una lista de los titulos que usaremos
pares = {'ingresos del Gobierno Federal': ['ingresos_gobierno_federal_presupuestado_r',
                                           'ing_gob_fed_neto_(mdp)'],
         'ingresos no tributarios': ['ingresos_no_tributatios_presupuestado_r',
                                     'ing_no_trib_neto_(mdp)_r'],
         'ingresos tributarios': ['ingresos_tributarios_presupuestado_r',
                                  'ing_trib_neto_(mdp)_r'],
         'ingresos por ISR': ['isr_presupuestado_r', 'isr_neto_(mdp)_r'],
         'ingresos por IVA': ['iva_presupuestado_r', 'iva_neto_(mdp)_r'],
         'ingresos por IEPS': ['ieps_presupuestado_r', 'ieps_neto_(mdp)_r'],
         'ingresos por importaciones': ['importaciones_presupuestado_r',
                                        'importaciones_neto_(mdp)_r']}



colors = ['rgb(51,63,80)', 'rgb(105,26,48)', 'rgb(127,127,127)', '#aaa662', '#054907', '#030764',
          '#b59410', '#8c000f']

cols_to_compare = pares['ingresos del Gobierno Federal']
comparison = descriptive.cross_tab_lif(
    df_comparacion,cols_to_compare, cumsum=True, absolute_change=True,
    perc_change=True, title='ingresos del Gobierno Federal', style=False)
col_names=[str(col) for col in comparison]
comparison.columns=col_names
##### App

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([

    html.Div([
    html.H2('Dashboard UPIT'),
    html.Img(src='/assets/shcp.png')
    ], className='banner'),
    html.Div([
        html.H3('Series de ingresos tributarios'),
        dcc.Dropdown(
            id='tipo',
            options=[
                {'label': 'Ingresos fiscales brutos', 'value': 'brutos'},
                {'label': 'Ingresos fiscales netos', 'value': 'netos'}
            ],
            value='brutos',
            clearable=False,
        )
    ], className='twelve columns'),

    html.Div([
        html.Div([
            dcc.Graph(id='IngresosFiscales',)
        ]),
    ], className='twelve columns'),

    html.Div([
        html.H3('Comparativo anual de ingresos')
        ], className = 'twelve columns'),
    html.Div([
        dcc.Dropdown(
            id='comp_anual_brutos_netos',
            options=[
                {'label': 'Brutos', 'value': 'bruto'},
                {'label': 'Netos', 'value': 'neto'}
            ],
            value='bruto',
            clearable=False
        )], className = 'three columns'),
    html.Div([  
        dcc.Dropdown(
            id='comp_anual_columnas',
            clearable=False,

        )
    ], className='four columns'),
    html.Div([
        dcc.Dropdown(
            placeholder="Seleccione año final",
            id='comp_anual_anio_final',
            clearable=False,
        )
    ], className='two columns'),
    html.Div([
        dcc.Dropdown(
            id='comp_anual_anio_inicial',
            placeholder="Seleccione año inicial",
            clearable=False,
        )
    ], className='two columns'),
    html.Div([
        dcc.RadioItems(
            id='comp_anual_acumulado',
            options=[
                {'label': 'Flujo', 'value': 'flujo'},
                {'label': 'Acumulado', 'value': 'acumulado'},
            ],
            value = 'flujo',
            labelStyle={'display': 'inline-block'}
        )
    ], className='three columns'),
    html.Div([
        dcc.RadioItems(
            id='comp_anual_diferencia',
            options=[
                {'label': 'Valores', 'value': 'valores'},
                {'label': 'Diferencia', 'value': 'diferencia'},
            ],
            value = 'valores',
            labelStyle={'display': 'inline-block'}
        )
    ], className='two columns'),
    html.Div([
        dcc.RadioItems(
            id='comp_anual_percentual',
            labelStyle={'display': 'inline-block'}
        )
    ], className='two columns'),
    html.Div([
        html.Div([
            dcc.Graph(id='ComparativoAnual',)
        ], className='twelve columns'),
    ]),

    html.Div([
        html.H3('Comparativo con LIF'),
        dcc.Dropdown(
            id='comparativo',
            options=[
                {'label': 'Ingresos del Gobierno Federal', 'value': 'ingresos del Gobierno Federal'},
                {'label': 'Ingresos no tributarios', 'value': 'ingresos no tributarios'},
                {'label': 'Ingresos tributarios', 'value': 'ingresos tributarios'},
                {'label': 'Ingresos por ISR', 'value': 'ingresos por ISR'},
                {'label': 'Ingresos por IVA', 'value': 'ingresos por IVA'},
                {'label': 'Ingresos por IEPS', 'value': 'ingresos por IEPS'},
                {'label': 'Ingresos por Importaciones', 'value': 'ingresos por importaciones'}
            ],
            value='ingresos del Gobierno Federal',
            clearable=False,
        )], className = 'twelve columns'),
    html.Div([
        dcc.RadioItems(
            id='acumulado',
            options=[
                {'label': 'Flujo', 'value': 'flujo'},
                {'label': 'Acumulado', 'value': 'acumulado'},
            ],
            value = 'flujo',
            labelStyle={'display': 'inline-block'}
        )
    ], className='two columns'),
    html.Div([
        dcc.RadioItems(
            id='diferencia',
            options=[
                {'label': 'Valores', 'value': 'valores'},
                {'label': 'Diferencia', 'value': 'diferencia'},
            ],
            value = 'valores',
            labelStyle={'display': 'inline-block'}
        )
    ], className='two columns'),
    html.Div([
        dcc.RadioItems(
            id='porcentual',
            options=[
                {'label': 'MDP', 'value': 'MDP'},
                {'label': 'Porcentaje', 'value': 'Porcentaje'},
            ],
            value = 'MDP',
            labelStyle={'display': 'inline-block'}
        )
    ], className='two columns'),
    html.Div([
        html.Div([
            dcc.Graph(id='ComparativoLIF',)
        ], className='twelve columns'),
    ]),

    html.Div([
        html.H3('Comparativo compensaciones y devoluciones'),
        ], className='twelve columns'),
    html.Div([    
        dcc.Dropdown(
            id='comps_dev_tipo',
            options=[
                {'label': 'Compensaciones', 'value': 'compensaciones'},
                {'label': 'Devoluciones', 'value': 'devoluciones'},
                {'label': 'Regularizaciones', 'value': 'regularizaciones'},
            ],
            value=['compensaciones', 'devoluciones'],
            clearable=False,
            multi=True,
        )], className = 'three columns'),
    html.Div([
        dcc.Dropdown(
            id='comps_dev_tax',
            options=[
                {'label': 'IVA', 'value': 'IVA'},
                {'label': 'ISR', 'value': 'ISR'},
                {'label': 'IEPS', 'value': 'IEPS'}
            ],
            value=['IVA', 'ISR'],
            clearable=False,
            multi=True,
        )], className = 'four columns'),
    html.Div([
        dcc.Dropdown(
            placeholder="Seleccione año final",
            id='comps_dev_anio_final',
            options = [{'label': str(n), 'value': str(n)} for n
                        in range(2015, ingresos_totales.index.max().year + 1)],
            clearable=False,
            value = ingresos_totales.index.max().year
        )
    ], className='two columns'),
    html.Div([
        dcc.Dropdown(
            id='comps_dev_anio_inicial',
            placeholder="Seleccione año inicial",
            clearable=False,
        )
    ], className='two columns'),
    html.Div([
        dcc.RadioItems(
            id='comps_dev_acumulado',
            options=[
                {'label': 'Flujo', 'value': 'flujo'},
                {'label': 'Acumulado', 'value': 'acumulado'},
            ],
            value = 'flujo',
            labelStyle={'display': 'inline-block'}
        )
    ], className='three columns'),
    html.Div([
        dcc.RadioItems(
            id='comps_dev_diferencia',
            options=[
                {'label': 'Valores', 'value': 'valores'},
                {'label': 'Diferencia', 'value': 'diferencia'},
            ],
            value = 'valores',
            labelStyle={'display': 'inline-block'}
        )
    ], className='two columns'),
    html.Div([
        dcc.RadioItems(
            id='comps_dev_percentual',
            labelStyle={'display': 'inline-block'}
        )
    ], className='two columns'),
    html.Div([
        html.Div([
            dcc.Graph(id='CompensacionesDevoluciones',)
        ], className='twelve columns'),
    ]),
    ])


@app.callback(dash.dependencies.Output('IngresosFiscales', 'figure'), 
             [dash.dependencies.Input('tipo', 'value')])
def create_figure(tipo):
    '''
    Creamos figura
    '''
    dict_cols = {
        'brutos': cols_ingresos_tributarios_principales_brutos,
        'netos': cols_ingresos_tributarios_principales_netos}
    cols = dict_cols[tipo]
    df = ingresos_totales[cols].copy()
    df = df.loc[df.notna().all(1)]
    layout = go.Layout(title= 'Ingresos fiscales {} (MDP de 2019)'.format(tipo),
                       paper_bgcolor='rgba(255,255,255, 0.9)',
                       plot_bgcolor='rgba(255,255,255, 0.9)')
    traces = [go.Scatter(
                x = df.index,
                y = df[col],
                name = graph_names[col],
                line=dict(color=colors[i])) for i, col in enumerate(cols)]
    fig = go.Figure(data=traces, layout=layout,)
    fig.update_layout(legend_orientation="h")
    return fig




@app.callback(Output('comp_anual_columnas', 'options'),
              [Input('comp_anual_brutos_netos', 'value')])
def update_columns_comparativo_anual(tipo):
    columns = dashboard_aux.columns[tipo]
    options = [{'label': dashboard_aux.graph_names[col], 'value': col}
                for col in columns]
    return options

@app.callback(Output('comp_anual_columnas', 'value'),
              [Input('comp_anual_columnas', 'options')])
def update_columns_comparativo_anual_value(available_options):
    return available_options[0]['value']


@app.callback(Output('comp_anual_percentual', 'options'),
              [Input('comp_anual_diferencia', 'value')])
def update_percentual_comparativo_anual(diff):
    if diff == 'diferencia':
        options = [{'label': 'MDP', 'value': 'MDP'},
                   {'label': 'Porcentual',  'value': 'Porcentual'}]
    else:
        options = [{'label': 'MDP', 'value': 'MDP'}]
    return options

@app.callback(Output('comp_anual_percentual', 'value'),
             [Input('comp_anual_percentual', 'options')])
def update_percentual_value_comparativo_anual(available_options):
    return available_options[0]['value']


@app.callback(Output('comp_anual_anio_final', 'options'),
              [Input('comp_anual_columnas', 'value')])
def update_years_comparativo_anual_final(col):
    serie = ingresos_totales[col].copy()
    serie = serie[serie.notna()]
    min_year = serie.index.min().year + 1
    max_year = serie.index.max().year
    range_years = range(min_year, max_year + 1)

    options = [{'label': str(year), 'value': str(year)}
                for year in range_years]
    return options

@app.callback(Output('comp_anual_anio_final', 'value'),
              [Input('comp_anual_anio_final', 'options')])
def update_years_value_comparativo_anual_final(available_options):
    return available_options[-1]['value']


@app.callback(Output('comp_anual_anio_inicial', 'options'),
              [Input('comp_anual_columnas', 'value'),
               Input('comp_anual_anio_final', 'value')])

def update_years_comparativo_anual_inicial(col, anio_final):
    serie = ingresos_totales[col].copy()
    serie = serie[serie.notna()]
    min_year = serie.index.min().year
    max_year = int(anio_final)
    range_years = range(min_year, max_year)

    options = [{'label': str(year), 'value': str(year)}
                for year in range_years]
    return options

@app.callback(Output('comp_anual_anio_inicial', 'value'),
             [Input('comp_anual_anio_inicial', 'options')])

def update_years_value_comparativo_anual_inicial(available_options):
    return available_options[-1]['value']

@app.callback(Output('ComparativoAnual', 'figure'),
              [Input('comp_anual_columnas', 'value'),
               Input('comp_anual_anio_final', 'value'),
               Input('comp_anual_anio_inicial', 'value'),
               Input('comp_anual_acumulado', 'value'),
               Input('comp_anual_diferencia', 'value'),
               Input('comp_anual_percentual', 'value')])
def update_comparativo_anual_figure(column, anio_final, anio_inicial,
                                    cumsum, diff, perc):

    impuesto = dashboard_aux.graph_names[column]
    title = 'Comparativo {}'.format(impuesto)
    years_str = (str(anio_inicial), str(anio_final))
    if diff == 'diferencia':
        names = ['Diferencia entre {} y {}'.format(years_str[1], years_str[0])]
    cols = [anio_inicial, anio_final]

    comparativo_anual = descriptive.cross_tab(
        df=ingresos_totales,
        cols = [column],
        years=[anio_inicial, anio_final],
        absolute_change=False,
        perc_change=False,
        cumsum=False,
        style=False)

    comparativo_anual = comparativo_anual[column]

    if cumsum == 'acumulado':
        comparativo_anual = comparativo_anual.cumsum()
        title += ', acumulado anual '

    if diff == 'diferencia':
        comparativo_anual['diff'] = comparativo_anual[anio_final] -\
                                     comparativo_anual[anio_inicial]
        cols = ['diff']
        title += '<br>(Diferencia entre {} y {})'.format(
        str(anio_inicial), str(anio_final))

        if perc == 'Porcentual':
            comparativo_anual['diff_perc'] = (comparativo_anual['diff'] / 
                                              comparativo_anual[anio_inicial]) * 100
            cols = ['diff_perc']

    if perc == 'Porcentaje':
        title += ' (%)'
    else:
        title += ' (MDP de 2019)'

    layout = go.Layout(title= title,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)')
    traces = [go.Scatter(
                x = comparativo_anual.index,
                y = comparativo_anual[col],
                name = years_str[i],
                line=dict(color=colors[i])) for i, col in enumerate(cols)]
    fig = go.Figure(data=traces, layout=layout).update_layout(legend_orientation="h")
    if diff == 'diferencia':
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='red')
        fig.update_yaxes(range=[-1.5*(abs(comparativo_anual[cols[0]])).max(),
                         1.5*(abs(comparativo_anual[cols[0]])).max()])
    return fig






@app.callback(Output('ComparativoLIF', 'figure'), 
             [Input('comparativo', 'value'),
              Input('acumulado', 'value'),
              Input('diferencia', 'value'),
              Input('porcentual', 'value')])

def create_comparativo(impuesto, cumsum, diff, perc):
    '''
    Creamos figura
    '''
    names = ['Presupuestado', 'Observado']
    if diff == 'diferencia':
        names = ['Diferencia entre observado y presupuestado']

    cols_to_compare = pares[impuesto]
    df = df_comparacion.loc['2019-01-01':,cols_to_compare].copy()
    df = df.loc[df.notna().all(1)]
    cols = cols_to_compare
    title = 'Comparativo {}'

    if cumsum == 'acumulado':
        df = df.cumsum()
        title += ', acumulado anual '

    if diff == 'diferencia':
        df['diff'] = df[cols_to_compare[1]] - df[cols_to_compare[0]]
        cols = ['diff']
        title += '<br>(Diferencia entre Observado y Presupuestado)'

        if perc == 'Porcentaje':
            df['diff_perc'] = (df['diff'] / df[cols_to_compare[0]]) * 100
            cols = ['diff_perc']

    if perc == 'Porcentaje':
        title += ' (%)'
    else:
        title += ' (MDP de 2019)'

    layout = go.Layout(title= title.format(impuesto),
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)')
    traces = [go.Scatter(
                x = df.index,
                y = df[col],
                name = names[i],
                line=dict(color=colors[i])) for i, col in enumerate(cols)]
    fig = go.Figure(data=traces, layout=layout).update_layout(legend_orientation="h")
    if diff == 'diferencia':
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='red')
        fig.update_yaxes(range=[-1.5*(abs(df[cols[0]])).max(), 1.5*(abs(df[cols[0]])).max()])
    return fig



@app.callback(Output('comps_dev_percentual', 'options'),
              [Input('comps_dev_diferencia', 'value')])
def update_percentual_comps_dev(diff):
    if diff == 'diferencia':
        options = [{'label': 'MDP', 'value': 'MDP'},
                   {'label': 'Porcentual',  'value': 'Porcentual'}]
    else:
        options = [{'label': 'MDP', 'value': 'MDP'}]
    return options

@app.callback(Output('comps_dev_percentual', 'value'),
             [Input('comps_dev_percentual', 'options')])
def update_percentual_value_comps_dev(available_options):
    return available_options[0]['value']


@app.callback(Output('comps_dev_anio_inicial', 'options'),
               [Input('comps_dev_anio_final', 'value')])
def update_years_comps_dev_inicial(anio_final):
    min_year = 2014
    max_year = int(anio_final)
    range_years = range(min_year, max_year)

    options = [{'label': str(year), 'value': str(year)}
                for year in range_years]
    return options

@app.callback(Output('comps_dev_anio_inicial', 'value'),
             [Input('comps_dev_anio_inicial', 'options')])
def update_years_value_comps_dev_inicial(available_options):
    return available_options[-1]['value']

@app.callback(Output('CompensacionesDevoluciones', 'figure'),
              [Input('comps_dev_tipo', 'value'),
               Input('comps_dev_tax', 'value'),
               Input('comps_dev_anio_final', 'value'),
               Input('comps_dev_anio_inicial', 'value'),
               Input('comps_dev_acumulado', 'value'),
               Input('comps_dev_diferencia', 'value'),
               Input('comps_dev_percentual', 'value')])
def update_comparativo_anual_figure(tipos, taxes, anio_final, anio_inicial,
                                    cumsum, diff, perc):
    title = 'Gastos fiscales, '
    cols = []
    dev_comp_reg_dict = dashboard_aux.dev_comp_reg
    for gasto_fiscal in tipos:
        for tax in taxes:
            cols.append(dev_comp_reg_dict[gasto_fiscal][tax])
    df_to_plot = ingresos_totales[cols].copy()
    df_to_plot['total'] = df_to_plot.sum(1)

    comparativo_anual = descriptive.cross_tab(
        df=df_to_plot,
        cols = ['total'],
        years=[anio_inicial, anio_final],
        absolute_change=False,
        perc_change=False,
        cumsum=False,
        style=False)
    comparativo_anual = comparativo_anual['total']


    title = 'Comparativo gastos fiscales'
    if len(tipos) > 1:
        gastos_en_tit = ' + '.join(tipos)
    else:
        gastos_en_tit = tipos[0]
    if len(taxes) > 1:
        taxes_en_tit = ' + '.join(tipos)
    else:
        taxes_en_tit = taxes[0]
    title += '<br>' + gastos_en_tit + ' de ' + taxes_en_tit + '<br>'
    years_str = (str(anio_inicial), str(anio_final))
    if diff == 'diferencia':
        names = ['Diferencia entre {} y {}'.format(years_str[1], years_str[0])]
    cols = [anio_inicial, anio_final]

    if cumsum == 'acumulado':
        comparativo_anual = comparativo_anual.cumsum()
        title += 'Acumulado anual, '

    if diff == 'diferencia':
        comparativo_anual['diff'] = comparativo_anual[anio_final] -\
                                     comparativo_anual[anio_inicial]
        cols = ['diff']
        title += 'Diferencia entre {} y {}, '.format(
        str(anio_inicial), str(anio_final))

        if perc == 'Porcentual':
            comparativo_anual['diff_perc'] = (comparativo_anual['diff'] / 
                                              comparativo_anual[anio_inicial]) * 100
            cols = ['diff_perc']

    if perc == 'Porcentaje':
        title += ' (%)'
    else:
        title += ' (MDP de 2019)'

    layout = go.Layout(title= title,
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)')
    traces = [go.Scatter(
                x = comparativo_anual.index,
                y = comparativo_anual[col],
                name = years_str[i],
                line=dict(color=colors[i])) for i, col in enumerate(cols)]
    fig = go.Figure(data=traces, layout=layout).update_layout(legend_orientation="h")
    if diff == 'diferencia':
        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='red')
        fig.update_yaxes(range=[-1.5*(abs(comparativo_anual[cols[0]])).max(),
                         1.5*(abs(comparativo_anual[cols[0]])).max()])
    return fig




if __name__ == "__main__":
    app.run_server(debug=False)



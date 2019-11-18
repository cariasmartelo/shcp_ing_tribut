'''
SHCP UPIT Forecasting Public Revenue
Models
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.tsa.vector_ar.var_model import VAR
from dateutil.relativedelta import relativedelta
import descriptive
import matplotlib as mpl
from fbprophet import Prophet
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

def split_param_in_constr_and_fit(params, model_name):
    '''
    Para correr varias versiones de los modelos y seleccionar el que mejor
    se ajusta, es necesario correr cada uno de los modelos varias veces. 
    Para ello, la función run_prediction en vez de tomar un solo diccionario
    de parámetros, toma una lista de diccionarios de parámetros. Dependiendo
    del modelo que se va a estimar, estos parámetros deben especificarse
    al crear el objeto (por ejemplo al llamar model = VAR() o 
    model = ARIMA()), o al hacer el siguiente paso que es model.fit(). Esta
    función divide un diccionario único de parámetros de dos diccionarios,
    uno que se usa al crear el model y otro al hacer .fit. Para ello, crea
    un diccionario splitter que mapea para cada modelo, los parámetors que
    van en el constructor y los parámetros que van en fit.
    Inputs:
        params: diccionario
        model_name: str ('ARIMA', 'SARIMA', 'VAR', 'DT', 'GB', 'RF')
    '''
    # Creamos diccionario que usaremos para hacer el split. Es un diccionario
    # que mepea los modelos ARIMA y SARIMA con un diccionario que mapea los
    # parámetros que van al constructos y los parámetros que van al fit. 
    # Estos los obtuve de las páginas de internet que describen el uso de los
    # modelos. 
    splitter = {
    'ARIMA': {'constructor': ['order'],
              'fit': ['transparams', 'method', 'trend', 'solver', 'maxiter',
                      'tol', 'disp', 'callback',  'start_ar_lags']},
    'SARIMA': {'constructor': ['order', 'seasonal_order', 'trend', 
                               'measurement_error', 'time_varying_regression',
                               'mle_regression', 'simple_differencing',
                               'enforce_stationarity', 'enforce_invertibility',
                               'hamilton_representation', 'concentrate_scale',
                               'trend_offset'],
               'fit': ['transformed', 'cov_type', 'cov_kwds', 'method',
                       'maxiter', 'full_output', 'disp', 'callback', 'return_params',
                       'optim_score', 'optim_complex_step', 'optim_hessian']}}
    # Si el modelo es VAR< ya sabemos que todo va al momento de hacer fit y
    # nada al momento de construir. Por ello, creamos params_consts como un diccionario
    # vacio y asignamos a params_fit todos los parámetros indicados en el input.
    if model_name == 'VAR':
        params_constr = {}
        params_fit = params
        return params_constr, params_fit
    # Si el modelo es alfuno de DT, RF, GB o LR, todos los parámetros se indican al
    # onstruir, por lo que hacemos exactamente lo al revez que con el VAR.
    elif model_name in ['DT', 'RF', 'GB', 'LR']:
        params_constr = params
        params_fit = {}
        return params_constr, params_fit
    # De lo contrario, si es ARIMA o SARIMA, hacemos la división del duccionario.
    else:
        # Del diccionario que el usuario especificó, obtenemos las que están
        # bajo el nombre de 'constructor' y bajo el nombre de 'fit' haciendo un 
        # loop
        constructor_keys = [key for key in splitter[model_name]['constructor']
                            if key in params]
        fitter_keys = [key for key in splitter[model_name]['fit']
                       if key in params]
        # Usando esas llaves, creamos los dos diccionarios.
        params_constr = {key: params[key] for key in constructor_keys}
        params_fit = {key: params[key] for key in fitter_keys}

        return params_constr, params_fit


def construct_exog(exog_base, transformation=None, pct_changes=None,
                   month_dummies=None, year_dummies=None, lags=None,
                   notna=True):
    '''
    Ampliar el DF de variables exeogenas, añadiendo transformaciones, cambios
    porcentuales, dummies de meses, dummies de años, lags y filtrando los NAs.
    Inputs:
        exog_base: DF
        pct_changes: list with the pct changes to include
        month_dummies: Bool or list with months
        year_dummies: list with years
        lags: int (number of lags)
    return
        df
    '''
    # Creamos una copia de la base exógena para evitar modificar la base original.
    exog_copy = exog_base.copy()
    # Si el usuario especifica una transformación:
    if transformation:
        # Se trabsforma usando la función transformation del script descriptive.
        exog_copy = exog_copy.apply(lambda x: 
            descriptive.transformation(x, transformation))
    # Si el usuario especifica cambios porcentuales,
    if pct_changes:
        # Creamos un nuevo df con el mismo indice que el exog_copy, al cual
        # concatenaremos todos los DF de cambios porcentuales.
        exog_pct_changes = pd.DataFrame(index=exog_copy.index)
        # Hacemos un loop para cada uno de los cambios porcentuales
        # indicados
        for change in pct_changes:
            # Cramos un df que específico al cambio porcentual relevante.
            exog_pct_change = exog_base.pct_change(change)
            # Añadimos el prefijo 'pct_change'
            exog_pct_change = exog_pct_change.add_prefix('pct_change_')
            # Añadimos el suffix del cambio porcentual que estamos obteniendo.
            exog_pct_change = exog_pct_change.add_suffix('_{}'.format(change))
            # Concatenamos el df del cambio porcentual con el DF que habiamos
            # creado anteriormente.
            exog_pct_changes = pd.concat([exog_pct_changes, exog_pct_change],
                                         axis=1)
        # Concatenamos el DF que tiene todos los cambios porcentuales al
        # DF de variables exógenas.
        exog_copy = pd.concat([exog_copy, exog_pct_changes], axis=1)
    # Si el usuario especifica lags, hacemos lo mismo que con los cambios
    # porcentuales, pero usamos la función específica de lags que creamos. 
    if lags:
        exog_copy = add_lags(exog_copy, lags)
    # Si el usuario especifica dummies de meses, ya sea True para todos
    # los meses, o una lista de meses: 
    if month_dummies:
        # Obtenemos una serie del índice que indica el mes
        months_series = pd.Series(exog_copy.index.map(lambda x: x.month))
        # Asignamos el mismo índice a la serie de meses, pues al obtenerlos
        # no se guarda el índice.
        months_series.index = exog_copy.index
        # Creamos dummies usando la funcieon get dummies de pandas. Se crea
        # un Df con una columna para cada mes, asignada al valor 1 o 0.
        dummies = pd.get_dummies(months_series)
        # Si el ususario especificó una lista de meses, nos quedamos unicamente
        # con las dummies de esos meses filtrando el DF de dummies
        if isinstance(month_dummies, list):
            dummies = dummies[month_dummies]
        # Añadimos el prefijo mes_ a cada dummy.
        dummies = dummies.add_prefix('mes_')
        # Concatenamos las dummies al DF de exog.
        exog_copy = pd.concat([exog_copy, dummies], axis=1)
    # Si el ususario especifica dummies de años, hacemos lo mismo que con los
    # meses.
    if year_dummies:
        year_series = pd.Series(exog_copy.index.map(lambda x: x.year))
        year_series.index = exog_copy.index
        dummies = pd.get_dummies(year_series)
        if isinstance(year_dummies, list):
            dummies = dummies[year_dummies]
        dummies = dummies.add_prefix('year_')
        exog_copy = pd.concat([exog_copy, dummies], axis=1)
    # Si el usuario especifica notna, o si se toma el valor por default,
    # nos quedamos unicamente con las observaciones completas.
    if notna:
        exog_copy = exog_copy.loc[exog_copy.notna().all(1)]

    return exog_copy

def add_lags(df, lags):
    '''
    add lags to df.
    Inputs:
        df: DF
        lags = list
    Output:
        df with lags
    '''
    # Creamos una copia para no modificar original
    df_copy = df.copy()
    # Creamos un DF vacio que tendrá todos los lags.
    df_lags = pd.DataFrame(index=df_copy.index)
    # Hacemos un loop para cada lag
    for lag in lags:
        # Creamos un DF especifico al lag. Añadimos el prefijo 'lag_'
        df_lag = df_copy.shift(lag).add_prefix('lag_')
        # Añadimos el suffix del numero de lag que estamos creando.
        df_lag = df_lag.add_suffix('_{}'.format(lag))
        # Concatenamos con DF que abarca todos los lags.
        df_lags = pd.concat([df_lags, df_lag], axis=1)
    # Concatenamos con el DF copia.
    df_copy = pd.concat([df_copy, df_lags], axis=1)

    return df_copy


def split_train_test(df, test_begin, notna=True):
    '''
    Split between train and test using the date of the start of test
    given.
    Inputs:
        df: Pd.DataFrame
        test_begin: datetime
        notna: Bool
    Output:
        (train, test)
    '''
    # Creamos DF copia para no modificar original.
    df_copy = df.copy()
    # Obtenemos el df train usando la fecha inicial. La fecha inicial
    # la tomamos como la fecha inicial del test set.
    train = df.loc[df.index < test_begin]
    test = df.loc[test_begin:]
    # Eliminamos valores NA's si el usuario lo indica, o si se toma valor
    # default.
    if notna:
        train = train.loc[train.notna().all(1)]

    return (train, test)

def run_model(model_name, endog, exog_train=None, exog_test=None,
              prediction_start=None, prediction_end=None,
              params_constr=None, params_fit=None, freq='MS'):
    '''
    Correr un modelo predictivo para un train y un test set especifico, y
    para un set de parámetros específicos. Esta es la función sustancia del script
    porque es la que construye los objetos modelo, la que hace fit y la que
    predice.
    Inputs:
        model_name: str
        endog: DF
        exog_Ttain: DF
        exog_test: DF
        prediction_start: Date
        prediction_end: Date
        params_constr: Diccionario de parámetros para construir modelo
        paramos_fit: Diccionario de parámetros para hacer fit.
    Output:
        predicciones (DF o series)
    '''
    # Si no se especifican parámetros para construir, los asignamos a un
    # diccionario vacío.
    if not params_constr:
        params_constr = {}
    # Lo mismo para los parámetros de fit.
    if not params_fit:
        params_fit={}
    # Creamos un diccionario que mapea los nombres de los modelos con los
    # constructores de cada modelo.
    model_dict = {'ARIMA': ARIMA,
                  'SARIMA': sm.tsa.statespace.SARIMAX,
                  'VAR': VAR,
                  'DT': DecisionTreeRegressor,
                  'RF': RandomForestRegressor,
                  'GB': GradientBoostingRegressor,
                  'LR': LogisticRegression}
    # Usamos el diccionario para obtener el objeto model que usaremos.
    model = model_dict[model_name]
    # Si se trata de un modelo econométrico, statsmodelos piden que
    # coloquemos el DF endogeno y el DF exogeno y algunos parámetros
    # al momento de construir, y algunos parámetros después al momento
    # de hacer fit
    if model_name in ['ARIMA', 'SARIMA', 'VAR']:
        regr = model(endog=endog, exog=exog_train, **params_constr)
        # Y parámetros en fit
        regr = regr.fit(**params_fit)
    # Si es un modelo de ML, la libreria sklearn pide que coloquemos
    # los parámetros al consturir el modelo, y luego el X_train, y Y_train
    # al hacer .fit()
    elif model_name in ['RF', 'DT', 'GB', 'LR']:
        regr = model(**params_constr)
        # Y parámetros en fit
        regr = regr.fit(exog_train, endog)
    # Una vez construido el modelo, hacemos predicciones. Si el modelo
    # es univariado, es sencillo. Llamamos método precit con la fecha inicial
    # y la fecha final y el DF de variables exógenas para la predicción.
    if model_name in ['ARIMA', 'SARIMA']:
        predictions = regr.predict(start=prediction_start, end=prediction_end, 
                                  exog=exog_test)
    # Si el modelo es un var, tenemos que primero obtener el numero de lags que
    # el modelo eligió.
    elif model_name in ['VAR']:
        lags_used = regr.k_ar
        print('LAGS_USED', lags_used)
        # Caso esquina en el que el modelo selecciona 0 como el mejor AR.
        # Lo forzamos a que sea 1 usando maxlags=None
        if lags_used == 0:
            regr = model(endog=endog, exog=exog_train, **params_constr)
        # Y parámetros en fit
            regr = regr.fit(maxlags=None)
        # Imprimimos una notoficación de que corregimos esto.
            print('VAR model selected lags 0, for params {} and split {}'
                  .format(params_fit, prediction_start))
        # En este particular, sabremos que los lags usados fueron 1
            lags_used = 1
        # Obtenemos el numero de pasos a predecir, haciendo la resta entre
        # el final de la predicción y el inicio.
        steps = relativedelta(prediction_end, prediction_start)
        # steps gives number of years and number of months.
        # + 1 because we need to count initial month
        if freq == 'MS':
            steps = steps.years * 12 + steps.months + 1
        elif freq =='YS':
            steps = steps.years + 1
        # Usamos el métdo forecast, que pide la variable endógena que se
        # va a usar (que tiene los lags necesarios), el numero de pasos y
        # el DF de variables exógenas.
        prediction  = regr.forecast(y=endog.values[-lags_used:],
            steps=steps, exog_future=exog_test)
        # Creamos las fechas de predicción usando pd.date_range
        if freq == 'MS':
            prediction_dates = pd.date_range(
                prediction_start, prediction_end, freq='MS')
        elif freq == 'YS':
            prediction_dates = pd.date_range(
                prediction_start, prediction_end, freq='YS')

        # Creamos un DF donde pondremos las predicciones y especificamos
        # el indice.
        predictions = pd.DataFrame(index=prediction_dates)
        # El resultado de forecast es un numpy array de dimensiones
        # (steps, cols) Hacemos un loop para extraer cada una de las
        # predicciones y asignarla al DF.
        for i, var in enumerate(endog.columns):
            var_pred = prediction[:,i]
            var_pred = pd.Series(var_pred, index=prediction_dates)
            predictions[var] = var_pred
    # FInalmente, si el modelo es de ML, usamod el método predict que unicamente
    # toma X test. En este caso, la función que esta llamando a run_model
    # que sería (predict_with_ml_models), llamará run models el numero de 
    # pasos a predecir.
    elif model_name in ['RF', 'DT', 'GB', 'LR']:
        try:
            predictions = regr.predict(exog_test)
        except:
            print('Could not predict with {}'
                  ', for split {}, params {}'.format(
                model_name, prediction_start, params_constr))
            return False

    return predictions

def predict_with_econometric_model(model_name, params, df, endog_vars, prediction_start,
                                   prediction_end, transformation, lags_endog=None,
                                   exog_df=None, train_start=None, freq='MS'):
    '''
    Predict var model usando VAR. Principalmente es para hacer predicciones
    de las variables de USA y de las variables exógenas de México.
    '''
    # Creamos DF para modelo VAR
    df_copy = df[endog_vars].apply(
            lambda x: descriptive.transformation(x, transformation))
    if train_start:
        df_copy = df_copy.loc[train_start:]    
    df_copy = df_copy.loc[df_copy.notna().all(1)]
    # Asignamos var_exog a None, despues cambiamos eso si es necesario
    var_exog_train = None
    var_exog_test = None
    if not exog_df is None: 
        exog_df = exog_df[df_copy.index.min():]
        # Dividimos entre train y test:
        var_exog_train, var_exog_test = split_train_test(exog_df,
                                                         prediction_start)
        # Igualamos índices
        df_copy = df_copy.loc[var_exog_train.index]

    # Separamos diccionario de parametros entre fit y constructor
    params_constr, params_fit = split_param_in_constr_and_fit(params, model_name)
    # Obtenemos predicciones
    try:
        predictions_d = run_model(model_name=model_name,
                                  endog=df_copy,
                                  exog_train=var_exog_train,
                                  exog_test=var_exog_test, 
                                  prediction_start=prediction_start,
                                  prediction_end=prediction_end,
                                  params_constr=params_constr,
                                  params_fit=params_fit,
                                  freq=freq)
    except:
        print('model {} for split {} for var {} did not converge'.format(
            model_name, prediction_start, endog_vars))
        return (False, False)
    # Concatenamos predicciones con valores anteriores
    # predictions_d = pd.concat([df_copy, predictions_d])
    # Obtenemos predicciones en niveles, for var
    if freq=='MS':
        initial_date = predictions_d.index.min() - relativedelta(months=1)
    elif freq=='YS':
        initial_date = predictions_d.index.min() - relativedelta(years=1)
    if isinstance(predictions_d, pd.core.series.Series):
        predictions_d = predictions_d.rename(endog_vars[0])
        predictions = descriptive.revert_transformation(predictions_d,
            transformation, float(df.loc[initial_date, predictions_d.name]),
            initial_date)
    else:
        predictions = predictions_d.apply(lambda x: 
            descriptive.revert_transformation(x, transformation, 
                float(df.loc[initial_date, x.name]), initial_date))
    return(predictions_d, predictions.loc[prediction_start:])

def predict_with_ml_model(model_name, params, df, endog_vars, prediction_start,
                           prediction_end, transformation, lags_endog,
                           exog_df=None, train_start=None, freq='MS'):
    '''
    Predict var model usando ML. Principalmente es para hacer predicciones
    de las variables de USA y de las variables exógenas de México.
    '''
    # Creamos DF que no tocaremos
    df_to_predict = df.loc[df.index < prediction_start, endog_vars].copy()
    # Creamos copia que sí tocaremos
    df_to_predict_predicted = df_to_predict.copy()
    if train_start:
        df_to_predict_predicted = df_to_predict_predicted.loc[train_start:]
    df_to_predict_predicted = df_to_predict_predicted.loc[df_to_predict_predicted.notna().all(1)]
    # Creamos diferencia de logaritmos de igresos a predecir.
    d_df_to_predict = df_to_predict_predicted.apply(
        lambda x: descriptive.transformation(x, transformation))
    # Split params
    params_constr, params_fut = split_param_in_constr_and_fit(params, model_name)
    prediction_start_running = prediction_start
    while prediction_start_running <= prediction_end:
        # Creamos array endogeno
        X_Y = build_x_y(endog=df_to_predict_predicted,
                        endog_transformed=d_df_to_predict,
                        exog=exog_df, prefix_transformed=transformation + '_',
                        lags_endog=lags_endog)
        # Obtenemos begin_test, que en este caso es un mes antes de begin
        if freq=='MS':
            begin_test = prediction_start - relativedelta(months=1)
        elif freq=='YS':
            begin_test = prediction_start - relativedelta(years=1)
        train, test =  split_train_test(X_Y, begin_test)
       # Filtramos train y obtenemos X y Y usando el orden de las columnas
        train = train.loc[train.notna().all(1)]
        X_train = train.iloc[:,:-2]
        X_test = test.iloc[:,:-2]
        Y_train = train.iloc[:,-1]
        Y_test = test.iloc[:,-1]
        predicted_labels = test.iloc[:,-2]
        prediction_d = run_model(model_name=model_name,
                                 endog=Y_train,
                                 exog_train=X_train,
                                 exog_test=X_test, 
                                 params_constr=params_constr,
                                 freq=freq)
        if prediction_d is False:
            return (False, False)

        dictionary_predictions = {key: prediction_d[i] for i, key in\
                                  enumerate(predicted_labels.values)}
        preds_to_append = pd.DataFrame(dictionary_predictions,
                                        index=[prediction_start_running])
        # Hacemos append
        d_df_to_predict = d_df_to_predict.append(preds_to_append, sort=True)
        # Obtenemos valores iniciales para revertir
        start_transformed = d_df_to_predict.loc[d_df_to_predict.notna().all(1)]\
                                .index.min()
        if freq == 'MS':
           initial_date = start_transformed - relativedelta(months=1)
        elif freq == 'YS':
           initial_date = start_transformed - relativedelta(years=1)

        initial_values = df_to_predict_predicted.loc[initial_date]
        # Construios de nuevo ingresos a predecir predicted
        df_to_predict_predicted = d_df_to_predict.apply(
            lambda x: descriptive.revert_transformation(x, transformation, 
                initial_values[x.name], initial_date))
        if freq == 'MS':
            prediction_start_running = prediction_start_running\
            + relativedelta(months=1)
        elif freq == 'YS':
            prediction_start_running = prediction_start_running\
            + relativedelta(years=1)

    return(d_df_to_predict, df_to_predict_predicted.loc[prediction_start:])


def build_x_y(endog, endog_transformed, exog, prefix_transformed,
              lags_endog=None):
    '''
    Build X_Y df using the endogenous df, the endogenous df transformed
    to the format that will be predicted, and the egogeonous df.
    Inputs:
        endog: DF
        exog: Df
        exog: DF
        lags_endog: list
    Outut:
        DF
    '''
    ml_endog = pd.concat([endog, endog_transformed.add_prefix(prefix_transformed)],
                         axis=1)
    ml_endog = add_lags(ml_endog, lags_endog)
    # Creamos variable a predecir
    outcome_var = endog_transformed.shift(-1).copy()
    # Creamos variable de fecha para hacer el melt
    outcome_var['fecha'] = outcome_var.index
    # Hacemos el melt para tener una variable y no cuatro por fecha
    outcome_var = pd.melt(outcome_var, id_vars = ['fecha'])\
                    .sort_values(['fecha', 'variable'])
    # Regresamos la fecha al índice
    outcome_var.index = outcome_var['fecha']
    outcome_var.drop('fecha', axis=1, inplace=True)
    # Creamos dummies de las variables a predecir
    dummies_taxes_to_predict = pd.get_dummies(outcome_var['variable'],
                                             prefix='to_predict')
    # Obtenemos los nombres de las summies que hicimos
    dummies_names = [col for col in dummies_taxes_to_predict]
    # Concatenamos las dummies con la variable outcome
    outcome_var = pd.concat([outcome_var, dummies_taxes_to_predict], axis=1)
    # Ordenamos las variables
    outcome_var = outcome_var[dummies_names + ['variable', 'value']]
    # contarenamos el df endogeno con el df de outcome.
    ml_endog_outcome = ml_endog.merge(outcome_var, left_index=True, right_index=True, how='left')
    # Concatenamos el df exógeno con el endógeno y el outcome.
    X_Y = pd.merge(exog, ml_endog_outcome, left_index=True, right_index=True)

    return X_Y

def run_predictions(model_name, params, begin_and_ends, df, endog_cols,
                    endog_transformation,  endog_lags=None, exog=None,
                    exog_transformation=None, us_prediction_dict=None,
                    us_prediction_vars=None, mex_prediction_dict=None, 
                    mex_prediction_vars=None, exog_vars=None, other_exog=None,
                    exog_pct_changes=None, exog_month_dummies=None,
                    exog_year_dummies=None, exog_lags=None, train_start=None,
                    freq='MS'):
    '''
    Wrapper of run predictions. It will build the exog df, it will rin model
    '''
    predictions = {}
    results_accuracies = []
    yearly_differences = []
    for prediction_period in begin_and_ends:
        begin, end = prediction_period
        steps = relativedelta(end, begin)
        # steps gives number of years and number of months.
        # + 1 because we need to count initial month
        if freq=='MS':
            steps = steps.years * 12 + steps.months + 1
        elif freq=='YS':
            steps = steps.years + 1
        predictions[begin] = {}
        if exog:
            exog_df = pd.DataFrame(index=pd.date_range(
                    df.index.min(), end, freq=freq))
            if not exog_vars is False:
                if mex_prediction_dict:
                    mex_exog = mex_prediction_dict[begin].copy()
                    if mex_prediction_vars:
                        mex_exog = mex_exog[mex_prediction_vars]
                    exog_df = exog_df.merge(mex_exog, left_index=True,
                                      right_index=True)
                if us_prediction_dict:
                    us_exog = us_prediction_dict[begin].copy()
                    if us_prediction_vars:
                        us_exog = us_exog[us_prediction_vars]
                    exog_df = exog_df.merge(us_exog, left_index=True,
                                  right_index=True)
                if exog_vars:
                    exog_df = exog_df[exog_vars] 
            exog_df = construct_exog(exog_df, exog_transformation, exog_pct_changes,
                                  exog_month_dummies, exog_year_dummies, exog_lags)
        else:
            exog_df = None
        if not other_exog is None:
            exog_df = exog_df.merge(other_exog, left_index=True, right_index=True)


        if model_name in ['ARIMA', 'SARIMA', 'VAR']:
            run_function = predict_with_econometric_model
        if model_name in ['DT', 'RF', 'GB', 'LR']:
            run_function = predict_with_ml_model
        for param in params:
            _, prediction = run_function(model_name=model_name,
                                         params=param,
                                         df=df,
                                         endog_vars=endog_cols,
                                         prediction_start=begin,
                                         prediction_end=end,
                                         transformation=endog_transformation,
                                         lags_endog=endog_lags,
                                         exog_df=exog_df,
                                         train_start=train_start,
                                         freq=freq)
            if prediction is False:
                continue
            if isinstance(prediction, pd.core.series.Series):
                prediction = prediction.rename(endog_cols[0]).to_frame()
            # Obtenemos prediccion concatenada
            if freq=='MS':
                observed_values = df.loc[:begin - relativedelta(months=1),
                                         endog_cols]
            elif freq=='YS':
                observed_values = df.loc[:begin - relativedelta(years=1),
                                         endog_cols]
            obs_and_pred = pd.concat([observed_values, prediction], sort=True)
            predictions[begin][str(param)] = obs_and_pred
            # Calculamos precisión
            if freq == 'MS':
                pred_period = str(steps) + 'MS'
            elif freq=='YS':
                pred_period = str(steps) + 'YS'

            exog_cols = list(exog_df.columns)
            m_dict_results = get_dict_results(
                df, prediction, begin, end, model_name, param, pred_period,
                exog_cols, endog_cols, freq)
            print('COMPLETED {} for {} with params {} for split {}'.format(
                model_name, endog_cols, param, begin))
            if freq == 'MS':
                y_prediction = obs_and_pred.resample('YS').sum()
                y_df = df.resample('YS').sum()
                predicted_close = y_prediction.iloc[-2:]
                observed_close = y_df[endog_cols].loc[predicted_close.index]
                # print(predicted_close)
                # print(observed_close)
                yearly_diffs = observed_close - predicted_close
                if isinstance(yearly_diffs, pd.core.series.Series):
                    yearly_diffs = yearly_diffs.to_frame()
                yearly_dict = yearly_diffs.to_dict()
                yearly_dict['params'] = str(param)
                yearly_dict['model'] = model_name
                yearly_dict['split_date'] = begin
                yearly_dict['endog_vars'] = str(endog_cols)
                yearly_dict['exog_vars'] = str(exog_vars)
                yearly_differences.append(yearly_dict)

            results_accuracies += m_dict_results

    return (predictions, results_accuracies, yearly_differences)


def compute_accuracy_scores(observed, predicted, output_dict=True):
    '''
    From the predicted series and the real series, compute mean root squared error and
    mean absolute error.
    '''
    accuracy_scores = {}
    resid = observed - predicted
    rss = resid ** 2
    accuracy_scores['rmse'] = np.sqrt(rss.mean())
    accuracy_scores['mae'] = abs(resid).mean()
    accuracy_scores['mape'] = (abs(resid / observed)).mean()
    accuracy_scores['forecast_biass'] = resid.mean()
    if output_dict:
        return accuracy_scores
    else:
        return (accuracy_scores['rmse'], accuracy_scores['mae'],
                accuracy_scores['mape'], accuracy_scores['forecast_biass'])

def get_dict_results(df, prediction, begin, end, model_name,
                     param, pred_period, exog_cols, endog_cols,
                     freq='MS'):
    '''
    Build diccionary of accuracies for different time horizons: First 6
    months, last 12 months and the entire period.
    Inputs:
        df: DataFrame
        prediction: Series
        begin: Date
        end: Date
        model_name: Str
        param: Dict
        pred_period: Str
        exog_cols: list
        endig_cols: list
    Ouput:
        Dictionary
    '''
    results_accuracies = []
    if freq == 'MS':
        accuracy_periods = {}
        accuracy_periods['first6'] = pd.date_range(
            start=begin, end=begin + relativedelta(months=5), freq='MS')
        accuracy_periods['last12'] = pd.date_range(
            start=begin + relativedelta(months=6), end=end, freq='MS')
        accuracy_periods['first18'] = pd.date_range(
            start=begin, end=end, freq='MS')
        # accuracy_periods['second_close'] = pd.date_range(
        #     start=begin + relativedelta(months=6), end=end, freq='MS')
        results_horizons = {}
        for name, period in accuracy_periods.items():
            results_horizon = []
            prediction_for_acc = prediction.loc[period]
            df_copy = df.copy()
            if 'close' in name:
                prediction_for_acc = prediction_for_acc.resample('YS').sum()
                df_copy = df_copy.resample('YS').sum()
            accuracies = prediction_for_acc.apply(
                lambda x: compute_accuracy_scores(df_copy[x.name], x))
            for var, accuracy in accuracies.iteritems():
                results = {'model':model_name,
                           'params':str(param),
                           'split_date':begin,
                           'pred_period':pred_period,
                           'exog_vars': str(exog_cols),
                           'endog_vars': str(endog_cols),
                           'variable': var}
                accuracy_format = {k + '_' + name: v for k, v in accuracy.items()}
                results = {**results, **accuracy_format}
                results_horizon.append(results)
            results_horizons[name] = (results_horizon)
        for i in range(len(results_horizon)):
            results_accuracies.append({
                **results_horizons['first6'][i],
                **results_horizons['last12'][i],
                **results_horizons['first18'][i]})
                # **results_horizons['second_close'][i]})

    elif freq =='YS':
        accuracies = prediction.apply(
            lambda x: compute_accuracy_scores(df[x.name], x))
        for var, accuracy in accuracies.iteritems():
            results = {'model':model_name,
                       'params':str(param),
                       'split_date':begin,
                       'pred_period':pred_period,
                       'exog_vars': str(exog_cols),
                       'endog_vars': str(endog_cols),
                       'variable': var}
            results_accuracies.append({**results, **accuracy})

    return results_accuracies
  

def plot_prediction(observed, predicted, model_type, accuracy, var_predicted, model_params,
                    save_to=None, min_date=None, max_date=None, ticks='auto', ticks_freq=1, 
                    figsize=(9, 7), legend_out=True):
    '''
    Plot observed and predicted series.
    '''
    # Usando copias para no modificar originales
    observed_plot = observed.copy()
    predicted_plot = predicted.copy()

    # Convirtiendo a dataframe si prediction solo es una serie
    if isinstance(predicted_plot, pd.core.series.Series):
        predicted_plot = predicted_plot.to_frame()

    # Restringiendo a fechas especificadas
    if min_date:
        observed_plot = observed_plot.loc[observed_plot.index >= min_date]
        predicted_plot = predicted_plot.loc[predicted_plot.index >= min_date]

    if max_date:
        observed_plot = observed_plot.loc[observed_plot.index <= max_date]
        predicted_plot = predicted_plot.loc[predicted_plot.index <= max_date]

    # Imprimiendo descripción del modelo

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(observed_plot, color='k', )
    ax.plot(predicted_plot, linestyle='--')
    # Configurando xticks
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
    ## Configure Y ticks: 
    if observed.mean() < 1000:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
    else:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.grid()
    # Configurando leyenda. Primero va obserned y después predicciones con accuracy.
    rmse = [t[0] for t in accuracy]
    mae = [t[1] for t in accuracy]
    legend = ['observed']
    legend_pred = [col for col in predicted_plot.columns]
    legend_pred = [col + " RMSE: {0:,.3f}, MAE: {1:,.3f}".format(rmse[i], mae[i])\
                   for i, col in enumerate(legend_pred)]
    legend += legend_pred
    if legend_out:
        ax.legend(legend, bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        ax.legend(legend)

    #Título
    legible_var_name = " ".join(var_predicted.replace('mdp', 'mdp 2019').split('_')).upper()
    ax.set_title('Prediction of {}'.format(legible_var_name))

    if save_to:
        plt.savefig(save_to)
    plt.show()
    plt.close()


def get_start_and_end_dates(list_of_starts, duration):
    '''
    get a list of tuples with the begining of the prediction and the end
    of the prediction period.
    Inputs:
        list_of_starts : [dates]
        duration: int
    output:
        list of tuples
    '''
    rv = []
    for start in list_of_starts:
        begining = pd.to_datetime(start)
        end = begining + relativedelta(months=duration - 1)
        rv.append((begining, end))
    return rv








# def run_ml(model_name, all_models_params, outcome_var, global_params,
#            plot_extra, lags, outcome_var_tr=None, covars=None):
#     '''
#     Run Machine Learning regression.
#     '''
#     # Obtener los splits para hacer las predicciones
#     model_dict = {'DT': DecisionTreeRegressor,
#                   'RF': RandomForestRegressor,
#                   'GB': GradientBoostingRegressor}

#     model = model_dict[model_name]
#     model_params = all_models_params[model_name]
#     results_l = []
#     time_splits = pd.date_range(start=global_params['pred_start'],
#                                 end=global_params['pred_end'],
#                                 freq='12MS')

#     # Correr modelo para cada una de las especificaciones en model params
#     for param in model_params:
#         # Print Header
#         print_model_param(param, model_name)
        
#         # Crear DF en el que se van a poner todas las predicciones. Tanto para la variable
#         # transformada como para la variable en niveles.
#         predictions_tr = pd.DataFrame(index=outcome_var.index)
#         # Esta lista se usará para los labels del plot
#         predictions_tr_acc = []
#         predictions = pd.DataFrame(index=outcome_var.index)
#         # Esta lista se usará para los labels del plot
#         predictions_acc = []

#         # Creamos un DF con los lags
#         lagged_df = create_lagged_features(outcome_var_tr, lags)

#         for i, split_date in enumerate(time_splits):
#             # if i == len(time_splits) - 1:
#             #     break
#             #Iniciio de prediccion y final de predicción para obtener predicción
#             start = split_date
#             months = int(global_params['pred_period'].replace('MS', ''))
#             end = split_date + relativedelta(months=months)
#             initial_date = start - relativedelta(months=1)


#             # Creamos train, test y prediction
#             X_train = lagged_df.loc[lagged_df.index<start]
#             X_forecast = lagged_df.loc[start]
#             y_train = outcome_var_tr.loc[X_train.index]
#             y_test = outcome_var_tr[pd.date_range(start, end, freq='MS')]

#             # Creamos modelo y hacemos fit
#             regr = model(**param)
#             regr.fit(X_train, y_train)

#             # De los parametros globales, obtenemos periodos a predecir
#             steps = int(global_params['pred_period'].replace('MS', '')) + 1
#             # Llamamos función para predecir de manera recursiva
#             prediction_tr = recursivelly_predict(regr, X_forecast, steps)
#             # Asignamos el índice de la serie de predicción
#             prediction_tr.index = y_test.index
#             # Obtenemos resultados de precisión
#             dict_results_tr = get_dict_results(
#                 outcome_var_tr, prediction_tr, model_name, param, 
#                 split_date=split_date, pred_period=global_params['pred_period'],
#                 transformation=global_params['transformation'],
#                 variable=outcome_var.name)
#             # Incluimos RMSE MAE en la lista que sirve para los labels del plor.
#             predictions_tr_acc.append((dict_results_tr['rmse'], dict_results_tr['mae']))
#             # Incluimos los resultados en la lista de resultados
#             results_l.append(dict_results_tr)
#             # Obtenemos nombre de la predicción
#             pred_name = model_name + '_pred_' + str(i)
#             # Hacemos merge de la predicción con el DF de predicciones
#             predictions_tr = predictions_tr.merge(
#                 prediction_tr.rename(pred_name),
#                 left_index=True, right_index=True, how='outer')
#             # Obtener el valor de la variable en el momento previo a la transformaci´øn
#             # usando la fecha inmediata anterior.
#             initial_state = outcome_var[initial_date]
#             # Revertimos transformación y creamos variable de predicción en niveles
#             prediction = descriptive.revert_transformation(
#                 transformed=prediction_tr, 
#                 applied_transformation=global_params['transformation'],
#                 initial_value=initial_state,
#                 initial_date=initial_date)

#             # Append results to results list. If the transformation was a differece, 
#             # the reverted prediction has one overlap with the observed, and that needs
#             # to be removed before computing accuracy.
#             prediction_to_acc = prediction.loc[pd.date_range(start, end, freq='MS')]
#             dict_results = get_dict_results(
#                 outcome_var, prediction_to_acc, model_name, param,
#                 split_date=split_date, pred_period=global_params['pred_period'],
#                 transformation='levels', variable=outcome_var.name)

#             # Incluimos los resultados en la lista de resultados
#             results_l.append(dict_results)
#             # Hacemos merge de la predicción con el DF de predicciones
#             predictions = predictions.merge(
#                 prediction.rename(pred_name), left_index=True, right_index=True,
#                 how='outer')

#             # Obtener precisión de cada predicción
#             predictions_acc.append((dict_results['rmse'], dict_results['mae']))

#         # Plotting results

#         # return (predictions, predictions_tr)
#         graph_min_date = pd.to_datetime(global_params['pred_start']) - relativedelta(years=1)

#         plot_prediction(
#             outcome_var_tr, predictions_tr, model_name,
#             predictions_tr_acc,
#             global_params['outcome_col_transformed'],
#             param, legend_out=True, min_date=graph_min_date,
#             ticks='monthly', ticks_freq=2)

#         plot_prediction(
#             outcome_var, predictions, model_name,
#             predictions_acc,
#             global_params['outcome_col'],
#             param, legend_out=True, min_date=graph_min_date,
#             ticks='monthly', ticks_freq=2)

#     return results_l
 


# def print_model_param(model_param, model_name):
#     '''
#     Print model params
#     '''
#     description = model_name + ': '
#     for k, val in model_param.items():
#         if isinstance(val, pd.core.series.Series):
#             description += k + ' ' + k + ', '
#         else:
#             description += k + ' ' + str(val) + ', '
#     print(description)


# def prophet_make_future_dataframe(model_obj, pred_period):
#     '''
#     Return future Dataframe in Prophet format
#     '''
#     freq = pred_period[-2:]
#     periods = int(pred_period[:-2])
#     return model_obj.make_future_dataframe(periods=periods, freq=freq)


# def create_lagged_features(serie, lags):
#     '''
#     From a Pandas Series, make a DataFrame of lags.
#     Inputs:
#         outcome_var_tr: pd.Series
#         lags: int
#     Output:
#         DF
#     '''
#     # creamos DF vacio con el mismo índice que la serie
#     lagged_df = pd.DataFrame(index=serie.index)
#     # Crear cada uno de los lags en el DataFrame usando el método shift.
#     for lag in range(1, lags + 1):
#         lagged_df['lag_{}'.format(lag)] = serie.shift(lag)
#     # Nos quedamos con lo que no es NA
#     lagged_df = lagged_df.loc[lagged_df.notna().all(1)]
#     return lagged_df


# def recursivelly_predict(regr, X_forecast, steps):
#     '''
#     Recursivelly predicts y with the regr Machine learning.
#     inputs:
#         x_forecast: Pandas Series
#     output:
#         Pandas Series
#     '''
#     # Convertimos X_forecast en un numpy array, lo cual hará más facil el proceso recursivo. El reshape es necesario
#     # Para el predict
#     prediction_tr = []
#     X_forecast_rec = np.array(X_forecast).reshape(1, -1)
#     while steps > 0:
#         y_pred = regr.predict(X_forecast_rec)
#         prediction_tr.append(y_pred[0])
#         X_forecast_rec = np.insert(X_forecast_rec, 0, y_pred[0])
#         X_forecast_rec = X_forecast_rec[:-1]
#         X_forecast_rec = X_forecast_rec.reshape(1, -1)
#         steps -= 1
#     return pd.Series(prediction_tr) 

# def run_model_joint(model_name, all_models_params, outcome_var, global_params,
#                     plot_extra, outcome_var_tr=None, covars=None):
#     '''
#     '''
#     # Obtener los splits para hacer las predicciones
#     model_dict = {'ARIMA': ARIMA,
#                   'SARIMA': sm.tsa.statespace.SARIMAX,
#                   'ELASTICITY': ELASTICITY,
#                   'PROPHET': Prophet,
#                   'DT': DecisionTreeRegressor,
#                   'RF': RandomForestRegressor,
#                   'GB': GradientBoostingRegressor}

#     model = model_dict[model_name]
#     model_params = all_models_params[model_name]
#     results_l = []
#     time_splits = pd.date_range(start=global_params['pred_start'],
#                                 end=global_params['pred_end'],
#                                 freq='12MS')

#     # Correr modelo para cada una de las especificaciones en model params
#     for param in model_params:
#         # Print Header
#         print_model_param(param, model_name)
#         # Algunos modelos no se pueden estimar para toda la serie y necesitan 
#         # de training and testing. Por eso incluyo try en la siguiente parte
#         if model_name in ['ARIMA', 'SARIMA', 'ELASTICITY']:
#             if model_name == 'ELASTICITY':
#                 model_obj = model(outcome_var, gdp=covars, **param)
#             else:
#                 model_obj = model(outcome_var_tr, **param)
#             # Try por si el modelo no converge
#             try:
#                 results = model_obj.fit()
#             except:
#                 print('Could not fit {} for split {}'.format(model_name, 'GENERAL'))
#                 continue

#             if model_name == 'ELASTICITY':
#                 outcome_var_name = " ".join(global_params['outcome_col'].split('_'))
#                 if plot_extra:
#                     results.plot(outcome_var_name)

#         # Plot elasticity and growth rates del model mecánico.
#             else:
#                 fitted = results.fittedvalues
#                 rmse, mae, _, __ = compute_accuracy_scores(outcome_var_tr, fitted,
#                                                     False)
#         # Append results to results list
#                 results_l.append(
#                     get_dict_results(
#                         outcome_var_tr, fitted, model_name, param,
#                             transformation=global_params['transformation'],
#                             variable=outcome_var.name))

#         # Crear DF en el que se van a poner todas las predicciones. Tanto para la variable
#         # transformada como para la variable en niveles.
#         predictions_tr = pd.DataFrame(index=outcome_var.index)
#         predictions_tr_acc = []
#         predictions = pd.DataFrame(index=outcome_var.index)
#         predictions_acc = []

#         for i, split_date in enumerate(time_splits):
#             # if i == len(time_splits) - 1:
#             #     break
#             #Iniciio de prediccion y final de predicción para obtener predicción
#             start = split_date
#             months = int(global_params['pred_period'].replace('MS', ''))
#             end = split_date + relativedelta(months=months)
#             initial_date = start - relativedelta(months=1)

#             if model_name in ['ARIMA', 'SARIMA', 'PROPHET']:
#                 # Estimar modelo Prophet. Incluye crear un df especifico, crear el DataFrame del futuro
#                 # Y crear predicciones obteniendo yhat.
#                 if model_name == 'PROPHET':
#                     prophet_df = outcome_var_tr.loc[outcome_var.index < start]\
#                         .to_frame().reset_index()
#                     prophet_df = prophet_df.rename(columns={
#                         'fecha':'ds', global_params['outcome_col']: 'y'})
#                     model_obj = Prophet(**param)
#                     model_obj.fit(prophet_df)
#                     future = prophet_make_future_dataframe(model_obj, global_params['pred_period'])
#                     prediction_tr = model_obj.predict(future)
#                     if plot_extra:
#                         model_obj.plot(prediction_tr)
#                     prediction_tr.set_index(prediction_tr['ds'], inplace=True)
#                     prediction_tr = prediction_tr.loc\
#                         [pd.date_range(initial_date, end, freq='MS'),
#                         'yhat']
#                 elif model_name in ['ARIMA', 'SARIMA']:
#                     model_obj = model(
#                         outcome_var_tr.loc[outcome_var_tr.index < start], **param)
#                     try:    
#                         results = model_obj.fit()
#                     except:
#                         print('Could not fit {} for split {}'.format(model_name, split_date))
#                         continue
#                     prediction_tr = results.predict(start=start, end=end)
#             #append results to results list

#                 prediction_tr_to_acc = \
#                     prediction_tr.loc[pd.date_range(start, end, freq='MS')]

#                 dict_results_tr = get_dict_results(
#                     outcome_var_tr, prediction_tr_to_acc, model_name, param, 
#                     split_date=split_date, pred_period=global_params['pred_period'],
#                     transformation=global_params['transformation'],
#                     variable=outcome_var.name)

#                 predictions_tr_acc.append((dict_results_tr['rmse'], dict_results_tr['mae']))

#                 results_l.append(dict_results_tr)
#                 predictions_tr = predictions_tr.merge(
#                     prediction_tr.rename(model_name + '_pred_' + str(i)),
#                     left_index=True, right_index=True, how='outer')
#             # Obtener el valor de la variable en el momento previo a la transformaci´øn
#             # usando la fecha inmediata anterior.
#                 initial_state = outcome_var[initial_date]
#                 prediction = descriptive.revert_transformation(
#                     transformed=prediction_tr, 
#                     applied_transformation=global_params['transformation'],
#                     initial_value=initial_state,
#                     initial_date=initial_date)
#                 pred_name = model_name + '_pred_' + str(i)

#             elif model_name == 'ELASTICITY':
#                 prediction = results.predict(start=start, end=end)
#                 pred_name = model_name + '_pred_' + str(i) + ' ELAST {0:.2f}'\
#                             .format(results.elasticity_used)
            
#             # Append results to results list. If the transformation was a differece, 
#             # the reverted prediction has one overlap with the observed, and that needs
#             # to be removed before computing accuracy.
#             prediction_to_acc = prediction.loc[pd.date_range(start, end, freq='MS')]
#             dict_results = get_dict_results(
#                 outcome_var, prediction_to_acc, model_name, param,
#                 split_date=split_date, pred_period=global_params['pred_period'],
#                 transformation='levels', variable=outcome_var.name)
#             results_l.append(dict_results)

#             predictions = predictions.merge(
#                 prediction.rename(pred_name), left_index=True, right_index=True,
#                 how='outer')

#             # Obtener precisión de cada predicción
#             predictions_acc.append((dict_results['rmse'], dict_results['mae']))

#         # Plotting results

#         # return (predictions, predictions_tr)
#         graph_min_date = pd.to_datetime(global_params['pred_start']) - relativedelta(years=1)
#         if not model_name == 'ELASTICITY':
#             plot_prediction(
#                 outcome_var_tr, predictions_tr, model_name,
#                 predictions_tr_acc,
#                 global_params['outcome_col_transformed'],
#                 param, legend_out=True, min_date=graph_min_date,
#                 ticks='monthly', ticks_freq=2)
#         plot_prediction(
#             outcome_var, predictions, model_name,
#             predictions_acc,
#             global_params['outcome_col'],
#             param, legend_out=True, min_date=graph_min_date,
#             ticks='monthly', ticks_freq=2)

#     return results_l





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
from dateutil.relativedelta import relativedelta
import descriptive
import matplotlib as mpl
from fbprophet import Prophet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from mechanical import ELASTICITY

def run_model_joint(model_name, all_models_params, outcome_var, global_params,
                    plot_extra, outcome_var_tr=None, covars=None):
    '''
    '''
    # Obtener los splits para hacer las predicciones
    model_dict = {'ARIMA': ARIMA,
                  'SARIMA': sm.tsa.statespace.SARIMAX,
                  'ELASTICITY': ELASTICITY,
                  'PROPHET': Prophet,
                  'DT': DecisionTreeRegressor,
                  'RF': RandomForestRegressor,
                  'GB': GradientBoostingRegressor}

    model = model_dict[model_name]
    model_params = all_models_params[model_name]
    results_l = []
    time_splits = pd.date_range(start=global_params['pred_start'],
                                end=global_params['pred_end'],
                                freq=global_params['pred_period'])

    # Correr modelo para cada una de las especificaciones en model params
    for param in model_params:
        # Print Header
        print_model_param(param, model_name)
        # Algunos modelos no se pueden estimar para toda la serie y necesitan 
        # de training and testing. Por eso incluyo try en la siguiente parte
        if model_name in ['ARIMA', 'SARIMA', 'ELASTICITY']:
            if model_name == 'ELASTICITY':
                model_obj = model(outcome_var, gdp=covars, **param)
            else:
                model_obj = model(outcome_var_tr, **param)
            # Try por si el modelo no converge
            try:
                results = model_obj.fit()
            except:
                print('Could not fit {} for split {}'.format(model_name, 'GENERAL'))
                continue

            if model_name == 'ELASTICITY':
                outcome_var_name = " ".join(global_params['outcome_col'].split('_'))
                if plot_extra:
                    results.plot(outcome_var_name)

        # Plot elasticity and growth rates del model mecánico.
            else:
                fitted = results.fittedvalues
                rmse, mae, _, __ = compute_accuracy_scores(outcome_var_tr, fitted,
                                                    False)
        # Append results to results list
                results_l.append(
                    get_dict_results(
                        outcome_var_tr, fitted, model_name, param,
                            transformation=global_params['transformation']))

        # Crear DF en el que se van a poner todas las predicciones. Tanto para la variable
        # transformada como para la variable en niveles.
        predictions_tr = pd.DataFrame(index=outcome_var.index)
        predictions_tr_acc = []
        predictions = pd.DataFrame(index=outcome_var.index)
        predictions_acc = []

        for i, split_date in enumerate(time_splits):
            if i == len(time_splits) - 1:
                break
            #Iniciio de prediccion y final de predicción para obtener predicción
            start = split_date
            end = time_splits[i + 1] - relativedelta(months=1)
            initial_date = start - relativedelta(months=1)

            if model_name in ['ARIMA', 'SARIMA', 'PROPHET']:
                # Estimar modelo Prophet. Incluye crear un df especifico, crear el DataFrame del futuro
                # Y crear predicciones obteniendo yhat.
                if model_name == 'PROPHET':
                    prophet_df = outcome_var_tr.loc[outcome_var.index < start]\
                        .to_frame().reset_index()
                    prophet_df = prophet_df.rename(columns={
                        'fecha':'ds', global_params['outcome_col_transformed']: 'y'})
                    model_obj = Prophet(**param)
                    model_obj.fit(prophet_df)
                    future = prophet_make_future_dataframe(model_obj, global_params['pred_period'])
                    prediction_tr = model_obj.predict(future)
                    if plot_extra:
                        model_obj.plot(prediction_tr)
                    prediction_tr.set_index(prediction_tr['ds'], inplace=True)
                    prediction_tr = prediction_tr.loc\
                        [pd.date_range(initial_date, end, freq='MS'),
                        'yhat']
                elif model_name in ['ARIMA', 'SARIMA']:
                    model_obj = model(
                        outcome_var_tr.loc[outcome_var_tr.index < start], **param)
                    try:    
                        results = model_obj.fit()
                    except:
                        print('Could not fit {} for split {}'.format(model_name, split_date))
                        continue
                    prediction_tr = results.predict(start=start, end=end)
            #append results to results list

                prediction_tr_to_acc = \
                    prediction_tr.loc[pd.date_range(start, end, freq='MS')]

                dict_results_tr = get_dict_results(
                    outcome_var_tr, prediction_tr_to_acc, model_name, param, 
                    split_date=split_date, pred_period=global_params['pred_period'],
                    transformation=global_params['transformation'])

                predictions_tr_acc.append((dict_results_tr['rmse'], dict_results_tr['mae']))

                results_l.append(dict_results_tr)
                predictions_tr = predictions_tr.merge(
                    prediction_tr.rename(model_name + '_pred_' + str(i)),
                    left_index=True, right_index=True, how='outer')
            # Obtener el valor de la variable en el momento previo a la transformaci´øn
            # usando la fecha inmediata anterior.
                initial_state = outcome_var[initial_date]
                prediction = descriptive.revert_transformation(
                    transformed=prediction_tr, 
                    applied_transformation=global_params['transformation'],
                    initial_value=initial_state,
                    initial_date=initial_date)
                pred_name = model_name + '_pred_' + str(i)

            elif model_name == 'ELASTICITY':
                prediction = results.predict(start=start, end=end)
                pred_name = model_name + '_pred_' + str(i) + ' ELAST {0:.2f}'\
                            .format(results.elasticity_used)
            
            # Append results to results list. If the transformation was a differece, 
            # the reverted prediction has one overlap with the observed, and that needs
            # to be removed before computing accuracy.
            prediction_to_acc = prediction.loc[pd.date_range(start, end, freq='MS')]
            dict_results = get_dict_results(
                outcome_var, prediction_to_acc, model_name, param,
                split_date=split_date, pred_period=global_params['pred_period'],
                transformation='levels')
            results_l.append(dict_results)

            predictions = predictions.merge(
                prediction.rename(pred_name), left_index=True, right_index=True,
                how='outer')

            # Obtener precisión de cada predicción
            predictions_acc.append((dict_results['rmse'], dict_results['mae']))

        # Plotting results

        # return (predictions, predictions_tr)
        graph_min_date = pd.to_datetime(global_params['pred_start']) - relativedelta(years=1)
        if not model_name == 'ELASTICITY':
            plot_prediction(
                outcome_var_tr, predictions_tr, model_name,
                predictions_tr_acc,
                global_params['outcome_col_transformed'],
                param, legend_out=True, min_date=graph_min_date,
                ticks='monthly', ticks_freq=2)
        plot_prediction(
            outcome_var, predictions, model_name,
            predictions_acc,
            global_params['outcome_col'],
            param, legend_out=True, min_date=graph_min_date,
            ticks='monthly', ticks_freq=2)

    return results_l


def run_ml(model_name, all_models_params, outcome_var, global_params,
           plot_extra, lags, outcome_var_tr=None, covars=None):
    '''
    Run Machine Learning regression.
    '''
    # Obtener los splits para hacer las predicciones
    model_dict = {'DT': DecisionTreeRegressor,
                  'RF': RandomForestRegressor,
                  'GB': GradientBoostingRegressor}

    model = model_dict[model_name]
    model_params = all_models_params[model_name]
    results_l = []
    time_splits = pd.date_range(start=global_params['pred_start'],
                                end=global_params['pred_end'],
                                freq=global_params['pred_period'])

    # Correr modelo para cada una de las especificaciones en model params
    for param in model_params:
        # Print Header
        print_model_param(param, model_name)
        
        # Crear DF en el que se van a poner todas las predicciones. Tanto para la variable
        # transformada como para la variable en niveles.
        predictions_tr = pd.DataFrame(index=outcome_var.index)
        # Esta lista se usará para los labels del plot
        predictions_tr_acc = []
        predictions = pd.DataFrame(index=outcome_var.index)
        # Esta lista se usará para los labels del plot
        predictions_acc = []

        # Creamos un DF con los lags
        lagged_df = create_lagged_features(outcome_var_tr, lags)

        for i, split_date in enumerate(time_splits):
            # No correr para último split
            if i == len(time_splits) - 1:
                break
            #Iniciio de prediccion y final de predicción para obtener predicción
            start = split_date
            end = time_splits[i + 1] - relativedelta(months=1)
            # Para revertir transformación
            initial_date = start - relativedelta(months=1)

            # Creamos train, test y prediction
            X_train = lagged_df.loc[lagged_df.index<start]
            X_forecast = lagged_df.loc[start]
            y_train = outcome_var_tr.loc[X_train.index]
            y_test = outcome_var_tr[pd.date_range(start, end, freq='MS')]

            # Creamos modelo y hacemos fit
            regr = model(**param)
            regr.fit(X_train, y_train)

            # De los parametros globales, obtenemos periodos a predecir
            steps = int(global_params['pred_period'].replace('MS', ''))
            # Llamamos función para predecir de manera recursiva
            prediction_tr = recursivelly_predict(regr, X_forecast, steps)
            # Asignamos el índice de la serie de predicción
            prediction_tr.index = y_test.index
            # Obtenemos resultados de precisión
            dict_results_tr = get_dict_results(
                outcome_var_tr, prediction_tr, model_name, param, 
                split_date=split_date, pred_period=global_params['pred_period'],
                transformation=global_params['transformation'])
            # Incluimos RMSE MAE en la lista que sirve para los labels del plor.
            predictions_tr_acc.append((dict_results_tr['rmse'], dict_results_tr['mae']))
            # Incluimos los resultados en la lista de resultados
            results_l.append(dict_results_tr)
            # Obtenemos nombre de la predicción
            pred_name = model_name + '_pred_' + str(i)
            # Hacemos merge de la predicción con el DF de predicciones
            predictions_tr = predictions_tr.merge(
                prediction_tr.rename(pred_name),
                left_index=True, right_index=True, how='outer')
            # Obtener el valor de la variable en el momento previo a la transformaci´øn
            # usando la fecha inmediata anterior.
            initial_state = outcome_var[initial_date]
            # Revertimos transformación y creamos variable de predicción en niveles
            prediction = descriptive.revert_transformation(
                transformed=prediction_tr, 
                applied_transformation=global_params['transformation'],
                initial_value=initial_state,
                initial_date=initial_date)

            # Append results to results list. If the transformation was a differece, 
            # the reverted prediction has one overlap with the observed, and that needs
            # to be removed before computing accuracy.
            prediction_to_acc = prediction.loc[pd.date_range(start, end, freq='MS')]
            dict_results = get_dict_results(
                outcome_var, prediction_to_acc, model_name, param,
                split_date=split_date, pred_period=global_params['pred_period'],
                transformation='levels')

            # Incluimos los resultados en la lista de resultados
            results_l.append(dict_results)
            # Hacemos merge de la predicción con el DF de predicciones
            predictions = predictions.merge(
                prediction.rename(pred_name), left_index=True, right_index=True,
                how='outer')

            # Obtener precisión de cada predicción
            predictions_acc.append((dict_results['rmse'], dict_results['mae']))

        # Plotting results

        # return (predictions, predictions_tr)
        graph_min_date = pd.to_datetime(global_params['pred_start']) - relativedelta(years=1)

        plot_prediction(
            outcome_var_tr, predictions_tr, model_name,
            predictions_tr_acc,
            global_params['outcome_col_transformed'],
            param, legend_out=True, min_date=graph_min_date,
            ticks='monthly', ticks_freq=2)

        plot_prediction(
            outcome_var, predictions, model_name,
            predictions_acc,
            global_params['outcome_col'],
            param, legend_out=True, min_date=graph_min_date,
            ticks='monthly', ticks_freq=2)

    return results_l

# def arima(df, params=None, outcome_var=None):
#     '''
#     Create ARIMA model and fit to data
#     Inputs:
#         df: DF
#         outcome_var: str
#         params: dictionary
#     Output:
#         ARIMA Results Class
#     '''
#     if isinstance(df, pd.core.series.Series):
#         objective_ts = df
#     else:
#         objective_ts = df[outcome_var]
#     arima_model = ARIMA(objective_ts, **params)
#     results_arima = arima_model.fit()
#     return results_arima

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


def get_dict_results(outcome_var, prediction, model, param, transformation,
                     split_date=None, pred_period=None, dynamic=None):
    '''
    '''
    rmse, mae, mape, forecast_biass = compute_accuracy_scores(
        outcome_var, prediction, False)
    results = {}
    results['model'] = model
    results['param'] = param
    results['split_date'] = split_date
    results['pred_period'] = pred_period
    results['dynamic'] = dynamic
    results['transformation'] = transformation
    results['rmse'] = rmse
    results['mae'] = mae
    results['mape'] = mape
    results['forecast_biass'] = forecast_biass

    return results

def print_model_param(model_param, model_name):
    '''
    Print model params
    '''
    description = model_name + ': '
    for k, val in model_param.items():
        if isinstance(val, pd.core.series.Series):
            description += k + ' ' + k + ', '
        else:
            description += k + ' ' + str(val) + ', '
    print(description)


def prophet_make_future_dataframe(model_obj, pred_period):
    '''
    Return future Dataframe in Prophet format
    '''
    freq = pred_period[-2:]
    periods = int(pred_period[:-2])
    return model_obj.make_future_dataframe(periods=periods, freq=freq)


def create_lagged_features(serie, lags):
    '''
    From a Pandas Series, make a DataFrame of lags.
    Inputs:
        outcome_var_tr: pd.Series
        lags: int
    Output:
        DF
    '''
    # creamos DF vacio con el mismo índice que la serie
    lagged_df = pd.DataFrame(index=serie.index)
    # Crear cada uno de los lags en el DataFrame usando el método shift.
    for lag in range(1, lags + 1):
        lagged_df['lag_{}'.format(lag)] = serie.shift(lag)
    # Nos quedamos con lo que no es NA
    lagged_df = lagged_df.loc[lagged_df.notna().all(1)]
    return lagged_df

def recursivelly_predict(regr, X_forecast, steps):
    '''
    Recursivelly predicts y with the regr Machine learning.
    inputs:
        x_forecast: Pandas Series
    output:
        Pandas Series
    '''
    # Convertimos X_forecast en un numpy array, lo cual hará más facil el proceso recursivo. El reshape es necesario
    # Para el predict
    prediction_tr = []
    X_forecast_rec = np.array(X_forecast).reshape(1, -1)
    while steps > 0:
        y_pred = regr.predict(X_forecast_rec)
        prediction_tr.append(y_pred[0])
        X_forecast_rec = np.insert(X_forecast_rec, 0, y_pred[0])
        X_forecast_rec = X_forecast_rec[:-1]
        X_forecast_rec = X_forecast_rec.reshape(1, -1)
        steps -= 1
    return pd.Series(prediction_tr) 
    

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

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

def run_model(model_name, model_params, outcome_var, outcome_var_tr, global_params, dynamic=True):
    '''
    '''
    # Obtener los splits para hacer las predicciones
    model_dict = {'ARIMA': ARIMA,
                  'SARIMA': sm.tsa.statespace.SARIMAX}
    model = model_dict[model_name]
    results_l = []
    time_splits = pd.date_range(start=global_params['pred_start'],
                                end=global_params['pred_end'],
                                freq=global_params['pred_period'])

    # Correr modelo para cada una de las especificaciones en model params
    for param in model_params:
        # Print Header
        description = model_name + ': '
        for k, val in param.items():
            description += k + ' ' + str(val) + ', '
        print(description)
        model_obj = model(outcome_var_tr, **param)
        results = model_obj.fit()
        fitted = results.fittedvalues
        rmse, mae = compute_accuracy_scores(outcome_var_tr, fitted,
                                           False)
        #append results to results list
        results_l.append(
        get_dict_results(outcome_var_tr, fitted, model_name, param,
                            transformation = global_params['transformation']))
        plot_prediction(outcome_var_tr, fitted.rename('pred'), model_name, [(rmse, mae)],
                        global_params['outcome_col_transformed'],
                        param, legend_out=True, ticks= 'yearly')



        # Crear DF en el que se van a poner todas las predicciones. Tanto para la variable
        # transformada como para la variable en niveles.
        predictions_transformed = pd.DataFrame(index=outcome_var_tr.index)
        predictions_transformed_accuracy = []
        predictions = pd.DataFrame(index=outcome_var.index)
        predictions_accuracy = []

        for i, split_date in enumerate(time_splits):
            if i == len(time_splits) - 1:
                break
            #Iniciio de prediccion y final de predicción para obtener predicción
            start = split_date
            end = time_splits[i + 1] - relativedelta(months=1)
            model_obj = model(
                outcome_var_tr.loc[outcome_var_tr.index < start], **param)
            results = model_obj.fit()
            prediction_transformed = results.predict(start=start, end=end,
                                                         dynamic=dynamic)
            #append results to results list
            results_l.append(
            get_dict_results(outcome_var_tr, prediction_transformed, model_name,
                                param, split_date=split_date,
                                pred_period=global_params['pred_period'],
                                dynamic=dynamic, 
                                transformation = global_params['transformation']))

            predictions_transformed = predictions_transformed.merge(
                prediction_transformed.rename(model_name + '_pred_' + str(i)),
                left_index=True, right_index=True, how='outer')
            # Fecha inmediata anterior a inicio de predicción. Sirve para obtener el valor de la
            # variable en ese momento y revertir transformación
            initial_date = start - relativedelta(months=1)
            initial_state = outcome_var[initial_date]
            prediction = descriptive.revert_transformation(prediction_transformed,
                                                           global_params['transformation'],
                                                           initial_state, initial_date)
            #append results to results list
            results_l.append(
            get_dict_results(outcome_var, prediction, model_name, param,
                                split_date=split_date,
                                pred_period=global_params['pred_period'],
                                dynamic=dynamic,
                                transformation='levels'))

            predictions = predictions.merge(
                prediction.rename(model_name + '_pred_' + str(i)),
                left_index=True, right_index=True, how='outer')

            # Obtener precisión de cada predicción
            predictions_transformed_accuracy.append(
                compute_accuracy_scores(outcome_var_tr, prediction_transformed,
                                        False))
            predictions_accuracy.append(
                compute_accuracy_scores(outcome_var, prediction, False))

        # Plotting results
        graph_min_date = pd.to_datetime(global_params['pred_start']) - relativedelta(years=1)
        plot_prediction(
            outcome_var_tr, predictions_transformed, model_name,
            predictions_transformed_accuracy,
            global_params['outcome_col_transformed'],
            param, legend_out=True, min_date=graph_min_date,
            ticks='monthly', ticks_freq=2)
        plot_prediction(
            outcome_var, predictions, model_name,
            predictions_accuracy,
            global_params['outcome_col'],
            param, legend_out=True, min_date=graph_min_date,
            ticks='monthly', ticks_freq=2)

    return results_l


def arima(df, params=None, outcome_var=None):
    '''
    Create ARIMA model and fit to data
    Inputs:
        df: DF
        outcome_var: str
        params: dictionary
    Output:
        ARIMA Results Class
    '''
    if isinstance(df, pd.core.series.Series):
        objective_ts = df
    else:
        objective_ts = df[outcome_var]
    arima_model = ARIMA(objective_ts, **params)
    results_arima = arima_model.fit()
    return results_arima

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
    if output_dict:
        return accuracy_scores
    else:
        return (accuracy_scores['rmse'], accuracy_scores['mae'])


def get_dict_results(outcome_var, prediction, model, param, transformation,
                     split_date=None, pred_period=None, dynamic=None):
    '''
    '''
    rmse, mae = compute_accuracy_scores(outcome_var, prediction, False)
    results = {}
    results['model'] = model
    results['param'] = param
    results['split_date'] = split_date
    results['pred_period'] = pred_period
    results['dynamic'] = dynamic
    results['transformation'] = transformation
    results['rmse'] = rmse
    results['mae'] = mae

    return results


    
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
    ax.grid()
    # Configurando leyenda. Primero va obserned y después predicciones con accuracy.
    legend = ['observed']
    legend_pred = [col for col in predicted_plot.columns]
    legend_pred = [col + ' RMSE: {0:.3f}, MAE: {0:.3f}'.format(accuracy[i][0], accuracy[i][1])\
                    for i, col in enumerate(legend_pred)]
    legend += legend_pred
    if legend_out:
        ax.legend(legend, bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        ax.legend(legend)

    #Título
    legible_var_name = " ".join(var_predicted.split('_')).upper()
    ax.set_title('Prediction of {}'.format(legible_var_name))

    if save_to:
        plt.savefig(save_to)
    plt.show()
    plt.close()

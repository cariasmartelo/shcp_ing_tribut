'''
SHCP UPIT Forecasting Public Revenue
Models
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima_model import ARIMA, ARMA
from dateutil.relativedelta import relativedelta
import descriptive

def run_models(model, model_params, outcome_var, outcome_var_tr, global_params, dynamic=True):
    '''
    '''
    time_splits = pd.date_range(start=global_params['pred_start'],
                                end=global_params['pred_end'],
                                freq=global_params['pred_period'])
    for param in model_params:
        try:
            arima_model = arima(outcome_var_tr, param)
            fitted = arima_model.fittedvalues
            plot_prediction(outcome_var_tr, fitted, model,
                            global_params['outcome_col'] + '_' + global_params['transformation'],
                            param, {'figsize': (7, 7)})
        except:
            continue
        predictions_transformed = pd.DataFrame(index=outcome_var_tr.index)
        predictions = pd.DataFrame(index=outcome_var.index)

        for i, split_date in enumerate(time_splits):
            if i == len(time_splits) - 1:
                break
            start = split_date
            end = time_splits[i + 1] - relativedelta(months=1)
            prediction_transformed = arima_model.predict(start=start, end=end, dynamic=dynamic)
            initial_date = start - relativedelta(months=1)

            initial_state = outcome_var[initial_date]
            prediction = descriptive.revert_transformation(prediction_transformed,
                                                           global_params['transformation'],
                                                           initial_state, initial_date)

            accuracy_scores = compute_accuracy_scores(outcome_var_tr, prediction_transformed)
            rmse = round(accuracy_scores['rmse'], 3)
            mae = round(accuracy_scores['mae'], 3)
            prediction_transformed.name = 'pred_' + str(i) + '_' + 'RMSE: {} '.format(rmse)\
                + 'MAE: {} '.format(mae)
            predictions_transformed = predictions_transformed.merge(prediction_transformed,
                                                                    left_index=True,
                                                                    right_index=True,
                                                                    how='outer')

            accuracy_scores = compute_accuracy_scores(outcome_var, prediction)
            rmse = round(accuracy_scores['rmse'], 3)
            mae = round(accuracy_scores['mae'], 3)
            prediction.name = 'pred_' + str(i) + '_' + 'RMSE: {} '.format(rmse)\
                + 'MAE: {} '.format(mae)
            predictions = predictions.merge(prediction, left_index=True, right_index=True,
                                            how='outer')
        graph_min_date = pd.to_datetime(global_params['pred_start']) - relativedelta(years=1)

        plot_prediction(outcome_var_tr, predictions_transformed, model,
                        global_params['outcome_col'] + '_' + global_params['transformation'],
                        param, plot_params={'figsize': (7, 7), 'legend_out':True,
                                                          'min_date': graph_min_date,
                                                          'ticks': 'yearly'})
        plot_prediction(outcome_var, predictions, model,
                        global_params['outcome_col'],
                        param, plot_params={'figsize': (7, 7), 'legend_out':True, 
                                                   'min_date': graph_min_date,
                                                   'ticks': 'yearly'})


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

def compute_accuracy_scores(observed, predicted):
    '''
    From the predicted series and the real series, compute mean root squared error and
    mean absolute error.
    '''
    accuracy_scores = {}
    resid = observed - predicted
    rss = resid ** 2
    accuracy_scores['rmse'] = np.sqrt(rss.mean())
    accuracy_scores['mae'] = abs(resid).mean()
    return accuracy_scores

def plot_prediction(observed, predicted, model_type, var_predicted, model_params, plot_params=None):
    '''
    Plot observed and predicted series.
    '''
    if not plot_params:
        plot_params = {}
    observed.name = 'observed'
    description = model_type + ': '
    for k, val in model_params.items():
         description += k + ': ' + str(val) + ', '
    print(description)
    legible_var_name = " ".join(var_predicted.split('_')).upper()
    if isinstance(predicted, pd.core.series.Series):
        predicted.name = 'predicted'
        accuracy_scores = compute_accuracy_scores(observed, predicted)
        rmse = round(accuracy_scores['rmse'], 3)
        mae = round(accuracy_scores['mae'], 3)
        prediction_start = predicted.index.min()
        prediction_end = predicted.index.max()
        to_plot = observed.to_frame()
        to_plot = to_plot.merge(predicted, how='outer', left_index=True, right_index=True)
        descriptive.plot_series(to_plot, title='Prediction of {}'.format(legible_var_name),
                                subtitle = ('From {} to {} \nRMSE: {}, MAE: {}'   
                                           .format(prediction_start.strftime('%Y-%m-%d'),
                                                   prediction_end.strftime('%y-%m-%d'),
                                                   rmse, mae)),
                                **plot_params)
    else:
        to_plot = observed.to_frame()
        to_plot = to_plot.merge(predicted, how='outer', left_index=True, right_index=True)
        descriptive.plot_series(to_plot, title='Prediction of {}'.format(legible_var_name),
                                **plot_params)




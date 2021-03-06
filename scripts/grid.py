import itertools
    
econometric_params = {
    'small': {
        'p': range(3, 4),
        'q': range(3, 4),
        'P': range(1, 2),
        'Q': range(1, 2),
        'sea_p': range(1, 2),
        'sea_q': range(1, 2),
        'sea_s': [12]},
    'big': {
        'p': range(1, 6),
        'q': range(1, 6),
        'P': range(1, 4),
        'Q': range(1, 4),
        'sea_p': range(0, 2),
        'sea_q': range(0, 2),
        'sea_s': [12]},
    'medium': {
        'p': range(2, 5),
        'q': range(2, 5),
        'P': range(2, 4),
        'Q': range(2, 4),
        'sea_p': range(0, 2),
        'sea_q': range(0, 2),
        'sea_s': [12]}
}

econometric_models  = {'small': {
    'ARIMA': {'order': [(x[0], 0, x[1]) for x in\
                list(itertools.product(econometric_params['small']['p'],
                                       econometric_params['small']['q']))]},
    'SARIMA': {'order': [(x[0], 0, x[1]) for x in\
                list(itertools.product(econometric_params['small']['P'],
                                       econometric_params['small']['Q']))],
               'seasonal_order': [(x[0], 0, x[1], x[2]) for x in\
                list(itertools.product(econometric_params['small']['sea_p'],
                                       econometric_params['small']['sea_q'],
                                       econometric_params['small']['sea_s']))],
               'enforce_stationarity': [False],
               'enforce_invertibility': [False]},
    'ELASTICITY': {'lag_window': [3, 6, 12], 'elasticity': [None, 1.3, 2]},
    'PROPHET': {'seasonality_mode':['additive', 'multiplicative'], 
                'weekly_seasonality': [False], 
                'daily_seasonality': [False]},
    'VAR': {'maxlags': [12], 'ic': ['aic', 'bic', 'fpe'], 'trend': ['c', 'ct']}},
                      'medium': {
    'ARIMA': {'order': [(x[0], 0, x[1]) for x in\
                list(itertools.product(econometric_params['medium']['p'],
                                       econometric_params['medium']['q']))]},
    'SARIMA': {'order': [(x[0], 0, x[1]) for x in\
                list(itertools.product(econometric_params['medium']['P'],
                                       econometric_params['medium']['Q']))],
               'seasonal_order': [(x[0], 0, x[1], x[2]) for x in\
                list(itertools.product(econometric_params['medium']['sea_p'],
                                       econometric_params['medium']['sea_q'],
                                       econometric_params['medium']['sea_s']))],
               'enforce_stationarity': [False],
               'enforce_invertibility': [False]},
    'ELASTICITY': {'lag_window': [3, 6, 12], 'elasticity': [None, 1.3, 2]},
    'PROPHET': {'seasonality_mode':['additive', 'multiplicative'], 
                'weekly_seasonality': [False], 
                'daily_seasonality': [False]},
    'VAR': {'maxlags': [12], 'ic': ['aic', 'bic', 'fpe'], 'trend': ['c', 'ct']}},    
                       'big': {
    'ARIMA': {'order': [(x[0], 0, x[1]) for x in\
                list(itertools.product(econometric_params['big']['p'],
                                       econometric_params['big']['q']))]},
    'SARIMA': {'order': [(x[0], 0, x[1]) for x in\
            list(itertools.product(econometric_params['big']['P'],
                                   econometric_params['big']['Q']))],
           'seasonal_order': [(x[0], 0, x[1], x[2]) for x in\
                list(itertools.product(econometric_params['big']['sea_p'],
                                       econometric_params['big']['sea_q'],
                                       econometric_params['big']['sea_s']))],
            'enforce_stationarity': [False],
            'enforce_invertibility': [False]},
            'ELASTICITY': {'lag_window': [3, 6, 12], 'elasticity': [None, 1.3, 2]},
            'PROPHET': {'seasonality_mode':['additive', 'multiplicative'], 
            'weekly_seasonality': [False], 
            'daily_seasonality': [False]},
            'VAR': {'maxlags': [12], 'ic': ['aic', 'bic', 'fpe'], 'trend': ['c', 'ct']}}}

ml_models = {'small': {
            'DT': {'criterion': ['mse'], 
                   'max_depth': [10], 
                   'max_features': [None],
                   'min_samples_split': [2],
                   'random_state': [1234]},
            'RF': {'n_estimators': [10],
                   'criterion': ['mse'],
                   'max_depth': [5], 
                   'max_features': ['sqrt'],
                   'min_samples_split': [2], 
                   'n_jobs':[-1],
                   'random_state': [1234]},
            'GB': {'n_estimators': [1], 
                   'learning_rate' : [0.1],
                   'subsample' : [0.1], 
                   'max_depth': [1],
                   'random_state': [1234]}},
          'big':{
        'DT': {'criterion': ['mse', 'friedman_mse', 'mae'], 
               'max_depth': [1,5,10,20,50,100], 
               'max_features': [None,'sqrt','log2'],
               'min_samples_split': [2,5,10],
               'random_state': [1234]},
        'RF': {'n_estimators': [1, 10, 100, 1000],
               'criterion': ['mse', 'mae'],
               'max_depth': [5,50], 
               'max_features': ['sqrt','log2'],
               'min_samples_split': [2,10], 
               'n_jobs':[-1],
               'random_state': [1234]},
        'GB': {'n_estimators': [1, 50, 100, 1000], 
               'learning_rate' : [0.1, 0.5],
               'subsample' : [0.1, 0.5, 1.0], 
               'max_depth': [1, 5, 10],
               'random_state': [1234]}},
          'medium': {
        'DT': {'criterion': ['friedman_mse', 'mae'], 
               'max_depth': [50, 100], 
               'max_features': ['log2'],
               'min_samples_split': [5],
               'random_state': [1234]},
        'RF': {'n_estimators': [100, 1000],
               'criterion': ['mse', 'mae'],
               'max_depth': [5,50], 
               'max_features': ['sqrt','log2'],
               'min_samples_split': [2,10], 
               'n_jobs':[-1],
               'random_state': [1234]},
        'GB': {'n_estimators': [100, 1000], 
               'learning_rate' : [0.1, 0.5],
               'subsample' : [0.1, 0.5, 1.0], 
               'max_depth': [5, 10],
               'random_state': [1234]}},
          'custom': {
        'RF': {'n_estimators': [1000],
               'criterion': ['mae'],
               'max_depth': [50], 
               'max_features': ['log2'],
               'min_samples_split': [10], 
               'n_jobs':[-1],
               'random_state': [1234]},
        'GB': {'n_estimators': [1000], 
               'learning_rate' : [0.1],
               'subsample' : [0.5], 
               'max_depth': [5],
               'random_state': [1234]}}
        }
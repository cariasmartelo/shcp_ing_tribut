'''
SHCP UPIT Forecasting Public Revenue
Mechanical model
'''
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

class ELASTICITY:
    '''
    Class to represent the predictive method that uses elasticity to GDP to forcast.
    Is uses two Pandas Series, the gdp and the series to predict. It computes growth
    rate of each, and then is calculates the elasticity. To predict, the class needs
    a lag_window, which is the periods over which to average elasticity before applying
    it to the forecast. For example, if lag window is 3, it uses the three months
    before the prediction start time to get the average elasticity and predict.
    '''
    def __init__(self, to_forecast, gdp, lag_window, elasticity=None):
        '''
        Inputs:
            gdp: pd.Series with time index
            to_forecast: pd.Series with time index
            lag_window:: int
        '''
        self.gdp = gdp
        self.to_forecast = to_forecast
        self.lag_window = lag_window
        self.elasticity_param = elasticity
        self.gdp_g, self.to_forecast_g = self.get_growth_rates()
        self.elasticity = self.to_forecast_g / self.gdp_g
        if self.elasticity_param:
            self.elasticity = self.elasticity.map(lambda x: self.elasticity_param)
        self.elasticity_used = None
        self.rolling_elasticity = None

    def get_growth_rates(self):
        '''
        Get the growth rate of gdp and to forecast. It is the interanual growth rate.
        '''
        gdp_growth = self.gdp[self.gdp.notna()].pct_change(12)
        to_forecast_growth = self.to_forecast[self.to_forecast.notna()]\
                                                              .pct_change(12)

        return (gdp_growth, to_forecast_growth)

    def fit(self):
        '''
        Fit model by building rolling elasticity
        '''
        self.rolling_elasticity = self.elasticity.rolling(window=self.lag_window,
                                                          min_periods=1).mean()
        # Returns the elasticity object to be able to work with run_model
        return self

    def predict(self, start, end):
        '''
        Get prediction from start to end dates. It uses the last available
        rolling elasticity to estimate the growth rate and forecast.
        '''
        prediction_date_range = pd.date_range(
            start, end,
            freq=self.to_forecast.index.freq)
        elasticity_date = start - relativedelta(months=1)
        gdp_growth_to_use = self.gdp_g.loc[prediction_date_range]
        self.elasticity_used = self.rolling_elasticity.loc[elasticity_date]
        to_forecast_growth = gdp_growth_to_use * self.elasticity_used
        base_values = self.to_forecast.shift(12).loc[prediction_date_range]
        prediction = base_values * (1 + to_forecast_growth)
        return prediction

    def plot(self, outcome_var_name, save_to=None):
        '''
        Plot growth rates of GDP and variable to predict, and elasticity and rolling
        elasticity.
        Inputs:
            outcome_var_name: str
            save_to: str
        ''' 
        fig, ax = plt.subplots(ncols=2, figsize=(16, 6))
        ax[0].plot(self.gdp_g, label='gpd_g')
        ax[0].plot(self.to_forecast_g, label='{}_g'.format(outcome_var_name))
        ax[0].legend()
        ax[0].grid()
        ax[0].set_title('GDP and {} growth rate'.format(outcome_var_name))
        ax[1].plot(self.elasticity, label='elasticity', color='r')
        ax[1].plot(self.rolling_elasticity, label='rolling_elasticity',
                   color='b', linestyle='--')
        ax[1].legend()
        ax[1].grid()
        ax[1].set_title('Elasticity of {} and GDP'.format(outcome_var_name))
        if save_to:
            plt.savefig(save_to)
        plt.show()



import numpy as np
import pmdarima
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import datetime as datetime
import pandas_datareader as pdr
import time
from scipy import signal
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px

def get_opt_diff(pd_series):
    '''
        function fot best diff for get confidence about stationary of the series with adfuler test
    '''
    
    adf_pvalue = adfuller(pd_series)[1]
    opt_diff = 0
    while True:
        if adf_pvalue > 0.05:
            opt_diff += 1
            pd_series = (pd_series - pd_series.shift(1)).dropna()
            adf_pvalue = adfuller(pd_series)[1]
        else:
            break
    return opt_diff

def get_model_mesage(model_num, p, opt_diff, q, P, D, Q, seasonality, model_aic, duration):
    '''
        a function for print message of the searchong on the models
    '''
    
    msg = (f'Model No. {model_num} ARIMA ({p}, {opt_diff}, {q})({P}, {D}, {Q}) [{seasonality}], AIC = {model_aic}, time = {duration} sec')
    print (msg)

def calculate_best_sarima_model(series, pmax, qmax, Pmax, Qmax, seasonality):
    '''
        a function for searching on models of SARIMA model
    '''
    
    model_count = 0
    best_aic = np.inf
    opt_diff = get_opt_diff(pd_series = series)
    
    seasonality_test = pmdarima.arima.OCSBTest(seasonality, lag_method='aic', max_lag=36)
    D = seasonality_test.estimate_seasonal_differencing_term(series)
    
    for p in range(0, pmax+1):
        for q in range(0, qmax+1):
            for P in range(0, Pmax+1):
                for Q in range(0, Qmax+1):
                    t1 = time.time()
                    model_count += 1
                    arima_model = ARIMA(series,
                                        exog = None,
                                        order = (p, opt_diff, q),
                                        seasonal_order=(P, 1, Q, seasonality),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False,
                                        trend=(0, 0, 1, 0))
                    fitted_arima = arima_model.fit()
                    model_aic = fitted_arima.aic
                    if model_aic<best_aic:
                        best_aic = model_aic
                        optimal_p = P
                        optimal_q = q
                        optimal_P = P
                        optimal_Q = Q
                        
                    t2 = time.time()
                    
                    get_model_mesage(model_count, p, opt_diff,
                                     q, P, D, Q,
                                     seasonality, round(model_aic, 1),
                                     round(t2-t1, 1))
                    
    best_model_params = {'optimal_p': optimal_p, 'optimal_q': optimal_q,
                  'optimal_P': optimal_P, 'optimal_Q': optimal_Q,
                  'opt_diff': opt_diff}
    
    #print best model
    best_model = ARIMA(series,
                        exog = None,
                        order = (optimal_p, opt_diff, optimal_q),
                        seasonal_order=(optimal_P, 1, optimal_Q, seasonality),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        trend=(0, 0, 1, 0))
    
    fitted_model = best_model.fit()
    return fitted_model, best_model_params

class sarima_model:
    '''
        main class
    '''
    
    def __init__(self, series, pmax, qmax, Pmax, Qmax, seasonality):
        self.series = series
        self.pamx = pmax
        self.qmax = qmax
        self.Pmax = Pmax
        self.Qmax = Qmax
        self.seasonality = seasonality
        
    def get_optimal_sarima_model(self):
        self.best_model, self.best_params = calculate_best_sarima_model(self.series, self.pamx, self.qmax,
                                                      self.Pmax, self.Qmax, self.seasonality)
        
        print('\n ================= best model ====================')
        print(self.best_params)
        print(self.best_model.summary())
        return self.best_model
        
    def forecast(self, steps, actual_values):
        self.actual = actual_values
        self.fsteps = steps
        self.forecast = self.best_model.forecast(steps = steps, exog = None)
        forecast_data = pd.DataFrame({'Forecast': self.forecast.values, 'Actual': self.actual.values})
        forecast_data.index = actual_values.index
        forecast_data['error'] = 100 * np.abs(((forecast_data['Forecast']/forecast_data['Actual']) - 1).round(3))
        self.forecast_data = forecast_data
        return forecast_data
    
    def plot_forecast(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.forecast_data['Actual'], label = 'Actual')
        plt.plot(self.forecast_data['Forecast'], label = 'Forecast')
        plt.title('Actual and Forecast values by best Sarima model')
        plt.legend()
        plt.show()
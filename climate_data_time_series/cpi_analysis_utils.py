import pandas as pd
import numpy as np
# Get yearmonth as int from the date
def get_timeindex(X):
    return ((X - pd.to_datetime('2008-07-01')) / np.timedelta64(1, 'M')).astype(int)

def get_date(t, t_to_date_dict):
    return t_to_date_dict.get(t)

def get_cpi_data_transformed():
    CPI_raw = pd.read_csv("release_time_series_report_data_nops/CPI.csv")
    # Convert date to Datetime
    CPI_raw['date'] = pd.to_datetime(CPI_raw['date'])
    CPI_raw['t'] = get_timeindex(CPI_raw['date'])

    CPI_raw.to_csv('release_time_series_report_data_nops/CPI_transformed.csv')
    return CPI_raw

def get_CPI_monthly_pct(X):
    CPI_monthly = X.groupby('t').mean().reset_index()
    CPI_monthly['pct'] = CPI_monthly['CPI'].pct_change()
    CPI_monthly['ln_CPI'] = np.log(CPI_monthly['CPI'])
    CPI_monthly['ln_diff'] = CPI_monthly['ln_CPI'].diff()
    return CPI_monthly.sort_values(by=['t'])

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

def performance(y_test, y_test_hat):
    # Evaluate performance
    RMSE = mean_squared_error(y_test, y_test_hat) ** 0.5
    MAPE = mean_absolute_percentage_error(y_test, y_test_hat)
    print("RMSE is: ", RMSE)
    print("MAPE is: ", MAPE)
    return RMSE, MAPE

import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsp
def ACF(X):
    # Find order P by plotting ACF/PACF
    tsp.plot_acf(X, title="ACF for CPI residuals")
    plt.show()
    tsp.plot_pacf(X, title="PACF for CPI residuals")
    plt.show()


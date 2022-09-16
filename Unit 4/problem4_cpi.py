import numpy as np


np.set_printoptions(suppress=True)
import pandas as pd
from sklearn.linear_model import LinearRegression

CPI_raw = pd.read_csv("release_time_series_report_data_nops/CPI.csv")
# Convert date to Datetime
CPI_raw['date'] = pd.to_datetime(CPI_raw['date'])
# Get yearmonth as int from the date
CPI_raw['t'] = ((CPI_raw['date'] - pd.to_datetime('2008-07-01')) / np.timedelta64(1, 'M')).astype(int)

# Split into training data and test data set, training data includes all months  prior to and not including September 2013.
CPI_train = CPI_raw[CPI_raw['date'] < "2013-09-01"]
CPI_test = CPI_raw[CPI_raw['date'] >= "2013-09-01"]


def get_grouped_mean(X):
    X_groupby_month = X.groupby('t')
    X_t = X_groupby_month.groups.keys()
    X_average = X_groupby_month[['CPI']].mean()
    X_t = np.array(list(X_t)).reshape(-1, 1)
    y = X_average.values.reshape(-1, 1)
    return X_t, y


# Fit first order linear trend in time series
CPI_train = get_grouped_mean(CPI_train)
X_train, y_train = CPI_train['t'], CPI_train['CPI']

CPI_test = get_grouped_mean(CPI_test)
X_test, y_test = CPI_train['t'], CPI_train['CPI']

trend_model = LinearRegression()
trend_model.fit(X_train, y_train)
y_train_hat = trend_model.predict(X_train)

# The coefficients
print("Intercept", trend_model.intercept_, "Coefficients: ", trend_model.coef_)

# Residuals from linear trend
residual_train = y_train - y_train_hat
# plt.scatter(X_train, R_train)
# plt.show()
print("Max Residuals: ", np.max(residual_train))


# ACF(R_train)

import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

res = AutoReg(residual_train, lags=2, trend='n').fit()
print("Auto regressive model params: ", res.summary())

# Verify this AR model + the trend model is same if we directly fit the AR model allowing for trend and constant
ar_trend_model = AutoReg(X_train, lags=2, trend='ct').fit()
print("Auto regressive model with linear trend and AR: ", ar_trend_model.summary())

from statsmodels.tools.eval_measures import rmse

# N = len(X_train) + len(X_test)
# y_test_hat = ar_trend_model.predict(start=len(X_train), end=N - 1)
# ar_trend_model.plot_predict(start=len(X_train), end=N - 1)
# print("RMSE: ", rmse(y_test.reshape(-1, 1), y_test_hat.reshape(-1, 1)))

trend_test = trend_model.predict(X_test)
X_test_residuals = X_test - trend_test

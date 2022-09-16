import numpy as np
np.set_printoptions(suppress=True)
from cpi_analysis_utils import *

CPI = get_cpi_data_transformed()
CPI_monthly = get_CPI_monthly_pct(CPI)

# Split into training data and test data set, training data includes all months  prior to and not including September 2013.
CPI_monthly_train = CPI_monthly[CPI_monthly['t']<62].iloc[1:, :]
CPI_monthly_test = CPI_monthly[CPI_monthly['t']>=62]

#Refit the trend + AR(2) model as down earlier in problem 4
#Step 1: detrend

#1. Description of how you compute the monthly inflation rate from  and a plot of the monthly inflation rate
import matplotlib.pyplot as plt
def plot_monthly_inflation_rate(CPI_monthly):
    plt.plot(CPI_monthly.t, CPI_monthly.pct)
    plt.xlabel("Monthly index")
    plt.ylabel("Inflation rate (pct change from previous month)")
    plt.title("Inflation rate over time (monthly)")
    plt.show()

plot_monthly_inflation_rate(CPI_monthly_train)
#There seems to be a polynomial trend in the chart, let's try to see what order works the best here, measured by test data set MSE

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

min_rmse = 1000
best_polydegree = -1
best_model = None
best_polyfeature = None

X_train = CPI_monthly_train.t.to_numpy().reshape(-1, 1)
y_train = CPI_monthly_train.pct.to_numpy().reshape(-1, 1)

X_test = CPI_monthly_test.t.to_numpy().reshape(-1, 1)
y_test = CPI_monthly_test.pct.to_numpy().reshape(-1, 1)

for degree in [2, 3, 4, 5]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_feature = poly.fit_transform(X_train)
    poly_model = LinearRegression()
    poly_model.fit(poly_feature, y_train)

    poly_feature_test = poly.fit_transform(X_test)
    y_test_hat = poly_model.predict(poly_feature_test)

    rmse, mape = performance(y_test_hat, y_test)
    if rmse < min_rmse:
        min_rmse = rmse
        best_polydegree = degree
        best_model = poly_model
        best_polyfeature = poly_feature

print("Best degree for polymodel is: ", best_polydegree, "with RMSE of: ", min_rmse)

#Turns out ot be a quadratic trend
detrend_residual_train = y_train - poly_model.predict(poly_feature)

plt.plot(X_train, detrend_residual_train)
plt.xlabel("Monthly index")
plt.ylabel("De-trended (quadratic trend) Inflation Pct residual")
plt.title("Residual over time (monthly)")
plt.show()

#Plot ACF and PACF to understand residuals better to see what pattern emerges
ACF(detrend_residual_train)
#Looks like a lag of 1 to me, but we will try different AR lag models and see which one has best fit

from statsmodels.tsa.ar_model import AutoReg
ar1_model = AutoReg(detrend_residual_train, lags=1, trend='n')
ar1_cpi_pct_fit = ar1_model.fit()
print("Auto regressive model params: ", ar1_cpi_pct_fit.summary())






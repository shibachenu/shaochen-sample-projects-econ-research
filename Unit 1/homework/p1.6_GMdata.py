# GM data
import zipfile
import numpy as np
import pandas as pd
np. set_printoptions(suppress=True)

# returns a 3-tuple of (list of city names, list of variable names, numpy record array with each variable as a field)
def read_mortality_csv(zip_file):
    import io
    import csv
    fields, cities, values = None, [], []
    with io.TextIOWrapper(zip_file.open('data_and_materials/mortality.csv')) as wrap:
        csv_reader = csv.reader(wrap, delimiter=',', quotechar='"')
        fields = next(csv_reader)[1:]
        for row in csv_reader:
            cities.append(row[0])
            values.append(tuple(map(float, row[1:])))
    dtype = np.dtype([(name, float) for name in fields])
    return cities, fields, np.array(values, dtype=dtype).view(np.recarray)


with zipfile.ZipFile("release_statsreview_release1.zip") as zip_file:
    m_cities, m_fields, m_values = read_mortality_csv(zip_file)

#Try to calculate beta from data

#You should arrange your vector in this order: intercept, JanTemp, JulyTemp, RelHum, Rain, Educ, Dens,
# NonWhite, WhiteCollar, Pop, House, Income, HC, NOx, SO2
#Now try rescale X and Ys by standardize them


N = len(m_cities)
P = len(m_values[0])

X = pd.DataFrame(m_values[m_fields[1:]]).to_numpy()
Y = m_values['Mortality']

from scipy import stats
X_std = (X - np.mean(X, axis=0))/np.std(X, axis=0)
Y_std = (Y - np.mean(Y))/np.std(Y)

X_prime = np.concatenate((np.ones_like(X_std[:, :1]), X_std), axis=1)

# Estimate beta in 2 different ways
beta = np.linalg.inv(X_prime.transpose().dot(X_prime)).dot(X_prime.transpose()).dot(Y_std)
print("beta with matrix form: ", beta)

#With skitlearn LinearRegression
# Linear regression
from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(X_prime, Y_std)
print("intercept:", model.intercept_, "coef: ", model.coef_)

import pylab as py
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pyplot as plt
#Plot patterns for each feature and Y scatters to search for potential log transformation needed
for field in m_fields:
    X_field = np.log(m_values[field])
    X_field_std = (X_field - np.mean(X_field))/np.std(X_field)
    est = sm.OLS(Y_std, X_field_std)
    residuals = Y_std - est.fit().predict(X_field_std)
    sm.qqplot(residuals, line='45')
    py.title("QQ plot for field: "+field)
    py.savefig("figure/Log Reg for "+field)


from gradient_descent import gradient_descent

# With Gradient descent
step_sizes = np.arange(0.0001, 0.5, step=0.001)
num_steps = len(step_sizes)
for i in range(num_steps):
    step_size = step_sizes[i]
    beta, error, epoch = gradient_descent(X_prime, Y, step_size=step_size)
    print("Step size:", step_size, "beta is: ", beta, "error is: ", error, "num of epochs is", epoch)

# Verify and try SGD
print("verify with sklearn GD algo")
from sklearn.linear_model import SGDRegressor

errors = np.zeros(num_steps)
for i in range(num_steps):
    step_size = step_sizes[i]
    sgdr = SGDRegressor(alpha=step_size)
    sgdr.fit(X_prime, Y)
    Y_hat = sgdr.predict(X_prime)
    error = np.mean((Y - Y_hat) ** 2)
    errors[i] = error
    print("Step size:", step_size, "beta is: ", sgdr.coef_, "error is: ", error)
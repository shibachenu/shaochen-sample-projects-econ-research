"""Linear regression, and correlation etc"""
import matplotlib.pyplot as plt  # import the library

Xs = np.random.normal(0, 2, 100)
Ys = 2 * Xs + np.random.normal(0, 0.5, 100)
# scatter plot
plt.scatter(Xs, Ys)  # Create the scatter plot, Xs and Ys are two numpy arrays of the same length
plt.show()

# line plot
plt.plot(Xs, Ys)

# matrix inversion
Zs = np.random.normal(2, 5, size=(4, 4))
print(Zs)
Z_inv = np.linalg.inv(Zs)
print(Z_inv)

# Distribution utils
import scipy.stats

T = 2.7
num_degrees_of_freedom = 90
scipy.stats.t.sf(T, num_degrees_of_freedom)

Y2s = Xs ** 2
from scipy.stats import pearsonr

print("Correlation coefficient is: ", pearsonr(Xs, Y2s))

"""Astronomy exercise on correlation"""

Xs = np.array(
    [0.0339, 0.0423, 0.213, 0.257, 0.273, 0.273, 0.450, 0.503, 0.503, 0.637, 0.805, 0.904, 0.904, 0.910, 0.910, 1.02,
     1.11, 1.11, 1.41,
     1.72, 2.03, 2.02, 2.02, 2.02])

Ys = np.array(
    [-19.3, 30.4, 38.7, 5.52, -33.1, -77.3, 398.0, 406.0, 436.0, 320.0, 373.0, 93.9, 210.0, 423.0, 594.0, 829.0, 718.0,
     561.0, 608.0, 1.04E3, 1.10E3, 840.0, 801.0, 519.0])

N = 24

import numpy as np

mean_X = np.mean(Xs)
print("Mean of Xs is", mean_X)
mean_Y = np.mean(Ys)
print("Mean of Y is", mean_Y)

# Standard dev
sd_X = np.std(Xs, ddof=1)
sd_Y = np.std(Ys, ddof=1)
print("Std for Xs: ", sd_X)
print("Std for Ys: ", sd_Y)

cov_XY = np.std(Ys, ddof=1)
print("Covariance of Xs and Ys are: ", cov_XY)

from scipy.stats import pearsonr

cor_XY = pearsonr(Xs, Ys)[0]
print("Correlation coefficient is: ", cor_XY)

beta1 = cor_XY * sd_Y / sd_X
beta0 = mean_Y - beta1 * mean_X

print("beta0 is", beta0, "beta1 is: ", beta1)

# Calculate R2

Y_hat = beta0 + beta1 * Xs

SumResidual = np.sum((Ys - Y_hat) ** 2)
SumTotal = np.sum((Ys - mean_Y) ** 2)

R2 = 1 - SumResidual / SumTotal

print("R2 is: ", R2)

"""Non-linear transformation for regression:

Each data point is one planet in our solar system (with the addition of the planetoid Pluto, which will be henceforth referred to as a planet for simplicity).

The  values are the semi-major axis of each planet's orbit around the Sun. A planetary orbit is elliptical in shape, and the semi-major axis is the longer of the two axes that define the ellipse. When the ellipse is nearly circular (which is true for most planets), the semi-major axis is approximately the radius of said circle. The  values are measured in units of Astronomical Units (AU). One AU is very close to the average distance between the Sun and Earth (defined as 149597870700 meters), hence, the Earth's semi-major axis is essentially 1 AU due to its very circular orbit.

The  values are the orbital period of the planet, measured in Earth years (365.25 days), so Earth also has a  year.
"""

import numpy as np

Xs = np.array([0.387, 0.723, 1.00, 1.52, 5.20, 9.54, 19.2, 30.1, 39.5])

Ys = np.array([0.241, 0.615, 1.00, 1.88, 11.9, 29.5, 84.0, 165.0, 248])

N = 9

# plot first for visual inspection
import matplotlib.pyplot as plt  # import the library

# scatter plot
plt.scatter(Xs, Ys)  # Create the scatter plot, Xs and Ys are two numpy arrays of the same length
# line plot
plt.plot(Xs, Ys)
plt.show()
# does not look very linear to me

# correlation
from scipy.stats import pearsonr

cor_XY = pearsonr(Xs, Ys)[0]
print("Correlation coefficient is: ", cor_XY)

# linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()

X_prime = Xs.reshape(-1, 1)
model.fit(X_prime, Ys)
Y_hat = model.predict(X_prime)
residuals = Ys - Y_hat

# plot residuals
plt.scatter(Xs, residuals)
plt.show()

plt.scatter(Y_hat, residuals)
plt.show()

# QQ plot to test normality
import statsmodels.api as sm

sm.qqplot(Xs, line='s')
plt.title("X distribution")
plt.show()

sm.qqplot(Ys, line='s')
plt.title("Y distribution")
plt.show()

sm.qqplot(residuals, line='s')
plt.title("Residuals distribution")
plt.show()

# Explore some transformation
LogY = np.log(Ys)
plt.plot(Xs, LogY)
plt.title("Log Y vs X")
plt.show()

LogX = np.log(X_prime)
plt.plot(LogX, Ys)
plt.title("Y vs Log X")
plt.show()

plt.plot(LogX, LogY)
plt.title("LogY vs Log X")
plt.show()

# Linear regression with transformed variables

model2 = LinearRegression()
model2.fit(LogX, LogY)

print("Transformed linear model params: intercept", model2.intercept_, "slope: ", model2.coef_)

"""Exoplanet mass data.

For this exercise, we will perform multiple linear regression on some exoplanetary data to see if we can find a relationship that can predict the mass of an exoplanet.
"""
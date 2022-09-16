import numpy as np

LogPlanetMass = np.array([-0.31471074, 1.01160091, 0.58778666, 0.46373402, -0.01005034,
                          0.66577598, -1.30933332, -0.37106368, -0.40047757, -0.27443685,
                          1.30833282, -0.46840491, -1.91054301, 0.16551444, 0.78845736,
                          -2.43041846, 0.21511138, 2.29253476, -2.05330607, -0.43078292,
                          -4.98204784, -0.48776035, -1.69298258, -0.08664781, -2.28278247,
                          3.30431931, -3.27016912, 1.14644962, -3.10109279, -0.61248928])

LogPlanetRadius = np.array([0.32497786, 0.34712953, 0.14842001, 0.45742485, 0.1889661,
                            0.06952606, 0.07696104, 0.3220835, 0.42918163, -0.05762911,
                            0.40546511, 0.19227189, -0.16251893, 0.45107562, 0.3825376,
                            -0.82098055, 0.10436002, 0.0295588, -1.17921515, 0.55961579,
                            -2.49253568, 0.11243543, -0.72037861, 0.36464311, -0.46203546,
                            0.13976194, -2.70306266, 0.12221763, -2.41374014, 0.35627486])

LogPlanetOrbit = np.array([-2.63108916, -3.89026151, -3.13752628, -2.99633245, -3.12356565,
                           -2.33924908, -2.8507665, -3.04765735, -2.84043939, -3.19004544,
                           -3.14655516, -3.13729584, -3.09887303, -3.09004295, -3.16296819,
                           -2.3227878, -3.77661837, -2.52572864, -4.13641734, -3.05018846,
                           -2.40141145, -3.14795149, -0.40361682, -3.2148838, -2.74575207,
                           -3.70014265, -1.98923527, -3.35440922, -1.96897409, -2.99773428])

StarMetallicity = np.array([0.11, -0.002, -0.4, 0.01, 0.15, 0.22, -0.01, 0.02,
                            -0.06, -0.127, 0., 0.12, 0.27, 0.09, -0.077, 0.3,
                            0.14, -0.07, 0.19, -0.02, 0.12, 0.251, 0.07, 0.16,
                            0.19, 0.052, -0.32, 0.258, 0.02, -0.17])

LogStarMass = np.array([0.27002714, 0.19144646, -0.16369609, 0.44468582, 0.19227189,
                        0.01291623, 0.0861777, 0.1380213, 0.49469624, -0.43850496,
                        0.54232429, 0.02469261, 0.07325046, 0.42133846, 0.2592826,
                        -0.09431068, -0.24846136, -0.12783337, -0.07364654, 0.26159474,
                        0.07603469, -0.07796154, 0.09440068, 0.07510747, 0.17395331,
                        0.28893129, -0.21940057, 0.02566775, -0.09211529, 0.16551444])

LogStarAge = np.array([1.58103844, 1.06471074, 2.39789527, 0.72754861, 0.55675456,
                       1.91692261, 1.64865863, 1.38629436, 0.77472717, 1.36097655,
                       0., 1.80828877, 1.7837273, 0.64185389, 0.69813472,
                       2.39789527, -0.35667494, 1.79175947, 1.90210753, 1.39624469,
                       1.84054963, 2.19722458, 1.89761986, 1.84054963, 0.74193734,
                       0.55961579, 1.79175947, 0.91629073, 2.17475172, 1.36097655])

N = 30

"""Choice of variable transformation.

All of these observed quantities have been transformed by taking the natural logarithm. When performing linear regression, it can help to have a general idea on how the predictors contribute to the predicted quantity.

For example, if one were attempting to predict the sales of a store based on the population of surrounding region, then we might expect that the sales will be cumulative in the population variables. In this case, it would be best to leave these variables as they are, performing the linear regression directly on them.

However, in astronomy and physics, it is very common for the predicted variable to be multiplicative in the predictors. For example, the power that a solar cell produces is the product of the amount of solar radiation and the efficiency of the cell. In that case, it is better to transform the variables by taking the logarithm as discussed previously.

LogPlanetMass is the logarithm of the observed exoplanet's mass in units of Jupiter's mass. A LogPlanetMass of zero is an exoplanet with the same mass as Jupiter. Jupiter is used as a convenient comparison, as large gas giants are the most easily detected, and thus most commonly observed, kind of exoplanet. LogPlanetRadius is the logarithm of the observed exoplanet's radius in units of Jupiter's radius, for much the same reason. LogPlanetOrbit is the logarithm of the observed planet's semi-major axis of orbit, in units of AU. StarMetallicity is the relative amount of metals observed in the parent star. It is equal to the logarithm of the ratio of the observed abundance of metal to the observed abundance of metal in the Sun. The Sun is a quite average star, so it serves as a good reference point. The most common metal to measure is Iron, but astronomers define any element that isn't Hydrogen or Helium as a metal. LogStarMass is the logarithm of the parent star's mass in units of the Sun's mass. LogStarAge is the logarithm of the parent star's age in giga-years.
"""


Xs = np.concatenate((np.ones(N), LogPlanetRadius, LogPlanetOrbit, StarMetallicity, LogStarMass, LogStarAge)).reshape(6,
                                                                                                                     N).transpose()
# Answers from course
Ys = LogPlanetMass
# Concatenate the variables into a matrix, np.ones_like inserts a row of ones into the start of the matrix for the intercept term.
# Taking the transpose places each variable as columns.
Xmat = np.array(
    (np.ones_like(LogPlanetRadius), LogPlanetRadius, LogPlanetOrbit, StarMetallicity, LogStarMass, LogStarAge)).T
from numpy.linalg import inv

# The beta estimator using the matrix inversion formula
betaVec = inv(Xmat.T.dot(Xmat)).dot(Xmat.T).dot(Ys)
print("beta estimated: ", betaVec)

residuals = Ys - Xmat.dot(betaVec)
print("residuals: ", residuals.shape)


"""T-test for the Exoplanet datasets"""
# Test stats calculation
P = betaVec.shape[0]
print("number of features: ", P)
cov = inv(Xmat.T.dot(Xmat))
sigma_hat = np.sqrt(np.sum(residuals ** 2) / (N - P))

print("signma hat", sigma_hat)
Ts = np.zeros(P)

for i in range(P):
    Ts[i] = betaVec[i]/(sigma_hat * np.sqrt(cov[i][i]))

print("Test stats: ", Ts)

#X1 >X5>X2>x3>X4

#validate results
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

est = sm.OLS(Ys, Xmat)
est2 = est.fit()

print(est2.summary())

#solution from the course
# First, estimate the standard deviation of the noise.
sigmaHat = np.sqrt( np.sum( np.square(Ys - Xmat.dot(betaVec) )) / ( N - Xmat.shape[1] ) )
# Now estimate the (matrix part of the) covariance matrix for beta
import numpy.linalg
betaCov = numpy.linalg.inv(Xmat.T.dot(Xmat))
# Use the formula for the t-test statistic for each variable
tVals = betaVec/(sigmaHat * np.sqrt(np.diagonal(betaCov)))
# Calculate the 2-sided p-values.
import scipy.stats
pvals = scipy.stats.t.sf(np.abs(tVals), N-Xmat.shape[1])*2


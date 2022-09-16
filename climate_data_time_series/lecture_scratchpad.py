#Autocovariance for different time series
import math

import numpy as np
import statsmodels.tsa.stattools as tsa

X1 = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
print("Autocov for X1: ", tsa.acovf(X1))

X2 = np.array([-1, 0, 1, 0, -1, 0, 1, 0])
print("Autocov for X2: ", tsa.acovf(X2))

#Information criterion comparisons

def AIC(k, L):
    return 2*k - 2*math.log(L)

def BIC(k, n, L):
    return k*math.log(n) - 2*math.log(L)


n=1000
k_A = 10
L_A = 10

k_B = 20
L_B = 10**6

print("AIC for model A is: ", AIC(k_A, L_A), "for model B is, ", AIC(k_B, L_B))
print("BIC for model A is: ", BIC(k_A, n, L_A), "for model B is, ", BIC(k_B, n, L_B))

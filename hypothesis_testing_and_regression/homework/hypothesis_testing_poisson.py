import numpy as np
import csv

#Problem 1.2: Hypothesis with Possion modeling for ray emission
filename = "data_and_materials/gamma-ray.csv"

with open(filename) as filename:
    reader = csv.reader(filename)
    headers = next(reader)
    gamma_ray = np.array(list(reader)).astype(float)

G = gamma_ray[:, 1]
T = gamma_ray[:, 0]
avg_ray_intervals = G/T
avg_ray_interval = np.mean(avg_ray_intervals)
print("Average ray emissions per second: ", avg_ray_interval)


from scipy import stats
#H0: lamda_i = lambda
lambda0 = avg_ray_interval


L0 = np.product(stats.poisson.pmf(G, lambda0 * T))
l0 = np.log(L0)

#H1: lamda_i = G_i/t_i
L1 = np.product(stats.poisson.pmf(G, G))
l1 = np.log(L1)

#LRT stats
T = -2 * (l0 - l1)

#Chi square test, critical value
df = len(G) - 1 #should be 99
critical_val = stats.chi2.ppf(0.95, df=df)

p_val = 1-stats.chi2.cdf(T, df=df)

print("Test stats is: ", T, "Critical value is: ", critical_val, "P value is: ", p_val)


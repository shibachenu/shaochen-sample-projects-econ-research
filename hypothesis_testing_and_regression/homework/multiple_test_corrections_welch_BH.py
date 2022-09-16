#Problem 1.4: Regression and multiple testing correction
import numpy as np

import pandas as pd
golub_data=pd.read_csv('data_and_materials/golub_data/golub.csv', sep=',').transpose()
golub_data = golub_data.drop(golub_data.index[0])
golub_data['patientId']= pd.to_numeric(golub_data.index)
golub_classnames = pd.read_csv('data_and_materials/golub_data/golub_cl.csv', sep=',', names=["patientId", "type"])
golub_classnames = golub_classnames.drop(golub_classnames.index[0])

P = len(golub_data.columns)-1 #num of genes

golub_merged = pd.merge(golub_data, golub_classnames, how='inner', on='patientId')

#Find mean and variance for ALL vs AML tumor data

X_ALL = golub_merged[golub_merged.type=="0"].iloc[:, 0:P]
X_ALL_mu = np.mean(X_ALL)
N_ALL = len(X_ALL)
var_X_ALL_mu = np.var(X_ALL, axis=0)/N_ALL

#Find all AML patients
X_AML = golub_merged[golub_merged.type=="1"].iloc[:, 0:P]
X_AML_mu = np.mean(X_AML)
N_AML = len(X_AML)
var_X_AML_mu = np.var(X_AML, axis=0)/N_AML

#Standard error for t-test
se_diff = np.sqrt(var_X_ALL_mu+var_X_AML_mu)

T_n = (X_ALL_mu - X_AML_mu)/se_diff

#Welch test stats degree of freedom
V_ALL = N_ALL-1
V_AML = N_AML - 1
V = (var_X_ALL_mu+var_X_AML_mu)**2/(var_X_ALL_mu**2/V_ALL + var_X_AML_mu**2/V_AML )

from scipy import stats
alpha = 0.05
T_critical = stats.t.ppf(1-alpha/2, df=V)

num_sig_genes = np.sum(T_n > T_critical)

P_values_uncorrected =  2*(1-stats.t.cdf(T_n, df=V))
num_sig_genes_pval = np.sum(P_values_uncorrected<alpha)

print("Num sig genes without correction is: ", num_sig_genes)


#Verify with existing lib for Welch t-test
t, p = stats.ttest_ind(X_ALL, X_AML, equal_var=False)
num_sig_ttest = np.sum(p<alpha)
print("Num sig genes without correct verified by stats t-test is: ", num_sig_ttest)

#Now, find the number of significantly associated genes using the Holm-Bonferroni and Benjamini-Hochberg corrections to the p-values.
rank = np.arange(1, P+1)
alpha_hom_bonferroni = alpha/(P+1-rank)

num_sig_genes_hom = np.sum(P_values_uncorrected<alpha_hom_bonferroni)

#Benjamini-Hochberg test
P_values_sorted = np.sort(p)
alpha_BH = alpha*rank/P

num_sig_genes_BH = np.sum(P_values_sorted<alpha_BH)

print("Num sig genes with Holm-Bonferroni is: ", num_sig_genes_hom, "Num sig genes with BH test is: ", num_sig_genes_BH)


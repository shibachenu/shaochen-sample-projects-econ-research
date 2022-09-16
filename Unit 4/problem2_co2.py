import numpy as np

np.set_printoptions(suppress=True)
from co2_analysis_utils import *

CO2 = get_co2_clean()
X = CO2['T'].values.reshape(-1, 1)
y = CO2['CO2'].values.reshape(-1, 1)

CO2_train, CO2_test = split(CO2)
# First order linear model
X_train = CO2_train['T'].values.reshape(-1, 1)
y_train = CO2_train['CO2'].values.reshape(-1, 1)
X_test = CO2_test['T'].values.reshape(-1, 1)
y_test = CO2_test['CO2'].values.reshape(-1, 1)
model1 = regression_plot(X_train, y_train, modelName="CO2 on Time order 1")
performance(y_test, model1.predict(X_test))

# Second order linear model
X_train2 = np.concatenate((X_train, X_train ** 2), axis=1)
X_test2 = np.concatenate((X_test, X_test ** 2), axis=1)
model2 = regression_plot(X_train2, y_train, modelName="CO2 on Time order 2")
performance(y_test, model2.predict(X_test2))

# Third order linear model
X_train3 = np.concatenate((X_train2, X_train ** 3), axis=1)
X_test3 = np.concatenate((X_test2, X_test ** 3), axis=1)
model3 = regression_plot(X_train3, y_train, modelName="CO2 on Time order 3")
performance(y_test, model3.predict(X_test3))

# Decided that model2 is the best fit, we now look at monthly average residuals
y_train_trend = model2.predict(X_train2)
CO2_train['CO2_detrended'] = CO2_train['CO2'].values.reshape(-1, 1) - y_train_trend
# Average monthly residuals (detrended) is considered seasonality factor in the model
P_train = CO2_train.groupby('Mn')['CO2_detrended'].mean().reset_index()
# print("Monthly CO2 detrended averages are: ", F_train)
# Plot the periodic signal . (Your plot should have 1 data point for each month, so 12 in total.)
# Clearly state the definition the Pi, and make sure your plot is clearly labeled.
# plot_interp1d(F_train['Mn'], F_train['CO2_detrended'], title="Monthly average CO2 concentration (detrended) for training data")

y_test_trend = model2.predict(X_test2)
CO2_test['CO2_detrended'] = CO2_test['CO2'].values.reshape(-1, 1) - y_test_trend
P_test = CO2_test.groupby('Mn')['CO2_detrended'].mean().reset_index()

X_all2 = np.concatenate((X, X ** 2), axis=1)
CO2['trend'] = model2.predict(X_all2)
CO2['CO2_detrended'] = CO2['CO2'] - CO2['trend']
P_all = CO2.groupby('Mn')['CO2_detrended'].mean().reset_index()
P_all = P_all.rename(columns={"CO2_detrended": "Seasonal"})
# plot_interp1d(F_all['Mn'], F_all['CO2_detrended'], title="Monthly average CO2 concentration (detrended) for complete data")

# y_train_hat = y_train_trend + F_train['CO2_detrended']
# y_test_hat = y_test_trend + F_test['CO2_detrended']
# plot_final_fit(X_train, y_train, y_train_hat, X_test, y_test, y_test_hat)

# Combining all the above into a final merged data model
CO2_final = pd.merge(CO2, P_all, how='inner', on='Mn')
CO2_final['Fit'] = CO2_final['trend'] + CO2_final['Seasonal']

# Mark training data vs testing data
CO2_final['Train'] = CO2_final.Date.isin(CO2_train.Date).astype(int)
CO2_final = CO2_final.sort_values(by=['Date'])

# Plot the final fit . Your plot should clearly show the final model on top of the entire time series,
# while indicating the split between the training and testing data.
plot_final_fit(CO2_final)
# print("CO2_final", CO2_final.head())

CO2_test = CO2_final[CO2_final['Train'] == 0]
# Performance evaluation
performance(CO2_test['CO2'], CO2_test['Fit'])

#Part (4):  (3 points) What is the ratio of the range of values of F to the amplitude of  Pi
# and the ratio of the amplitude of Pi  to the range of the residual  Ri (from removing both the trend and the periodic signal)?
# Is this decomposition of the variation of the CO concentration meaningful? (Maximum 200 words.)
F_range = CO2_final['trend'].max() - CO2_final['trend'].min()
P_range = CO2_final['Seasonal'].max() - CO2_final['Seasonal'].min()
R = CO2_final['CO2'] - CO2_final['Fit']
R_range = R.max() - R.min()

print("Ratio of F:trend to Pi is: ", F_range/P_range, "Ratio of P to R is: ", P_range/R_range)
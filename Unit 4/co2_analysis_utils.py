import numpy as np

np.set_printoptions(suppress=True)
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error


def get_co2_clean():
    CO2_raw = pd.read_csv("release_time_series_report_data_nops/CO2.csv", header=54)
    CO2_raw = CO2_raw.rename(columns=lambda x: x.strip())

    # Clean null observations: with value = -99.99
    CO2_clean = CO2_raw[CO2_raw['CO2'] != -99.99]
    print(CO2_clean.columns)

    CO2_trimmed = CO2_clean[['Yr', 'Mn', 'Date', 'CO2']]

    # i is the index as # of months since Jan 1958
    i = 12 * (CO2_trimmed['Yr'] - 1958) + CO2_trimmed['Mn'] - 1
    CO2_trimmed['T'] = (i + 0.5) / 12

    return CO2_trimmed


from sklearn.model_selection import train_test_split


def split(X, test_size=0.2, shuffle=False):
    # Spllit data into training and test
    return train_test_split(X, test_size=test_size, shuffle=shuffle)


def regression_plot(X, y, plot=False, modelName=""):
    model = LinearRegression()
    fit = model.fit(X, y)
    print(modelName, fit.intercept_, "coef: ", fit.coef_)
    y_hat = model.predict(X)
    residual = y - y_hat
    if plot:
        plt.scatter(X[:, 0], residual)
        plt.show()
    return model


def plot_interp1d(X, y, title=""):
    # Plot this residual per month, with interpl1d
    f = interpolate.interp1d(X, y)
    plt.plot(X, f(X), 'b-', X, y,
             'ro')  # plot the newly interpolated blue line with the existing data points as red points
    plt.xlabel("Month")
    plt.ylabel("De-trended CO2 seasonal average")
    plt.title(title)
    plt.show()


# Plot the final fit . Your plot should clearly show the final model on top of the entire time series,
# while indicating the split between the training and testing data.
def plot_final_fit(X):
    plt.scatter('Date', 'CO2', data=X, c='Train', label='Raw CO2')
    plt.plot('Date', 'Fit', data=X, linestyle='-', label='Predicted CO2')
    plt.xlabel("Date")
    plt.ylabel("CO2 concentration")
    plt.title("CO2 concentration trend vs predicted with trend+period model")
    plt.legend()
    plt.show()

def performance(y_test, y_test_hat):
    # Evaluate performance
    print("RMSE is: ", mean_squared_error(y_test, y_test_hat) ** 0.5)
    print("MAPE is: ", mean_absolute_percentage_error(y_test, y_test_hat))

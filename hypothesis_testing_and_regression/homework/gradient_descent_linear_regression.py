# Read data syn X and syn Y
import numpy as np
import pandas as pd
from gradient_descent import gradient_descent

X = pd.read_csv("data_and_materials/syn_X.csv").to_numpy()
Y = pd.read_csv("data_and_materials/syn_y.csv").to_numpy()

# Add intercepts to X
X_prime = np.concatenate((np.ones_like(X[:, :1]), X), axis=1)

# Estimate beta in 2 different ways
beta = np.linalg.inv(X_prime.transpose().dot(X_prime)).dot(X_prime.transpose()).dot(Y)
print("beta with matrix form: ", beta)

# Linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, Y)
print("intercept:", model.intercept_, "coef: ", model.coef_)

# With Gradient descent
step_sizes = np.arange(0.001, 0.5, step=0.005)
num_steps = len(step_sizes)
for i in range(num_steps):
    step_size = step_sizes[i]
    beta, error, epoch = gradient_descent(X_prime, Y, step_size=step_size)
    print("Step size:", step_size, "beta is: ", beta, "error is: ", error, "num of epochs is", epoch)

# plot the step size and error charts
# plot first for visual inspection
# import matplotlib.pyplot as plt  # import the library

# scatter plot
# plt.scatter(step_sizes, errors)  # Create the scatter plot, Xs and Ys are two numpy arrays of the same length
# line plot
# plt.plot(step_sizes, errors)
# plt.show()


# Verify with GD
print("verify with sklearn GD algo")
from sklearn.linear_model import SGDRegressor

errors = np.zeros(num_steps)
for i in range(num_steps):
    step_size = step_sizes[i]
    sgdr = SGDRegressor(alpha=step_size)
    sgdr.fit(X, Y)
    Y_hat = sgdr.predict(X)
    error = np.mean((Y - Y_hat) ** 2)
    errors[i] = error
    print("Step size:", step_size, "beta is: ", sgdr.coef_, "error is: ", error)



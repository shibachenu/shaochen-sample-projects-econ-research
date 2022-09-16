import numpy as np

def gradient_descent(X, Y, step_size, precision=10**(-6)):
    N = X.shape[0]
    P = X.shape[1] #num of features
    beta = np.zeros(P)

    Y_hat = X.dot(beta).reshape((N, 1))
    residuals = Y-Y_hat
    error = np.mean(abs(residuals/Y))

    epoch = 0

    while error>=precision and epoch < 100000:
        gradient = -2 * np.sum(X.T.dot(residuals), axis=1)
        beta = beta - step_size * gradient

        Y_hat = X.dot(beta).reshape((N, 1))
        residuals = Y - Y_hat
        error = np.mean(abs(residuals/ Y))

        epoch = epoch + 1
    return beta, error, epoch
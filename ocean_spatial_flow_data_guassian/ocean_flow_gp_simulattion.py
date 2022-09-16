import matplotlib.pyplot as plt
import numpy as np

from ocean_flow_utils import *
import pandas as pd


def kernel(sigma, ell, z1, z2):
    '''
    RBG: Exponential Kernel function for covariance
    :param sigma: std dev for noise
    :param el: scaling factor
    :param z1: vector 1: e.g. location or vel
    :param z2: vector 2: e.g. location or vel
    :return: Kernel value
    '''
    kernel = sigma ** 2 * np.exp(- np.linalg.norm(z1, z2) ** 2 / (2 * ell ** 2))
    return kernel


# Pick some point of interests: [339, 179]

def init():
    hi, vi = np.random.randint(0, high=N_hi, size=1)[0], np.random.randint(0, high=N_vi, size=1)[0]
    u_vel, v_vel = get_velocity_alltime(hi, vi)
    return hi, vi, u_vel, v_vel


from sklearn.model_selection import KFold

K = 10


def cv_split(u_vel, v_vel):
    vel_df = wrap_df(u_vel, v_vel)
    kf = KFold(n_splits=10, shuffle=False)
    result = next(kf.split(vel_df), None)
    vel_train, vel_test = vel_df.iloc[result[0]], vel_df.iloc[result[1]]
    return vel_train, vel_test


def get_cov(vel_1, vel_2, sigma, ell):
    # Construct the covariance matrix for the selected kernel functions and the selected set of parameters.
    # norm2 matrix size nxn
    n1 = len(vel_1)
    n2 = len(vel_2)
    norm2 = np.zeros((n1, n2))
    for i in range(n1):
        vel_i = np.array([vel_1.u_vel.to_numpy()[i], vel_1.v_vel.to_numpy()[i]])
        for j in range(n2):
            vel_j = np.array([vel_2.u_vel.to_numpy()[j], vel_2.v_vel.to_numpy()[j]])
            norm2[i][j] = np.linalg.norm([vel_i, vel_j])
    cov_matrix = (sigma ** 2) * np.exp(- norm2 / ell ** 2)
    return cov_matrix


def predict(mu_test, cov_train, cov_test, cov_train_test, cov_test_train, vel_train, mu_train, tau):
    mu_hat = mu_test + cov_test_train.dot(np.linalg.inv(cov_train + tau * np.identity(len(vel_train)))).dot(
        vel_train - mu_train)
    cov_hat = cov_test - cov_test_train.dot(np.linalg.inv(cov_train + tau * np.identity(len(vel_train)))).dot(
        cov_train_test)
    return mu_hat, cov_hat


from scipy.stats import multivariate_normal


def evaluate_ll(mu_test, cov_test, observed_test):
    '''
    Evaluate the performance of GP model with log likelihood of observed data
    :param mu_test: mean of GP
    :param cov_test: cov of GP
    :param observed_test: observed data
    :return: log likelihood of test data
    '''
    u_vel_mu = mu_test[:, 0]
    v_vel_mu = mu_test[:, 1]

    u_vel_obs = observed_test.u_vel.to_numpy().reshape(-1, 1)
    v_vel_obs = observed_test.v_vel.to_numpy().reshape(-1, 1)

    ll_u_vel = multivariate_normal.logpdf(u_vel_obs, mean=u_vel_mu, cov=cov_test, allow_singular=True)
    ll_v_vel = multivariate_normal.logpdf(v_vel_obs, mean=v_vel_mu, cov=cov_test, allow_singular=True)
    ll_test = np.sum(ll_u_vel) + np.sum(ll_v_vel)

    return abs(ll_test)


def train_evaluate(mu_train, mu_test, vel_train, vel_test, sigma, ell, tau):
    cov_train = get_cov(vel_train, vel_train, sigma, ell)  # 90x90 train cov
    cov_test = get_cov(vel_test, vel_test, sigma, ell)

    cov_train_test = get_cov(vel_train, vel_test, sigma, ell)
    cov_test_train = cov_train_test.T
    # Compute the conditional mean and variance of the testing data points,
    # given the mean and variance of the training data points and the data itself.
    mu_test, cov_test = predict(mu_test, cov_train, cov_test, cov_train_test, cov_test_train, vel_train, mu_train,
                                tau)

    # With the mu-test and sigma-test GP, we can compute the log likelihood of actually seeing test data
    ll_test = evaluate_ll(mu_test, cov_test, vel_test)
    print("Log likelihood of observed test based on GP: sigma: ", sigma, "l: ", ell, "is: ", ll_test)
    return ll_test


def plot_performance(sigmas, ells, lls):
    ax = plt.axes(projection='3d')
    # Data for three-dimensional scattered points
    ax.scatter3D(sigmas, ells, lls, cmap='Greens')
    plt.show()


def train_gp_model():
    # init the starting pos: in this case hardcoded
    hi, vi, u_vel, v_vel = init()
    print("Train GP model for location: ", (hi, vi))

    # 10-fold split vel of 2 directions into training and testing set
    vel_train, vel_test = cv_split(u_vel, v_vel)
    # Build an estimate for the mean at each possible time instance.
    # For example, you can estimate the mean as all zeros, or take averages – moving or otherwise – over your training data.
    mu_train = np.mean(vel_train)
    mu_test = 0
    # param to begin with, will tune later
    max_ll = 0
    best_sigma, best_ell = None, None
    sigmas = []
    ells = []
    tau = 0.001
    lls = []
    for sigma in np.arange(0.5, 2, step=0.5):
        for ell in np.arange(1, 8, step=1):
            sigmas.append(sigma)
            ells.append(ell)
            ll = train_evaluate(mu_train, mu_test, vel_train, vel_test, sigma, ell, tau)
            lls.append(ll)
            if ll > max_ll:
                max_ll = ll
                best_sigma, best_ell = sigma, ell

    print("Best GP model for velocities are with sigma: ", best_sigma, "ell: ", best_ell)
    plot_performance(sigmas, ells, lls)


# train_gp_model()

def plot_predictions(N_obs, X_obs, N_unobs, mu_unobs, cov_obs, title):
    plt.title(title)
    # plot points first with different colors
    obs_X_range = np.arange(0, N_obs)
    unobs_X_range = np.arange(N_obs, N_obs + N_unobs)
    plt.scatter(obs_X_range, X_obs, c='black', marker='o', label='Observed data')
    plt.scatter(unobs_X_range, mu_unobs, c='red', marker='o', label='Predicted conditional mean')
    plt.xlabel("Period/Day");
    plt.ylabel("Velocity");
    sigma_diags = np.diagonal(cov_obs)
    plt.plot(unobs_X_range, mu_unobs + 3 * np.sqrt(sigma_diags), c='r', label="2 Sigmas above the mean")
    plt.plot(unobs_X_range, mu_unobs - 3 * np.sqrt(sigma_diags), c='k', label="2 Sigmas below the mean")
    plt.legend()
    plt.show()


# We decided to pick the following param as optimial : sigma: 1.8, ell: 1, tau: 0.001

# Now we can predict even unobserved data sets, N is the num of days ahead we are predicting
def predict_future(N):
    sigma = 1.8
    ell = 1
    tau = 0.001
    hi, vi = (337, 179)
    # get observed 100 vel data
    u_vel, v_vel = get_velocity_alltime(hi, vi)
    vel_obs = wrap_df(u_vel, v_vel)
    u_vel_mean, v_vel_mean = np.mean(vel_obs.u_vel), np.mean(vel_obs.v_vel)
    cov_observed = get_cov(vel_obs, vel_obs, sigma, ell)  # 90x90 train cov

    u_vel_unobs = np.random.multivariate_normal(np.full((N), u_vel_mean), cov_observed, size=N)[:, 0]
    v_vel_unobs = np.random.multivariate_normal(np.full((N), v_vel_mean), cov_observed, size=N)[:, 0]

    # Plot obsed and predictions with error margin of 3 * sigma
    plot_predictions(N_period, vel_obs.u_vel, N, u_vel_unobs, cov_observed, title="Prediction for U velocity")
    plot_predictions(N_period, vel_obs.v_vel, N, v_vel_unobs, cov_observed, title="Prediction for V velocity")


predict_future(N=100)

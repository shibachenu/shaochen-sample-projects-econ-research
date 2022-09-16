from ocean_flow_utils import *
import numpy as np
import matplotlib.pyplot as plt


# This is the main util class for plotting, the whole plot is a (504*3 km) x (555* 3km) grid, each grid is 3km x 3 km


# The time period is 3 hr *100, total 300 hours, epislon is 3 hours
def initialize(N):
    '''
    #Random drop some particles on the graph
    :param N: num of particles
    :return:
    '''
    # his, vis = np.random.randint(0, high=N_hi, size=N), np.random.randint(0, high=N_vi, size=N)
    # return his, vis
    return np.array([347, 347, 346, 346, 337]), np.array([181, 182, 181, 182, 179])

# start position: X0 ~ Gaussian((100,350), sigma**2)
mu0 = (100, 350)
def init_gaussian(hi_var, vi_var):
    '''
    Initialize a starting point at T=0 drawn from 2-D Gaussian
    :param cov: covariance given
    :return: starting pos with hi, vi as coordinate
    '''
    cov = np.array([[hi_var, 0], [0, vi_var]])
    g_rv = np.random.multivariate_normal(mu0, cov, 1)
    hi, vi = g_rv[:, 0].astype(int), g_rv[:, 1].astype(int)
    return hi, vi

def gaussian_traj_sim():
    hi_var = np.random.randint(0, 100)
    vi_var = np.random.randint(0, 100)
    hi0, vi0 = init_gaussian(hi_var, vi_var)
    plot_trajectory(hi0, vi0)

gaussian_traj_sim()

def random_init_states_sim(N=5):
    his, vis = initialize(5)
    plot_trajectory(his, vis)

# random_init_states_sim()

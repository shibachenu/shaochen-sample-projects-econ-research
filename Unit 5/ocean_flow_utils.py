from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_hi = 555
N_vi = 504
N_period = 100


@lru_cache(maxsize=1000)
def get_velocity(T):
    T = T + 1
    # horizonal velocity of time: T
    timed_u_vel = pd.read_csv("OceanFlow/" + str(T) + "u.csv", header=None)
    # vertical velocity of time: T
    timed_v_vel = pd.read_csv("OceanFlow/" + str(T) + "v.csv", header=None)
    return timed_u_vel.to_numpy(), timed_v_vel.to_numpy()


# Get velocity (cache optimized) for time T, location indexed by loc_hi/horizontal index, loc_vi/vertical index
t_vel_dict = {}


# @lru_cache(maxsize=2500000)
def get_velocity_for_location(T, loc_hi, loc_vi):
    cached_vel = t_vel_dict.get(T)
    if cached_vel == None:
        cached_vel = get_velocity(T)
        t_vel_dict.setdefault(T, cached_vel)
    # horizonal velocity of time: T
    timed_u_vel, timed_v_vel = cached_vel
    return timed_u_vel[loc_vi, loc_hi], timed_v_vel[loc_vi, loc_hi]


# Get variance and magnitude of a location over time
def get_magnitude_alltime(loc_hi, loc_vi):
    N = 100
    mag_alltime = np.zeros(N)
    for T in range(N):
        u_vel, v_vel = get_velocity_for_location(T, loc_hi, loc_vi)
        mag = (u_vel ** 2 + v_vel ** 2) ** 0.5
        mag_alltime[T] = mag
    return np.var(mag_alltime), mag_alltime


# Question 1: find lowest variance location over time in terms of magnitude
def get_all_loc_variance(num_hi=N_hi, num_vi=N_vi):
    lowest_var = 1000
    lowest_loc = None
    highest_var = 0
    highest_loc = None

    loc_var = dict()

    for hloc in range(num_hi):
        for vloc in range(num_vi):
            mag_var, mags = get_magnitude_alltime(hloc, vloc)
            loc_var.setdefault((hloc, vloc), mag_var)
            # print("Loc: h:", hloc, "v:", vloc, "has magnitude variance of: ", mag_var)

            if mag_var != 0 and mag_var < lowest_var:
                lowest_var = mag_var
                lowest_loc = hloc, vloc

            if mag_var != 0 and mag_var > highest_var:
                highest_var = mag_var
                highest_loc = hloc, vloc

    print("Loc with lowest variance is: ", lowest_loc, "Lowest variance is: ", lowest_var)
    print("Loc with highest variance is: ", highest_loc, "Highest variance is: ", highest_var)

    return loc_var


# Is it a valid location with enough variance
def is_loc_valid(loc_hi, loc_vi):
    '''
    Check if a location is valid, defined as variance > certain theshold
    :param loc_hi:
    :param loc_vi:
    :return: true if valid, false if not
    '''
    mag_var, mag = get_magnitude_alltime(loc_hi, loc_vi)
    return True if mag_var > 0.5 else False


def wrap_df(u_vel, v_vel):
    return pd.DataFrame({'u_vel': u_vel, 'v_vel': v_vel})


loc_velocity_alltime_dic = {}


# Get velocity in forms of key: (hi, vi): val: (u_vel, v_vel) for a location for all times
@lru_cache(maxsize=200)
def get_velocity_alltime(hi, vi):
    vel_df = loc_velocity_alltime_dic.get((hi, vi))
    if vel_df != None:
        return vel_df
    u_vel = np.zeros(N_period)
    v_vel = np.zeros(N_period)

    for T in range(N_period):
        u_vel[T], v_vel[T] = get_velocity_for_location(T, hi, vi)

    loc_velocity_alltime_dic.setdefault((hi, vi), (u_vel, v_vel))

    return u_vel, v_vel


# Get correlation (max correlation of both directions) for 2 locations
@lru_cache(maxsize=200)
def get_max_cor(hi1, vi1, hi2, vi2):
    u_vel1, v_vel1 = get_velocity_alltime(hi1, vi1)
    u_vel2, v_vel2 = get_velocity_alltime(hi2, vi2)

    # Skip the ones with 0 variance
    if np.var(u_vel1) == 0 or np.var(v_vel1) == 0 or np.var(u_vel2) == 0 or np.var(v_vel2) == 0:
        return 0

    u_cor = abs(np.corrcoef(u_vel1, u_vel2)[1, 0])
    v_cor = abs(np.corrcoef(v_vel1, v_vel2)[1, 0])
    return max(u_cor, v_cor)


def trajectory(his, vis, T):
    '''
    Find the trajectory in the list of hi,vi pos over time range T
    :param hi:
    :param vi:
    :param T: end index of time
    :return: list of pos as trajectory: hi, vi
    '''
    trajectory = dict()
    trajectory.setdefault(0, (his, vis))
    for t in range(T):
        u_vels, v_vels = get_velocity_for_location(t, his, vis)  # unit is km/h
        hi_new, vi_new = (his + u_vels).astype(int), (vis + u_vels).astype(
            int)  # convert to km for hi and vi, and interval is 3 hrs
        trajectory.setdefault(t + 1, (hi_new, vi_new))
        his, vis = hi_new, vi_new
    return trajectory


def plot_points(his, vis, T, colors):
    plt.xlabel('Horizonal pos')
    plt.ylabel('Vertical pos')
    plt.scatter(his, vis, c=colors)
    plt.title("Period T=" + str(T))
    plt.xlim([0, N_hi + 2])
    plt.ylim([0, N_vi + 2])
    # plt.show()
    plt.savefig('images/flow_at_time' + str(T) + '.png')
    plt.close()


def plot_trajectory(hi0, vi0):
    print("Plot trajectory from starting pos: hi: ", hi0, "vi: ", vi0)
    colors = hi0 + vi0
    # Map key: period t, value is tuple of poss
    loc_trajectory = trajectory(hi0, vi0, N_period)

    for t in loc_trajectory:
        print("Plot point for t=", t)
        (hi_t, vi_t) = loc_trajectory.get(t)
        plot_points(hi_t, vi_t, t, colors=colors)

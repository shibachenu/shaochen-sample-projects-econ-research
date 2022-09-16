import numpy as np

from ocean_flow_utils import *

loc_var = get_all_loc_variance()
import operator

sorted_loc_var = sorted(loc_var.items(), key=operator.itemgetter(1), reverse=True)
print("Top 20 locations: ", sorted_loc_var[0:20])


# Provide the coordinates (in Kilometers) and the time stamp (in hours),
# of the point where the flow has its maximum x-axis velocity (the maximum signed value).

def find_max_u_vel():
    max_u_vel = 0
    max_loc = None
    max_T = -1
    for T in range(N_period):
        for hloc in range(N_hi):
            for vloc in range(N_vi):
                u_vel, v_vel = get_velocity_for_location(T, hloc, vloc)
                if u_vel > max_u_vel:
                    max_u_vel = u_vel
                    max_loc = hloc, vloc
                    max_T = T
        print("Completed for T: ", T)

    print("Max X horizontal vel is: ", max_u_vel, "pos is: ", max_loc, "max hour is: ", max_T)


# find_max_u_vel()


# Take the average of the velocity vector over all time and positions, so that you get an overall average velocity
# for the entire data set.

def get_average_vel():
    u_vel_sum = 0
    v_vel_sum = 0
    N_pos = N_hi * N_vi * N_period
    for T in range(N_period):
        for hloc in range(N_hi):
            for vloc in range(N_vi):
                u_vel, v_vel = get_velocity_for_location(T, hloc, vloc)
                u_vel_sum += u_vel
                v_vel_sum += v_vel
        print("Completed for T: ", T)

    print("Average for X-h vel is: ", u_vel_sum / N_pos, "Average for Y-v vel is:", v_vel_sum / N_pos)


# get_average_vel()

# Find 2 locations with max long range correlations, long range is defined as half of the island length in this case
# ~250 pos away, we will take a random searching approach

# Check if 2 locations are far enough, defined by its Euclidean distance as half of the max E-distance
max_e_distance = np.linalg.norm(np.array([N_hi, N_vi]) - np.array([0, 0]))


def is_far(hi1, vi1, hi2, vi2):
    dist = np.linalg.norm(np.array([hi1, vi1]) - np.array([hi2, vi2]))
    return dist > max_e_distance / 3


import random


def find_max_cor_locations(n_itr):
    max_cor = 0
    max_loc1 = None
    max_loc2 = None
    for _ in range(n_itr):
        hi1, vi1 = random.randint(0, N_hi - 1), random.randint(0, N_vi - 1)
        hi2, vi2 = random.randint(0, N_hi - 1), random.randint(0, N_vi - 1)
        if is_far(hi1, vi1, hi2, vi2):
            cor = get_max_cor(hi1, vi1, hi2, vi2)
            if cor > max_cor:
                max_cor = cor
                max_loc1 = hi1, vi1
                max_loc2 = hi2, vi2
        print("Completed for loc1: ", str(hi1) + "-" + str(vi1), "and loc2: ", str(hi2) + "-" + str(vi2))

    print("Max cor is: ", max_cor, "Max loc1: ", max_loc1, "Max loc2: ", max_loc2)

# Test find max cor
# find_max_cor_locations(10000)

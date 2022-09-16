import numpy as np

# Check if a matrix is positive semi-definitive, is valid covariance matrix?
X = np.array([[11, -3, 7, 5],
              [-3, 11, 5, 7],
              [7, 5, 11, -3],
              [5, 7, -3, 11]])


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


print("X is pos def? ", is_pos_def(X))

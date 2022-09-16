import numpy as np
import networkx as nx

am1 = np.array([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 1, 0, 1, 0]])

# simple graph, see if there is any self loop
print("Trace of this adjacency matrix: ", str(np.trace(am1)))

# Check if the matrix is symmetric
print("The matrix is symmetric? ", (am1.T == am1).all())

# An adjacency matrix is a square, binary matrix.
G = nx.from_numpy_matrix(am1)
print("The network graph is connected? ", nx.is_connected(G))

print("Number of connected components: ", nx.number_connected_components(G))

# Power l, such that A^l has no entry of 0s
for l in range(1, 5):
    am_l = np.linalg.matrix_power(am1, l)
    print("Matrix to the pwr of", l, "all elements non-zero?", (am_l != 0).all())

# Max num of degrees

print("Max num of degrees", np.max(np.sum(am1, axis=1)))

# Num of walk 5 from 1 to 1
am_5 = np.linalg.matrix_power(am1, 5)
print("Num of walks for 11: ", am_5[0][0])

# Read from edge list
G = nx.read_edgelist("directed_graph.txt", create_using=nx.DiGraph)
print("Num of nodes: ", nx.number_of_nodes(G), "Num of edges: ", nx.number_of_edges(G))

print("Num of self loops", nx.number_of_selfloops(G))

# for cycle in nx.simple_cycles(G):
#       print("Cycle is: ", str(cycle))

from scipy.stats import norm
import math

Tn = (10 * (0.103 - 0.1)) / (math.sqrt(0.5 * 0.5))
print("P-value is: ", str(1 - norm.cdf(Tn)))


#clustering and homophilly questions
am2 = np.array([[0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 1, 0, 1, 0]])

import networkx.algorithms.community as nx_comm
G2 = nx.from_numpy_matrix(am2)
modularity = nx_comm.modularity(G2, [{0, 2, 4, 6, 8}, {1, 3, 5, 7, 9}])
print("Modularity of this graph is", str(modularity))


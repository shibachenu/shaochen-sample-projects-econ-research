import numpy as np
import networkx as nx

am1 = np.array([[1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0]])

G1 = nx.from_numpy_array(am1, create_using=nx.DiGraph)
print("Eigenvector centrality for G1 is: ", nx.eigenvector_centrality(G1))

am2 = np.array([[1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

G2 = nx.from_numpy_array(am2, create_using=nx.DiGraph)
print("Eigenvector centrality for G2 is: ", nx.eigenvector_centrality(G2))

am3 = np.array([[1, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0]])

G3 = nx.from_numpy_array(am3, create_using=nx.DiGraph)
print("Eigenvector centrality for G3 is: ", nx.eigenvector_centrality(G3))

am4 = np.array([[0, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]])

G4 = nx.from_numpy_array(am4, create_using=nx.DiGraph)
print("Katz centrality for G4 is: ", nx.katz_centrality(G4))

am5 = np.array([[0, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1, 0]])

G5 = nx.from_numpy_array(am5, create_using=nx.DiGraph)
print("Katz centrality for G5 is: ", nx.katz_centrality(G5))

# networkx.pagerank
print("Page rank centrality scaled by out degree for G5 is: ", nx.pagerank_numpy(G5))

am6 = np.array([[0, 1, 0, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0]
                ])

G6 = nx.from_numpy_array(am6)
#Fileder vaectors are
print("Fielder vectors for G6 is: ", nx.linalg.fiedler_vector(G6))

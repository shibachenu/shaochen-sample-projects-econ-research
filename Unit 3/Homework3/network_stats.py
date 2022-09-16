import matplotlib.pyplot as plt
import numpy as np
from homework_utils import *
import time

Gs = load_networkdata()

for key, value in Gs.items():
    print("Properties for:", str(key))
    network_info(value)

# Degree centrality
# For each phase, each player, compute and list normalized degree centralization for each

def get_node_average(node_cs):
    node_average = {}
    for k, v in node_cs.items():
        average = np.average(v)
        node_average.setdefault(k, average)
    return dict(sorted(node_average.items(), key=lambda item: item[1], reverse=True))

t = time.time()
for i in Gs:
    G = Gs[i]
    dcs = nx.degree_centrality(G)
    print("For G: ", G.name, "Degree Centralities are: ", dcs)
print('Time elapsed to get degree centrality is: ', time.time() - t)

# Betweeness centrality
t = time.time()
node_cs = {}
for i in Gs:
    G = Gs[i]
    bcs = nx.betweenness_centrality(G, normalized=True)
    for k,v in bcs.items():
        node_cs.setdefault(k, []).append(v)
    print("For G: ", G.name, "Betweenness centralities are: ", bcs)
print('Time elapsed to get betweenness centrality is: ', time.time() - t)

print("Sorted node average for betweenness centrality is: ", get_node_average(node_cs))

#Eigen vector centrality
t = time.time()
node_es = {}
for i in Gs:
    G = Gs[i]
    ecs = nx.eigenvector_centrality(G)
    print("For G: ", G.name, "Eigen Vector centralities are: ", ecs)
    for k,v in ecs.items():
        node_es.setdefault(k, []).append(v)
print('Time elapsed to get eigenvector centrality is: ', time.time() - t)
print("Sorted node average for eigenvector centrality is: ", get_node_average(node_es))


#Plot network growth over time
def plot_network_growth(Gs):
    num_nodes = {}
    num_edges = {}
    for i in Gs:
        G = Gs[i]
        num_nodes.setdefault(i, G.number_of_nodes())
        num_edges.setdefault(i, G.number_of_edges())

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.scatter(num_nodes.keys(), num_nodes.values())
    ax2.scatter(num_edges.keys(), num_edges.values())

    ax1.set_title('Num of nodes over time')
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Num of nodes')

    ax2.set_title('Num of edges over time')
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Num of edges')
    plt.show()

#plot_network_growth(Gs)

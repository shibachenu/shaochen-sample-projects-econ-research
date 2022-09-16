import pandas as pd
import networkx as nx

def load_networkdata(directed=False):
    phases = {}
    G = {}
    for i in range(1, 12):
        var_name = "phase" + str(i)
        file_name = "https://raw.githubusercontent.com/ragini30/Networks-Homework/main/" + var_name + ".csv"
        phases[i] = pd.read_csv(file_name, index_col=["players"])
        phases[i].columns = "n" + phases[i].columns
        phases[i].index = phases[i].columns
        phases[i][phases[i] > 0] = 1
        if directed:
            G[i] = nx.from_pandas_adjacency(phases[i], create_using=nx.DiGraph())
        else:
            G[i] = nx.from_pandas_adjacency(phases[i])
        G[i].name = var_name
    return G


def network_info(G):
    print("Num of nodes: ", str(G.number_of_nodes()), "Num of edges: ", str(G.number_of_edges()))



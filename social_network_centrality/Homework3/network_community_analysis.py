from homework_utils import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

Gs = load_networkdata()
early_phases_co = {}

for i in range(1, 5):
    G = Gs[i]
    co = nx.average_clustering(G)
    early_phases_co.setdefault(i, co)

print("Early phase clustering coefficients: ", early_phases_co)

late_phases_co = {}

for i in range(5, 12):
    G = Gs[i]
    co = nx.average_clustering(G)
    late_phases_co.setdefault(i, co)

print("Late phases clustering coefficients:", late_phases_co)

print("Early phases average clustering: ", sum(early_phases_co.values()) / len(early_phases_co))
print("Late phases average clustering: ", sum(late_phases_co.values()) / len(late_phases_co))

print("networkx version", nx.__version__)


# Find sub-community in phase 10, 11

def louvain_community(G):
    partition = nx.algorithms.community.louvain_communities(G)
    # Modality scores with the parition
    mod_score = nx.algorithms.community.modularity(G, partition)
    print("Modularity score for: G", G.name, "is ", str(mod_score))


louvain_community(Gs[10])
louvain_community(Gs[11])

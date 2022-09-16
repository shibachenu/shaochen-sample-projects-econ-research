from homework_utils import load_networkdata
import networkx as nx
import pygraphviz as pgv
from matplotlib import pyplot as plt

Gs = load_networkdata()
#Phase 3 network viz
for i in range(1, 12):
    g = Gs[i]
    nx.draw(g, pos=nx.drawing.nx_agraph.graphviz_layout(g), with_labels=True)
    plt.title("Phase "+str(i))
    plt.show()




from homework_utils import *

Gs = load_networkdata(directed=True)

# Recall the definition of hubs and authorities. Compute the hub and authority score of each actor, and for each phase.
# With networkx you can use the nx.algorithms.link_analysis.hits function, set max_iter=1000000 for best results.
# Using this, what relevant observations can you make on how the relationship between n1 and n3 evolves over the phases. Can you make comparisons to your results in Part (g)?
for i in Gs:
    G = Gs[i]
    G_hits = nx.algorithms.link_analysis.hits(G, max_iter=1000000)
    print("For G: ", G.name, "Hubs and authorities are: ", G_hits)
from problem2_utils import *

# Problem 3: Influence of Hyper-parameters (Written Report)

# When we created the T-SNE plot in Problem 1, we ran T-SNE on the top 50 PC's of the data.
# But we could have easily chosen a different number of PC's to represent the data.
# Run T-SNE using 10, 50, 100, 250, and 500 PC's, and plot the resulting visualization for each.
# What do you observe as you increase the number of PC's used?

X, X_log = load_transform()
pcs = get_pcs(X_log, K=500)

for K in [10, 50, 100, 250, 500]:
    pcs_K = pcs[:, 0:K]
    tsne = TSNE(n_components=2, verbose=1, perplexity=40)
    z_tsne = tsne.fit_transform(pcs_K)
    scatter_plot(z_tsne[:, 0], z_tsne[:, 1], title="T-SNE plot of PCAed X with K=" + str(K))

from problem2_utils import *

#Visualization
X, X_log = load_transform()
# Provide at least one visualization which clearly shows the existence of the three main brain cell types described by the scientist, and explain how it shows this. Your visualization should support the idea that cells from a different group (for example, excitatory vs inhibitory) can differ greatly.

#PCA
pcs = get_pcs(X, 50)
#scatter_plot(z1[:, 0], z1[:, 1], "PCA first 2 components")
print("# of clusters pattern observed in the PCA charts is not clear")


#MDS but on PCs
mds = MDS(n_components=2, verbose=1, eps=1e-5)
mds.fit(pcs)
scatter_plot(mds.embedding_[:,0], mds.embedding_[:, 1], "MDS plot with top PCs")
print("# of clusters observed with MDS on 50 PCs")


#K means first
#K-means clustering
kmeans = KMeans(n_clusters=10, n_init=10)
y = kmeans.fit_predict(pcs)

#With T-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40)
z_tsne = tsne.fit_transform(pcs)
plt.scatter(z_tsne[:,0], z_tsne[:, 1], c=y)
plt.title("T-SNE plot with K-means colored")
plt.show()

print("Showing multiple clusters more than 3")

#Choosing K from K-means
K_max = 10
all_kmeans = [KMeans(n_clusters=i+1, n_init=100) for i in range(K_max)]
for i in range(K_max):
    all_kmeans[i].fit(pcs)

inertias = [all_kmeans[i].inertia_ for i in range(K_max)]
plt.plot(np.arange(1, K_max+1), inertias)
plt.title("KMeans sum of squares criteria")
plt.show()

print("The best K based on SS is: 6")





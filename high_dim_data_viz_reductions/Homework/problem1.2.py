import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def scatter_plot(X, Y, title, c):
    plt.scatter(X, Y, c)
    plt.title(title)
    plt.axis("equal")
    plt.show()

#Load cell and genes data
X = np.load("data/p1/X.npy")

#Transform to log2 (X+1)
X_log = np.log2(X+1)

#PCA for both raw data and log2 data

from sklearn.decomposition import PCA
pca1 = PCA()
z1 = pca1.fit_transform(X)
#Plot PCA of X raw
#scatter_plot(z1[:, 0], z1[:, 1], "PCA of raw X, first 2 components")

pca2 = PCA()
z2 = pca2.fit_transform(X_log)


#We can see 5 clusters looked visually most likely

#K-means clustering
z2_top50 = z2[:, 0:50]
kmeans = KMeans(n_clusters=5, n_init=10)
y = kmeans.fit_predict(z2_top50)

plt.scatter(z2_top50[:,0], z2_top50[:,1], c=y)
plt.title("PCA plot with K-means colored")
plt.show()

#With MDS
mds = MDS(n_components=2, verbose=1, eps=1e-5)
mds.fit(z2_top50)

plt.scatter(mds.embedding_[:,0], mds.embedding_[:, 1], c=y)
plt.title("MDS plot with K-means colored")
plt.show()

#With T-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40)
z_tsne = tsne.fit_transform(z2_top50)
plt.scatter(z_tsne[:,0], z_tsne[:, 1], c=y)
plt.title("T-SNE plot with K-means colored")
plt.show()

#K means diagnositics SS errors

K_max = 8
all_kmeans = [KMeans(n_clusters=i+1, n_init=100) for i in range(K_max)]
for i in range(K_max):
    all_kmeans[i].fit(z2_top50)

inertias = [all_kmeans[i].inertia_ for i in range(K_max)]
plt.plot(np.arange(1, K_max+1), inertias)
plt.title("KMeans sum of squares criteria")
plt.show()

#11. Visualizing cluster means

#Re-fit with original data
K_max = 8
all_kmeans = [KMeans(n_clusters=i+1, n_init=100) for i in range(K_max)]
for i in range(K_max):
    all_kmeans[i].fit(X)

inertias = [all_kmeans[i].inertia_ for i in range(K_max)]
plt.plot(np.arange(1, K_max+1), inertias)
plt.title("Original X: KMeans sum of squares criteria")
plt.show()


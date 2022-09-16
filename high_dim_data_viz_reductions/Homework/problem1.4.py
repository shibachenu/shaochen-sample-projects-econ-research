import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def scatter_plot(X, Y, title):
    plt.scatter(X, Y)
    plt.title(title)
    plt.axis("equal")
    plt.show()

#Load cell and genes data
X = np.load("data/p1/X.npy")

#kmeans = KMeans(n_clusters=30, n_init=10)
#y = kmeans.fit_predict(X)
#c_means = kmeans.cluster_centers_


#PCA plot for cluster centers
pca1 = PCA()
z1 = pca1.fit_transform(X)
scatter_plot(z1[:, 0], z1[:, 1], "PCA of K cluster centers, first 2 components")

#MDS plot for cluster centers
#With MDS
mds = MDS(n_components=2, verbose=1, eps=1e-5)
mds.fit(X)
scatter_plot(mds.embedding_[:,0], mds.embedding_[:, 1], "MDS of K cluster centers")


#With T-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40)
z_tsne = tsne.fit_transform(X)
scatter_plot(z_tsne[:,0], z_tsne[:, 1], title="T-SNE plot with K-means centers")


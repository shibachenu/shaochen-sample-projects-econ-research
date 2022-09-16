import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def load_transform():
    # Load cell and genes data
    X = np.load("data/p2_unsupervised_reduced/X.npy")

    # Log transformation
    X_log = np.log2(X + 1)
    #print("Shape of transformed X: ", X_log.shape)
    #print("Largest entry in the first col is", np.max(X_log[:, 0]))

    return X, X_log

def scatter_plot(X, Y, title):
    plt.scatter(X, Y)
    plt.title(title)
    plt.axis("equal")
    plt.show()

def get_pcs(X, K):
    try:
        pcs = np.load('data/pca.npy')
        return pcs
    except:
        pca1 = PCA()
        z1 = pca1.fit_transform(X)
        pcs = z1[:, 0:K]
        np.save("data/pca.npy", pcs)
        return pcs
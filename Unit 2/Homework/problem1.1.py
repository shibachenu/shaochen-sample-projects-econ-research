import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


def scatter_plot(X, Y, title, c):
    plt.scatter(X, Y, c)
    plt.title(title)
    plt.axis("equal")
    plt.show()

#Load cell and genes data
X = np.load("data/p1/X.npy")

print("Shape of X gene dataset", X.shape)

#What is the value of the largest entry in the first column? (Enter accurate to at least 5 decimal places)

print("Largest entry in the first col is", np.max(X[:, 0]))

#plot the first 2 columns too
#scatter_plot(X[:,0], X[:,1], title="Scatter plot X raw first 2 cols")

#Transform to log2 (X+1)
X_log = np.log2(X+1)
print("Largest entry in the first col is", np.max(X_log[:, 0]))


#PCA for both raw data and log2 data

from sklearn.decomposition import PCA
pca1 = PCA()
z1 = pca1.fit_transform(X)
#Plot PCA of X raw
#scatter_plot(z1[:, 0], z1[:, 1], "PCA of raw X, first 2 components")

pca2 = PCA()
z2 = pca2.fit_transform(X_log)


#For both the raw data and processed version, what percentage of the variance is explained by the first principal component? Enter an answer between 0 and 1.
print("Variance explained by first component, X raw", pca1.explained_variance_ratio_[0])
print("Variance explained by first component, X log2", pca2.explained_variance_ratio_[0])

#How many PC's are needed to explain 85% of the variance for both raw and processed data? To get a better idea of how the explained variance grow as more PCs are included, plot the cumulative explained variance versus number of PCs.

plt.plot(np.arange(0,100), np.cumsum(pca1.explained_variance_ratio_[0:100]))
plt.title("Total explained with PCA raw")
plt.show()

plt.plot(np.arange(0,500), np.cumsum(pca2.explained_variance_ratio_[0:500]))
plt.title("Total explained with PCA logX")
plt.show()

print("Num of componets needed for 85% variance is")

#Plot first 2 cols
scatter_plot(X_log[:,0], X_log[:,1], title="Scatter plot X log first 2 cols", c=None)

#Plot first 2 components
scatter_plot(z2[:, 0], z2[:, 1], "PCA of logged X, first 2 components", c=None)

#MDS plotting

mds = MDS(n_components=2, verbose=1, eps=1e-5)
mds.fit(X_log)
scatter_plot(mds.embedding_[:,0], mds.embedding_[:, 1], title="MDS plot of first 2 components", c=None)

#T-SNE plotting
#Project the data onto the top  50 PC's and run T-SNE with a perplexity value of 40 on the projected data to visualize the data in two dimensions.


tsne = TSNE(n_components=2, verbose=1, perplexity=40)
z_tsne = tsne.fit_transform(z2[:, 0:50])
scatter_plot(z_tsne[:,0], z_tsne[:, 1], title="T-SNE plot of PCAed X logged", c=None)

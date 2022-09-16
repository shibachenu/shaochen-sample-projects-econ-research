import numpy as np

from problem2_utils import *
from sklearn.linear_model import LogisticRegression


# Hyperparam1: TSNE perplexicity score
def perplexity_tuning(X, y):
    for p in [5, 40, 80, 100]:
        tsne = TSNE(n_components=2, verbose=1, perplexity=p)
        z_tsne = tsne.fit_transform(X)
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y)
        plt.title("TSNE with perplexity=" + str(p), size=10)
        plt.axis("equal")
        plt.show()


# Hyperparam2: Effect of number of PC's chosen on clustering
def num_pcs_clustering(X):
    inertias = np.zeros(4)
    Ks = [10, 100, 200, 500]
    for i in range(inertias.shape[0]):
        k = Ks[i]
        pcs_k = get_pcs(X, k)
        kmeans = KMeans(n_clusters=5, n_init=10)
        y = kmeans.fit_predict(pcs_k)
        plt.scatter(pcs_k[:, 0], pcs_k[:, 1], c=y)
        plt.title("PCA plot with K-means colored, from num of PCs: " + str(k))
        plt.show()
        inertia = kmeans.inertia_
        print("The SS score with num of PCs: ", k, "is: ", inertia)
        inertias[i] = inertia

    plt.scatter(Ks, inertias)
    plt.title("K means inertia/loss vs num of PCs")
    plt.axis("equal")
    plt.show()


# Hyperparam 3: Param 3: Clustering/Feature selection: Number of clusters chosen for use in unsupervised
# feature selection and how it affects the quality of the chosen features
def K_cluster_tuning(X, Ks, X_test, y_test):
    K_max = len(Ks)
    all_kmeans = [KMeans(n_clusters=i, n_init=100) for i in Ks]
    for i in range(K_max):
        kmeans_i = all_kmeans[i]
        y = kmeans_i.fit_predict(X)
        # Log regression on raw data and Y as label
        log_reg = LogisticRegression()
        log_reg.fit(X, y)
        print("Training score is: ", log_reg.score(X_test, y_test), "when K = ", Ks[i])

    inertias = [all_kmeans[i].inertia_ for i in range(K_max)]
    print("Ks: ", Ks, "inertias: ", inertias)

    plt.plot(Ks, inertias)
    plt.title("KMeans sum of squares criteria")
    plt.show()


X = np.load("data/p2_evaluation_reduced/X_train.npy")
X_log = np.log2(X + 1)
y = np.load("data/p2_evaluation_reduced/y_train.npy")

# perplexity_tuning(X_log, y)
# num_pcs_clustering(X_log)
X_test = np.load("data/p2_evaluation_reduced/X_test.npy")
X_test_log = np.log2(X_test + 1)
y_test = np.load("data/p2_evaluation_reduced/y_test.npy")

idx_random_rows = np.random.randint(X_log.shape[0], size=100)
X_rand = X_log[idx_random_rows, :]
K_cluster_tuning(X_rand, [3, 5, 6, 8, 10, 20], X_test_log, y_test)

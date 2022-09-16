from sklearn.linear_model import LogisticRegressionCV

from problem2_utils import *

# Visualization
X = np.load("data/p2_evaluation_reduced/X_train.npy")

# Get 500 random rows from X
#X = X[np.random.randint(X.shape[0], size=200), :]
X_log = np.log2(X + 1)
# Provide at least one visualization which clearly shows the existence of the three main brain cell types described by the scientist, and explain how it shows this. Your visualization should support the idea that cells from a different group (for example, excitatory vs inhibitory) can differ greatly.

# PCA
# pcs = get_pcs(X, 50)

# K means with K = 6
kmeans = KMeans(n_clusters=6, n_init=100)
y = kmeans.fit_predict(X_log)

# Log regression on raw data and Y as label
log_reg = LogisticRegressionCV(cv=5, Cs=[0.01, 0.1, 1, 10], max_iter=100, penalty="l1", solver="liblinear",
                               multi_class="ovr")
log_reg.fit(X_log, y)
print("Training score is: ", log_reg.score(X_log, y))
print("Cs: reversed regularization params: ", log_reg.C_)
print("Scores: from log regression", np.mean(list(log_reg.scores_.values()), axis=1))

log_coef_weights = np.max(np.abs(log_reg.coef_), axis=0)
top100_features = np.argsort(-log_coef_weights)[0:100]
print("Log regression coef weights", log_coef_weights)

X_topfeatures = X_log[:, top100_features]
log_reg_top = LogisticRegressionCV(cv=5, Cs=[0.01, 0.1, 1, 10], max_iter=20, penalty="l1", solver="liblinear",
                                   multi_class="ovr")
log_reg_top.fit(X_topfeatures, y)

X_test = np.load("data/p2_evaluation_reduced/X_test.npy")
X_test_log = np.log2(X_test+1)
X_test_topfeatures = X_test_log[:, top100_features]
y_test = np.load("data/p2_evaluation_reduced/y_test.npy")

score_test_top100 = log_reg_top.score(X_test_topfeatures, y_test)
print("Testing score trained from top features are: ", score_test_top100)

#Baseline comparisons

#Compare the obtained score with two baselines: random features (take a random selection of 100 genes)
idx_random_features = np.random.randint(X_log.shape[1], size=100)
X_randomfeatures = X_log[:, idx_random_features]
log_reg_random = LogisticRegressionCV(cv=5, Cs=[0.01, 0.1, 1, 10], max_iter=20, penalty="l1", solver="liblinear",
                                   multi_class="ovr")
log_reg_random.fit(X_randomfeatures, y)

X_test_random = X_test_log[:, idx_random_features]
score_test_random = log_reg_random.score(X_test_random, y_test)
print("Testing score trained from random features are: ", score_test_random)

#High-variance features (take the 100 genes with highest variance)
variance_X_log = np.var(X_log, axis=0)
top_variance_features_idx = np.argsort(-variance_X_log)[0:100]
variance_highvar_features = variance_X_log[top_variance_features_idx]
X_topvar_features = X_log[:, top_variance_features_idx]
log_reg_var = LogisticRegressionCV(cv=5, Cs=[0.01, 0.1, 1, 10], max_iter=20, penalty="l1", solver="liblinear",
                                   multi_class="ovr")
log_reg_var.fit(X_topvar_features, y)

X_test_var = X_test_log[:, top_variance_features_idx]
score_test_var = log_reg_var.score(X_test_var, y_test)
print("Testing score for highest variance features is: ", score_test_var)

#Plot histogram variance of top100 features and highest var features
variance_top_features = np.var(X_topfeatures, axis=0)
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
n_bins = 50
axs[0].hist(variance_highvar_features, bins=n_bins)
axs[0].title.set_text('Highest variance features')
axs[1].hist(variance_top_features, bins=n_bins)
axs[1].title.set_text('Top features picked by cross validation')
plt.show()




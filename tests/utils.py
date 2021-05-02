import numpy as np
from sklearn.cluster import KMeans


class ParameterConfig:
    def __init__(self):
        pass


def create_incomplete_matrix(Y):
    '''
        Y is the complete data matrix
        X has the same values as Y except a subset have been replace with NaN
    '''
    X = Y.copy()
    (N, d) = X.shape
    # missing entries indicated with NaN
    for i in range(N):
        X[i, np.random.randint(d)] = np.nan

    return X
            

def kmeans_missing(X, n_clusters, max_iter=10):
    """Perform K-Means clustering on data with missing values.

    Args:
    X: An [n_samples, n_features] array of data to cluster.
    n_clusters: Number of clusters to form.
    max_iter: Maximum number of EM iterations to perform.

    Returns:
    labels: An [n_samples] vector of integer labels.
    centroids: An [n_clusters, n_features] array of cluster centroids.
    X_hat: Copy of X with the missing values filled in.
    """

    # Initialize missing values to their column means
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    for i in range(max_iter):
        # do multiple random initializations in parallel
        cls = KMeans(n_clusters)

        # perform clustering on the filled-in data
        labels = cls.fit_predict(X_hat)
        centroids = cls.cluster_centers_

        # fill in the missing values based on their cluster centroids
        X_hat[missing] = centroids[labels][missing]

        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels

    return labels, centroids, X_hat
import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.cluster import KMeans
from sklearn import metrics
import unittest
import copy
import math
from klines import SetOfLines, SetOfPoints, CoresetForWeightedCenters, CorsetForKMeansForLines, CoresetStreamer


class TestKMeansIris(unittest.TestCase):
    ## KMeans w/ missing values using sklearn
    def kmeans_missing(self, X, n_clusters, max_iter=10):
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
            if i > 0:
                # initialize KMeans with the previous set of centroids. this is much
                # faster and makes it easier to check convergence (since labels
                # won't be permuted on every iteration), but might be more prone to
                # getting stuck in local minima.
                cls = KMeans(n_clusters, init=prev_centroids)
            else:
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
            prev_centroids = cls.cluster_centers_

        return labels, centroids, X_hat

    def test_scores(self):
        data = load_iris()
        X = data.data
        y = data.target
        k = len(np.unique(y))

        X_incomplete = X.copy()
        # missing entries indicated with NaN
        for i in range(X.shape[0]):
            X_incomplete[i, np.random.randint(X.shape[1])] = np.nan
            
        # X is the complete data matrix
        # X_incomplete has the same values as X except a subset have been replace with NaN
        labels, _, X_hat = self.kmeans_missing(X_incomplete, k)

        metrics.homogeneity_completeness_v_measure(labels, y)

        klines_mse_sklearn = ((X - X_hat)**2).mean()

        # ## Clustering using KLines
        displacements = np.nan_to_num(X_incomplete)

        N,d = X_incomplete.shape
        spans = np.nan_to_num(X_incomplete)
        spans[spans==0] = 1
        spans[spans!=1] = 0

        L = SetOfLines(spans, displacements, np.ones(N), np.ones(N))

        class ParameterConfig:
            def __init__(self):
                pass
            
        config = ParameterConfig()


        ## data
        k = 3
        m = min(int(N*0.1), 100)  # coreset size ~ reduction ratio
        tau = 1e-3

        config.a_b_approx_minimum_number_of_lines = min(int(N*0.1), 100) # constant 100, line 2, algo 2 BI-CRITERIA

        config.sample_size_for_a_b_approx = int(m*1.05) # |S| >= m, line 3 of algo 2
                                                        # note: there'll be a O(|S|^2) cost while computing algo 1
            
        config.farthest_to_centers_rate_in_a_b_approx = 0.25  # opp of 7/11, line 6, algo 2 BI-CRITERIA
        config.number_of_remains_multiply_factor = int(math.log(N))//k # this is `b` in algo 2, other paper, set as random here -  how to calculate it?
        config.closest_to_median_rate = (1-tau)/(2*k)  # refer line 4, algo 1, other paper

        config.median_sample_size = int(N*0.05)    # size of q_i, line 3, algo 2, other paper
        config.max_sensitivity_multiply_factor = 2  # for outliers in coresets
        config.number_of_remains = min(int(N*0.05), 20)


        def util():
            _, B, _ = CorsetForKMeansForLines(config).coreset(L, k, m, True)

            # reduce the coreset of random centers
            cwc = CoresetForWeightedCenters(config)
            MAX_ITER = 5
            while MAX_ITER > 0:        
                B = cwc.coreset(B, k, m)
                if B.get_size() <= k:
                    MAX_ITER = 0
                MAX_ITER -= 1 
            
            X_klines = L.get_projected_centers(B)
            kl_labels = L.get_indices_clusters(B)
            return X_klines, kl_labels

        klines_mse = []
        scores = []
        ITER = 10
        for i in range(ITER):
            X_klines, kl_labels = util()
            klines_mse.append(((X - X_klines)**2).mean())
            scores.append(metrics.homogeneity_completeness_v_measure(kl_labels, y))

        assert klines_mse_sklearn/np.array(klines_mse).mean() < 0.5

if __name__ == '__main__':
    unittest.main()

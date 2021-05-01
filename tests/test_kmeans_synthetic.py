#!/usr/bin/env python
# coding: utf-8

import unittest
from tests.utils import create_incomplete_matrix, kmeans_missing, ParameterConfig
import numpy as np
import math
from sklearn.datasets import make_blobs
from sklearn import metrics
from klines import SetOfLines, CoresetForWeightedCenters, CorsetForKMeansForLines, CoresetStreamer


class TestKMeansSynthetic(unittest.TestCase):
    def test_synthetic(self):
        N = 10**4
        d = 8
        k = 4
        X, y = make_blobs(n_samples=N, n_features=d, centers=k, cluster_std=4.0)

        X_incomplete = create_incomplete_matrix(X)
        _, _, X_hat = kmeans_missing(X_incomplete, k)

        mse_sklearn = ((X - X_hat)**2).mean()
        print(f'MSE sklearn: {mse_sklearn}')

        displacements = np.nan_to_num(X_incomplete)

        spans = np.nan_to_num(X_incomplete)
        spans[spans==0] = 1
        spans[spans!=1] = 0

        L = SetOfLines(spans, displacements, np.ones(N), np.ones(N))            
        config = ParameterConfig()

        ## data
        m = 100  # coreset size ~ reduction ratio
        tau = 1e-2

        config.a_b_approx_minimum_number_of_lines = 100 # constant 100, line 2, algo 2 BI-CRITERIA
        config.sample_size_for_a_b_approx = int(m*1.05) # |S| >= m, line 3 of algo 2
                                                        # note: there'll be a O(|S|^2) cost while computing algo 1
        config.farthest_to_centers_rate_in_a_b_approx = 4/11  # opp of 7/11, line 6, algo 2 BI-CRITERIA
        config.number_of_remains_multiply_factor = int(math.log(N))//k # this is `b` in algo 2, other paper, set as random here -  how to calculate it?
        config.closest_to_median_rate = (1-tau)/(2*k)  # refer line 4, algo 1, other paper
        config.median_sample_size = int(N*0.05)   # size of q_i, line 3, algo 2, other paper
        config.max_sensitivity_multiply_factor = 100  # for outliers in coresets
        config.number_of_remains = 20

        SAMPLE_SIZE = 200   # coreset size grws linear w/ SAMPLE_SIZE

        def stream():
            streamer = CoresetStreamer(SAMPLE_SIZE, k, config)
            coreset = streamer.stream(L)
            L1 = coreset[0]
            
            _, B, _ = CorsetForKMeansForLines(config).coreset(L1, k, int(L1.get_size()*0.6), True)
            cwc = CoresetForWeightedCenters(config)
            B = cwc.coreset(B, k, m)

            X_klines = L.get_projected_centers(B)    
            kl_labels = L.get_indices_clusters(B)
            
            return X_klines, kl_labels

        ITER = 5
        klines_mse = np.zeros(ITER)
        scores = [[]] * ITER
        for i in range(ITER):
            X_klines, kl_labels = stream()
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Scores: {np.array(scores).mean(axis=0)}")

        assert mse_sklearn / klines_mse.mean() > 0.2


if __name__ == '__main__':
    unittest.main()

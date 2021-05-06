#!/usr/bin/env python
# coding: utf-8

import unittest
from tests.utils import create_incomplete_matrix, kmeans_missing, ParameterConfig, customStreamer
import numpy as np
import math
from sklearn.datasets import make_blobs, make_gaussian_quantiles, make_classification, make_circles, make_hastie_10_2
from sklearn import metrics
from klines import SetOfLines


class TestKMeansSynthetic(unittest.TestCase):
    def test_synthetic_high_dev(self):
        print('''
            high cluster deviation
            #features = #classes
        ''')
        N = 10**3
        d = 4
        k = 4
        X, y = make_blobs(n_samples=N, n_features=d, centers=k, cluster_std=4.0)

        X_incomplete = create_incomplete_matrix(X)
        labels, _, X_hat = kmeans_missing(X_incomplete, k)

        sklearn_mse = ((X - X_hat)**2).mean()
        score = metrics.homogeneity_completeness_v_measure(labels, y)
        print(f'sklearn mse: {sklearn_mse}')
        print(f'sklearn scores: {score}')

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

        SAMPLE_SIZE = 50   

        ITER = 5
        klines_mse = np.zeros(ITER)
        scores = [[]] * ITER
        for i in range(ITER):
            print(f'Running KLines iter {i+1} of {ITER}')
            X_klines, kl_labels = customStreamer(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Klines scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse / klines_mse.mean() > 0.3


    def test_synthetic_high_dimension(self):
        print('''
            data with #features ~ 2x #classes
        ''')
        N = 10**3
        d = 8
        k = 4
        X, y = make_blobs(n_samples=N, n_features=d, centers=k)

        X_incomplete = create_incomplete_matrix(X)
        labels, _, X_hat = kmeans_missing(X_incomplete, k)

        sklearn_mse = ((X - X_hat)**2).mean()
        score = metrics.homogeneity_completeness_v_measure(labels, y)
        print(f'sklearn mse: {sklearn_mse}')
        print(f'sklearn scores: {score}')

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

        SAMPLE_SIZE = 50   

        ITER = 5
        klines_mse = np.zeros(ITER)
        scores = [[]] * ITER
        for i in range(ITER):
            print(f'Running KLines iter {i+1} of {ITER}')
            X_klines, kl_labels = customStreamer(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Klines scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse / klines_mse.mean() > 0.1


    def test_synthetic_gaussian(self):
        print('''
            data is Gaussian, divided into four quantiles
        ''')
        N = 10**3
        d = 4
        k = 4
        X, y = make_gaussian_quantiles(n_features=d, n_samples=N, n_classes=k)

        X_incomplete = create_incomplete_matrix(X)
        labels, _, X_hat = kmeans_missing(X_incomplete, k)

        sklearn_mse = ((X - X_hat)**2).mean()
        score = metrics.homogeneity_completeness_v_measure(labels, y)
        print(f'sklearn mse: {sklearn_mse}')
        print(f'sklearn scores: {score}')

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

        SAMPLE_SIZE = 50   # keep it < 100, works fast
                           

        ITER = 5
        klines_mse = np.zeros(ITER)
        scores = [[]] * ITER
        for i in range(ITER):
            print(f'Running KLines iter {i+1} of {ITER}')
            X_klines, kl_labels = customStreamer(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Klines scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse / klines_mse.mean() > 0.5


    def test_synthetic_zero_red(self):
        print('''
            data has no redundant features, but only half of the features are informative
        ''')
        N = 10**3
        d = 8
        k = 4
        X, y = make_classification(n_samples=N, n_features=d, n_redundant=0, n_informative=d//2, n_clusters_per_class=1, n_classes=k) 

        X_incomplete = create_incomplete_matrix(X)
        labels, _, X_hat = kmeans_missing(X_incomplete, k)

        sklearn_mse = ((X - X_hat)**2).mean()
        score = metrics.homogeneity_completeness_v_measure(labels, y)
        print(f'sklearn mse: {sklearn_mse}')
        print(f'sklearn scores: {score}')

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

        SAMPLE_SIZE = 50   # keep it < 100, works fast
                           

        ITER = 5
        klines_mse = np.zeros(ITER)
        scores = [[]] * ITER
        for i in range(ITER):
            print(f'Running KLines iter {i+1} of {ITER}')
            X_klines, kl_labels = customStreamer(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Klines scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse / klines_mse.mean() > 0.5


    def test_synthetic_many_classes(self):
        print('''
            data with #classes ~ 2x #features
        ''')
        N = 10**3
        d = 4
        k = 8
        X, y = make_blobs(n_samples=N, n_features=d, centers=k) 

        X_incomplete = create_incomplete_matrix(X)
        labels, _, X_hat = kmeans_missing(X_incomplete, k)

        sklearn_mse = ((X - X_hat)**2).mean()
        score = metrics.homogeneity_completeness_v_measure(labels, y)
        print(f'sklearn mse: {sklearn_mse}')
        print(f'sklearn scores: {score}')

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

        SAMPLE_SIZE = 50   # keep it < 100, works fast
                           

        ITER = 5
        klines_mse = np.zeros(ITER)
        scores = [[]] * ITER
        for i in range(ITER):
            print(f'Running KLines iter {i+1} of {ITER}')
            X_klines, kl_labels = customStreamer(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Klines scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse / klines_mse.mean() > 0.5


    def test_synthetic_circles(self):
        print('''
            two concentric circles
        ''')
        N = 10**3
        X, y = make_circles(n_samples=N, noise=1.0) 
        k = len(np.unique(y))

        X_incomplete = create_incomplete_matrix(X)
        labels, _, X_hat = kmeans_missing(X_incomplete, k)

        sklearn_mse = ((X - X_hat)**2).mean()
        score = metrics.homogeneity_completeness_v_measure(labels, y)
        print(f'sklearn mse: {sklearn_mse}')
        print(f'sklearn scores: {score}')

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

        SAMPLE_SIZE = 50   # keep it < 100, works fast
                           

        ITER = 5
        klines_mse = np.zeros(ITER)
        scores = [[]] * ITER
        for i in range(ITER):
            print(f'Running KLines iter {i+1} of {ITER}')
            X_klines, kl_labels = customStreamer(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Klines scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse / klines_mse.mean() > 0.5


    def test_synthetic_hastie(self):
        print('''
            binary classification data
            used in Hastie et al. 2009
        ''')
        N = 10**3
        X, y = make_hastie_10_2(n_samples=N) 
        k = len(np.unique(y))

        X_incomplete = create_incomplete_matrix(X)
        labels, _, X_hat = kmeans_missing(X_incomplete, k)

        sklearn_mse = ((X - X_hat)**2).mean()
        score = metrics.homogeneity_completeness_v_measure(labels, y)
        print(f'sklearn mse: {sklearn_mse}')
        print(f'sklearn scores: {score}')

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

        SAMPLE_SIZE = 50   # keep it < 100, works fast
                           

        ITER = 5
        klines_mse = np.zeros(ITER)
        scores = [[]] * ITER
        for i in range(ITER):
            print(f'Running KLines iter {i+1} of {ITER}')
            X_klines, kl_labels = customStreamer(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Klines scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse / klines_mse.mean() > 0.5

#!/usr/bin/env python
# coding: utf-8

import unittest
from tests.utils import create_incomplete_matrix, kmeans_missing, ParameterConfig
import numpy as np
import math
from sklearn import metrics
from klines import SetOfLines, CoresetForWeightedCenters, CorsetForKMeansForLines, CoresetStreamer


def stream(L, k, m, SAMPLE_SIZE, config):
    streamer = CoresetStreamer(SAMPLE_SIZE, k, config)
    # note this is O(sample^2 * n) + O(m)
    coreset = streamer.stream(L)
    L1 = coreset[0]
    
    _, B, _ = CorsetForKMeansForLines(config).coreset(L1, k, int(L1.get_size()*0.6), True)
    cwc = CoresetForWeightedCenters(config)
    B = cwc.coreset(B, k, m)

    X_klines = L.get_projected_centers(B)    
    kl_labels = L.get_indices_clusters(B)
    
    return X_klines, kl_labels


class TestClusteringBenchmarks(unittest.TestCase):
    '''
        benchmark datasets chosen from
        https://www.sciencedirect.com/science/article/pii/S2352340920303954
    '''

    def test_benchmark_Atom(self):
        print('Clustering Atom.npz')
        npzfile = np.load('data/Atom.npz')
        X, y = npzfile['X'], npzfile['y']
        (N, _), k = X.shape, np.unique(y).shape[0]
        print(f'#Datapoints {N}')

        X_incomplete = create_incomplete_matrix(X)
        labels, _, X_hat = kmeans_missing(X_incomplete, k)

        sklearn_mse = ((X - X_hat)**2).mean()
        score = metrics.homogeneity_completeness_v_measure(labels, y)
        print(f'MSE sklearn: {sklearn_mse}')
        print(f'MSE scores/measures: {score}')

        displacements = np.nan_to_num(X_incomplete)

        spans = np.nan_to_num(X_incomplete)
        spans[spans==0] = 1
        spans[spans!=1] = 0

        L = SetOfLines(spans, displacements, np.ones(N), np.ones(N))            
        config = ParameterConfig()

        ## data
        m = 60  # coreset size ~ reduction ratio
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
            X_klines, kl_labels = stream(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse / klines_mse.mean() > 0.4


    def test_benchmark_Chainlink(self):
        print('Clustering Chainlink.npz')
        npzfile = np.load('data/Chainlink.npz')
        X, y = npzfile['X'], npzfile['y']
        (N, _), k = X.shape, np.unique(y).shape[0]
        print(f'#Datapoints {N}')

        X_incomplete = create_incomplete_matrix(X)
        labels, _, X_hat = kmeans_missing(X_incomplete, k)

        sklearn_mse = ((X - X_hat)**2).mean()
        score = metrics.homogeneity_completeness_v_measure(labels, y)
        print(f'MSE sklearn: {sklearn_mse}')
        print(f'MSE scores/measures: {score}')

        displacements = np.nan_to_num(X_incomplete)

        spans = np.nan_to_num(X_incomplete)
        spans[spans==0] = 1
        spans[spans!=1] = 0

        L = SetOfLines(spans, displacements, np.ones(N), np.ones(N))            
        config = ParameterConfig()

        ## data
        m = 60  # coreset size ~ reduction ratio
        tau = 1e-2

        config.a_b_approx_minimum_number_of_lines = 40 # constant 100, line 2, algo 2 BI-CRITERIA
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
            X_klines, kl_labels = stream(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse / klines_mse.mean() > 0.8


    def test_benchmark_EngyTime(self):
        print('Clustering EngyTime.npz')
        npzfile = np.load('data/EngyTime.npz')
        X, y = npzfile['X'], npzfile['y']
        (N, _), k = X.shape, np.unique(y).shape[0]
        print(f'#Datapoints {N}')

        X_incomplete = create_incomplete_matrix(X)
        labels, _, X_hat = kmeans_missing(X_incomplete, k)

        sklearn_mse = ((X - X_hat)**2).mean()
        score = metrics.homogeneity_completeness_v_measure(labels, y)
        print(f'MSE sklearn: {sklearn_mse}')
        print(f'MSE scores/measures: {score}')

        displacements = np.nan_to_num(X_incomplete)

        spans = np.nan_to_num(X_incomplete)
        spans[spans==0] = 1
        spans[spans!=1] = 0

        L = SetOfLines(spans, displacements, np.ones(N), np.ones(N))            
        config = ParameterConfig()

        ## data
        m = 120  # coreset size ~ reduction ratio
        tau = 1e-2

        config.a_b_approx_minimum_number_of_lines = 100 # constant 100, line 2, algo 2 BI-CRITERIA
        config.sample_size_for_a_b_approx = int(m*1.05) # |S| >= m, line 3 of algo 2
                                                        # note: there'll be a O(|S|^2) cost while computing algo 1
        config.farthest_to_centers_rate_in_a_b_approx = 4/11  # opp of 7/11, line 6, algo 2 BI-CRITERIA
        config.number_of_remains_multiply_factor = int(math.log(N))//k # this is `b` in algo 2, other paper, set as random here -  how to calculate it?
        config.closest_to_median_rate = (1-tau)/(2*k)  # refer line 4, algo 1, other paper
        config.median_sample_size = int(N*0.05)   # size of q_i, line 3, algo 2, other paper
        config.max_sensitivity_multiply_factor = 100  # for outliers in coresets
        config.number_of_remains = 100

        SAMPLE_SIZE = 60   

        ITER = 5
        klines_mse = np.zeros(ITER)
        scores = [[]] * ITER
        for i in range(ITER):
            print(f'Running KLines iter {i+1} of {ITER}')
            X_klines, kl_labels = stream(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse / klines_mse.mean() > 0.5


    def test_benchmark_Hepta(self):
        print('Clustering Hepta.npz')
        npzfile = np.load('data/Hepta.npz')
        X, y = npzfile['X'], npzfile['y']
        (N, _), k = X.shape, np.unique(y).shape[0]
        print(f'#Datapoints {N}')

        X_incomplete = create_incomplete_matrix(X)

        ITER = 5
        sklearn_mse = np.zeros(ITER)
        sklearn_scores = [[]] * ITER
        for i in range(ITER):
            labels, _, X_hat = kmeans_missing(X_incomplete, k)
            sklearn_mse[i] = ((X - X_hat)**2).mean()
            sklearn_scores[i] = metrics.homogeneity_completeness_v_measure(labels, y)
        
        print(f'MSE sklearn: {sklearn_mse.mean()}')
        print(f'MSE scores/measures: {np.array(sklearn_scores).mean(axis=0)}')

        displacements = np.nan_to_num(X_incomplete)

        spans = np.nan_to_num(X_incomplete)
        spans[spans==0] = 1
        spans[spans!=1] = 0

        L = SetOfLines(spans, displacements, np.ones(N), np.ones(N))            
        config = ParameterConfig()

        ## data
        m = int(N*0.6)  # coreset size ~ reduction ratio
        tau = 1e-2

        config.a_b_approx_minimum_number_of_lines = 50 # constant 100, line 2, algo 2 BI-CRITERIA
        config.sample_size_for_a_b_approx = int(m*1.05) # |S| >= m, line 3 of algo 2
                                                        # note: there'll be a O(|S|^2) cost while computing algo 1
        config.farthest_to_centers_rate_in_a_b_approx = 4/11  # opp of 7/11, line 6, algo 2 BI-CRITERIA
        config.number_of_remains_multiply_factor = int(math.log(N))//k # this is `b` in algo 2, other paper, set as random here -  how to calculate it?
        config.closest_to_median_rate = (1-tau)/(2*k)  # refer line 4, algo 1, other paper
        config.median_sample_size = int(N*0.05)   # size of q_i, line 3, algo 2, other paper
        config.max_sensitivity_multiply_factor = 50  # for outliers in coresets
        config.number_of_remains = 20

        SAMPLE_SIZE = 50   

        klines_mse = np.zeros(ITER)
        scores = [[]] * ITER
        for i in range(ITER):
            print(f'Running KLines iter {i+1} of {ITER}')
            X_klines, kl_labels = stream(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse.mean() / klines_mse.mean() > 0.4


    def test_benchmark_Tetra(self):
        print('Clustering Tetra.npz')
        npzfile = np.load('data/Tetra.npz')
        X, y = npzfile['X'], npzfile['y']
        (N, _), k = X.shape, np.unique(y).shape[0]
        print(f'#Datapoints {N}')

        X_incomplete = create_incomplete_matrix(X)

        ITER = 5
        sklearn_mse = np.zeros(ITER)
        sklearn_scores = [[]] * ITER
        for i in range(ITER):
            labels, _, X_hat = kmeans_missing(X_incomplete, k)
            sklearn_mse[i] = ((X - X_hat)**2).mean()
            sklearn_scores[i] = metrics.homogeneity_completeness_v_measure(labels, y)
        
        print(f'MSE sklearn: {sklearn_mse.mean()}')
        print(f'MSE scores/measures: {np.array(sklearn_scores).mean(axis=0)}')

        displacements = np.nan_to_num(X_incomplete)

        spans = np.nan_to_num(X_incomplete)
        spans[spans==0] = 1
        spans[spans!=1] = 0

        L = SetOfLines(spans, displacements, np.ones(N), np.ones(N))            
        config = ParameterConfig()

        ## data
        m = int(N*0.6)  # coreset size ~ reduction ratio
        tau = 1e-2

        config.a_b_approx_minimum_number_of_lines = 50 # constant 100, line 2, algo 2 BI-CRITERIA
        config.sample_size_for_a_b_approx = int(m*1.05) # |S| >= m, line 3 of algo 2
                                                        # note: there'll be a O(|S|^2) cost while computing algo 1
        config.farthest_to_centers_rate_in_a_b_approx = 4/11  # opp of 7/11, line 6, algo 2 BI-CRITERIA
        config.number_of_remains_multiply_factor = int(math.log(N))//k # this is `b` in algo 2, other paper, set as random here -  how to calculate it?
        config.closest_to_median_rate = (1-tau)/(2*k)  # refer line 4, algo 1, other paper
        config.median_sample_size = int(N*0.05)   # size of q_i, line 3, algo 2, other paper
        config.max_sensitivity_multiply_factor = 50  # for outliers in coresets
        config.number_of_remains = 20

        SAMPLE_SIZE = 50   

        klines_mse = np.zeros(ITER)
        scores = [[]] * ITER
        for i in range(ITER):
            print(f'Running KLines iter {i+1} of {ITER}')
            X_klines, kl_labels = stream(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse.mean() / klines_mse.mean() > 0.6


    def test_benchmark_TwoDiamonds(self):
        print('Clustering TwoDiamonds.npz')
        npzfile = np.load('data/TwoDiamonds.npz')
        X, y = npzfile['X'], npzfile['y']
        (N, _), k = X.shape, np.unique(y).shape[0]
        print(f'#Datapoints {N}')

        X_incomplete = create_incomplete_matrix(X)

        ITER = 5
        sklearn_mse = np.zeros(ITER)
        sklearn_scores = [[]] * ITER
        for i in range(ITER):
            labels, _, X_hat = kmeans_missing(X_incomplete, k)
            sklearn_mse[i] = ((X - X_hat)**2).mean()
            sklearn_scores[i] = metrics.homogeneity_completeness_v_measure(labels, y)
        
        print(f'MSE sklearn: {sklearn_mse.mean()}')
        print(f'MSE scores/measures: {np.array(sklearn_scores).mean(axis=0)}')

        displacements = np.nan_to_num(X_incomplete)

        spans = np.nan_to_num(X_incomplete)
        spans[spans==0] = 1
        spans[spans!=1] = 0

        L = SetOfLines(spans, displacements, np.ones(N), np.ones(N))            
        config = ParameterConfig()

        ## data
        m = int(N*0.1)  # coreset size ~ reduction ratio
        tau = 1e-2

        config.a_b_approx_minimum_number_of_lines = 50 # constant 100, line 2, algo 2 BI-CRITERIA
        config.sample_size_for_a_b_approx = int(m*1.05) # |S| >= m, line 3 of algo 2
                                                        # note: there'll be a O(|S|^2) cost while computing algo 1
        config.farthest_to_centers_rate_in_a_b_approx = 4/11  # opp of 7/11, line 6, algo 2 BI-CRITERIA
        config.number_of_remains_multiply_factor = int(math.log(N))//k # this is `b` in algo 2, other paper, set as random here -  how to calculate it?
        config.closest_to_median_rate = (1-tau)/(2*k)  # refer line 4, algo 1, other paper
        config.median_sample_size = int(N*0.05)   # size of q_i, line 3, algo 2, other paper
        config.max_sensitivity_multiply_factor = 20  # for outliers in coresets
        config.number_of_remains = 20

        SAMPLE_SIZE = 50   

        klines_mse = np.zeros(ITER)
        scores = [[]] * ITER
        for i in range(ITER):
            print(f'Running KLines iter {i+1} of {ITER}')
            X_klines, kl_labels = stream(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse.mean() / klines_mse.mean() > 0.3


    def test_benchmark_Lsun3D(self):
        print('Clustering Lsun3D.npz')
        npzfile = np.load('data/Lsun3D.npz')
        X, y = npzfile['X'], npzfile['y']
        (N, _), k = X.shape, np.unique(y).shape[0]
        print(f'#Datapoints {N}')

        X_incomplete = create_incomplete_matrix(X)

        ITER = 5
        sklearn_mse = np.zeros(ITER)
        sklearn_scores = [[]] * ITER
        for i in range(ITER):
            labels, _, X_hat = kmeans_missing(X_incomplete, k)
            sklearn_mse[i] = ((X - X_hat)**2).mean()
            sklearn_scores[i] = metrics.homogeneity_completeness_v_measure(labels, y)
        
        print(f'MSE sklearn: {sklearn_mse.mean()}')
        print(f'MSE scores/measures: {np.array(sklearn_scores).mean(axis=0)}')

        displacements = np.nan_to_num(X_incomplete)

        spans = np.nan_to_num(X_incomplete)
        spans[spans==0] = 1
        spans[spans!=1] = 0

        L = SetOfLines(spans, displacements, np.ones(N), np.ones(N))            
        config = ParameterConfig()

        ## data
        m = int(N*0.5)  # coreset size ~ reduction ratio
        tau = 1e-2

        config.a_b_approx_minimum_number_of_lines = 50 # constant 100, line 2, algo 2 BI-CRITERIA
        config.sample_size_for_a_b_approx = int(m*1.05) # |S| >= m, line 3 of algo 2
                                                        # note: there'll be a O(|S|^2) cost while computing algo 1
        config.farthest_to_centers_rate_in_a_b_approx = 4/11  # opp of 7/11, line 6, algo 2 BI-CRITERIA
        config.number_of_remains_multiply_factor = int(math.log(N))//k # this is `b` in algo 2, other paper, set as random here -  how to calculate it?
        config.closest_to_median_rate = (1-tau)/(2*k)  # refer line 4, algo 1, other paper
        config.median_sample_size = int(N*0.05)   # size of q_i, line 3, algo 2, other paper
        config.max_sensitivity_multiply_factor = 20  # for outliers in coresets
        config.number_of_remains = 10

        SAMPLE_SIZE = 50   

        klines_mse = np.zeros(ITER)
        scores = [[]] * ITER
        for i in range(ITER):
            print(f'Running KLines iter {i+1} of {ITER}')
            X_klines, kl_labels = stream(L, k, m, SAMPLE_SIZE, config)
            klines_mse[i] = ((X - X_klines)**2).mean()
            scores[i] = metrics.homogeneity_completeness_v_measure(kl_labels, y)

        print(f"Klines MSE: {klines_mse.mean()}")
        print(f"Scores: {np.array(scores).mean(axis=0)}")

        assert sklearn_mse.mean() / klines_mse.mean() > 0.5

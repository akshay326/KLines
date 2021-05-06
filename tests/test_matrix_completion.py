# coding: utf-8

import numpy as np
import math
from klines import SetOfLines, CorsetForKMeansForLines, CoresetStreamer
from tests.utils import create_incomplete_matrix, ParameterConfig
import unittest


class TestMatrixCompletion(unittest.TestCase):

    def complete_matrix(self, N, inner_rank, d):
        # create synthetic data
        X = np.dot(np.random.randn(N, inner_rank), np.random.randn(inner_rank, d))
        MSE = (X ** 2).mean()

        X_incomplete = create_incomplete_matrix(X)

        displacements = np.nan_to_num(X_incomplete)
        spans = np.nan_to_num(X_incomplete)
        spans[spans==0] = 1
        spans[spans!=1] = 0

        L = SetOfLines(spans, displacements, np.ones(N), np.ones(N))

        config = ParameterConfig()

        ## data
        k = 2*d
        m = 100  # coreset size ~ reduction ratio
        tau = 1e-2

        config.a_b_approx_minimum_number_of_lines = 100 # constant 100, line 2, algo 2 BI-CRITERIA

        config.sample_size_for_a_b_approx = int(m*1.05) # |S| >= m, line 3 of algo 2
                                                        # note: there'll be a O(|S|^2) cost while computing algo 1
            
        config.farthest_to_centers_rate_in_a_b_approx = 4/11  # opp of 7/11, line 6, algo 2 BI-CRITERIA
        config.number_of_remains_multiply_factor = int(math.log(N))//k # this is `b` in algo 2, other paper, set as random here -  how to calculate it?
        config.closest_to_median_rate = (1-tau)/(2*k)  # refer line 4, algo 1, other paper

        config.median_sample_size = int(N*0.05)    # size of q_i, line 3, algo 2, other paper
        config.max_sensitivity_multiply_factor = 100  # for outliers in coresets

        config.number_of_remains = 20

        SAMPLE_SIZE = int(N*0.5)   

        streamer = CoresetStreamer(SAMPLE_SIZE, k, config)
        coreset = streamer.stream(L)
        L1 = coreset[0]

        _, B, _ = CorsetForKMeansForLines(config).coreset(L1, k, int(L1.get_size()*0.8), True)

        X_klines = L.get_projected_centers(B)
        klines_mse = ((X - X_klines)**2).mean()

        return klines_mse/MSE


    def test_accuracy(self):        
        ITERS = 10

        # problem size
        N = 100
        d = 5
        inner_rank = N//2

        mse_ratios = np.array([self.complete_matrix(N, inner_rank, d) for _ in range(ITERS)])

        assert np.average(mse_ratios) < 0.5

  

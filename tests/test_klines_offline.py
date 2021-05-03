#!/usr/bin/env python

import timeit
import numpy as np
import math
from klines import SetOfLines, CorsetForKMeansForLines, CoresetForWeightedCenters
import unittest


class TestOfflineClusteringSpeed(unittest.TestCase):
    def test_speed(self):
        ## data
        k = 3
        N = int(1e3)
        m = int(N*0.07)  # coreset size ~ reduction ratio
        tau = 1e-3

        straight_roads = np.load('data/road_segments_china.npy')
        straight_roads = straight_roads[np.random.choice(straight_roads.shape[0], N, replace=False)]
        L = [[x[0][0], x[0][1], x[1][0], x[1][1]] for x in straight_roads]

        ## construct set of lines
        L = SetOfLines([], [], [], [], L, True)

        class ParameterConfig:
            def __init__(self):
                pass

        config = ParameterConfig()
        config.a_b_approx_minimum_number_of_lines = int(N*0.01) # constant 100, line 2, algo 2 BI-CRITERIA
        config.sample_size_for_a_b_approx = int(m*1.01) # |S| >= m, line 3 of algo 2
                                                        # note: there'll be a O(|S|^2) cost while computing algo 1
        config.farthest_to_centers_rate_in_a_b_approx = 4.0/11  # opp of 7/11, line 6, algo 2 BI-CRITERIA
        config.number_of_remains_multiply_factor = int(math.log(N))//k # this is `b` in algo 2, other paper, set as random here -  how to calculate it?
        config.closest_to_median_rate = (1-tau)/(2*k)  # refer line 4, algo 1, other paper
        config.median_sample_size = int(N*0.05)    # size of q_i, line 3, algo 2, other paper
        config.max_sensitivity_multiply_factor = 50  # for outliers in coresets


        coreset = CorsetForKMeansForLines(config)


        # ## statistical analysis
        ###  MEAN AND VAR EVALUATION
        ITER = 2
        errors = np.array([coreset.coreset(L, k, m, True)[2]  for _ in range(ITER)])
        print(f"mean: {errors.mean()}")
        print(f"var: {errors.var()}")

        ## more tau => more variance
        ## more max_sensitivity_multiply_factor => less variance
        ## kept median_sample_size small, ~5% of N, coz coresets candidate set progressively reduces

        # note size of B will be ~ O(log(n) * m^2)
        # and ofcourse its not K-center
        _, B, _ = coreset.coreset(L, k, m, True)
        config.number_of_remains = int(math.log(B.get_size())) # this is also `b`, line 1, algo 2, other paper
                                                            # value copied from `recursive_robust_median` method
            
        cwc = CoresetForWeightedCenters(config)

        ### FOR TIME EVALUATION        
        X = []
        ITER = 2
        for _ in range(ITER):
            st = timeit.default_timer()
            cwc.coreset(B, k, m)
            X.append(timeit.default_timer() - st)
            
        X = np.array(X)
        print(f"Mean time taken for {ITER} calls is {X.mean()}s")

        assert X.mean() < 10

        import cProfile
        pr = cProfile.Profile()
        pr.enable()

        cwc.coreset(B, k, m)

        pr.disable()
        pr.print_stats(sort='cumtime')



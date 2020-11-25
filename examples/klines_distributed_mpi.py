#!/usr/bin/env python

from mpi4py import MPI

# to use klines
import sys
sys.path.insert(1, "./KLines")

import numpy as np
import copy
import math

from klines import SetOfLines, SetOfPoints, CorsetForKMeansForLines, CoresetForWeightedCenters, CoresetStreamer, CoresetNode

MASTER = 0
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
p = comm.Get_size()


#################################### Klines specific code
k = 3
d = 2
N_tot = int(1e4)
N = N_tot // p
m = max(100, int(N/1e2))  # coreset size ~ reduction ratio
tau = 1e-3


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

config.number_of_remains = 100
#################################### Klines specific code


if rank == MASTER:
    sendbuf = np.load('road_segments_china.npy') # 8.7e5 entries
    sendbuf = sendbuf[:N_tot]
    sendbuf = np.array_split(sendbuf, p)
else:
    sendbuf = None
        
# distribute
recbuf = comm.scatter(sendbuf, root=MASTER)

# now every process (including MASTER) has equal sized L ~ N/p
L = [[x[0][0], x[0][1], x[1][0], x[1][1]] for x in recbuf]
L = SetOfLines([], [], [], [], L, True)


# define the streamer
SAMPLE_SIZE = 50
streamer = CoresetStreamer(SAMPLE_SIZE, N, k, config)

coreset = streamer.stream(L)
print(f"PID {rank}. Time taken: {coreset[2]-coreset[1]}s. Lines size: {coreset[0].get_size()}")

# now MASTER gathers a list of coresets
lines = comm.gather(coreset[0], root=MASTER)

if rank == MASTER:
    ckl = CorsetForKMeansForLines(config)
    cwc = CoresetForWeightedCenters(config)
    L = lines[0]
    B = SetOfPoints()
    
    for i in range(1, len(lines)):
        L.add_set_of_lines(lines[i])
        L, Bi, _ = ckl.coreset(L=L, k=k, m=SAMPLE_SIZE, offline=True)
        B.add_set_of_points(Bi) 
        B = cwc.coreset(P=B, k=k, m=SAMPLE_SIZE)
    
    print(f"Total cost: {L.get_sum_of_distances_to_centers(B)}")
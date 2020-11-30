#################################################################
#     Corset for Weighted centers of points                     #
#     Paper: http://people.csail.mit.edu/dannyf/outliers.pdf    #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################

from __future__ import division

import time

import numpy as np

from klines.coreset_for_k_means_for_lines import CorsetForKMeansForLines
from klines.set_of_lines import SetOfLines

#from parameters_config import ParameterConfig
#from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute


class TreeNode:
    def __init__(self, L = SetOfLines(), rank = 1):
        self.rank = rank
        self.lines = L


"""
Class that performs the full streaming operation. Each step read m points from set of lines `L`, compress it and add it to the coreset tree.
Attributes:
    stack (list): A list that simulates the streaming comprassion tree operations
    m (int): size of chunk
    co (int): flag/number of points for read
    eps (float): error parameter
    delta (float): failure probability
"""


class CoresetStreamer:

    def __init__(self, sample_size, k, parameters_config):

        self.stack = []
        self.k = k
        self.sample_size = sample_size
        self.parameters_config = parameters_config

    ######################################################################

    def stream(self, L):
        """
        The method start to get in a streaming points set of lines `L`
        TODO: complete parameteres
        """
        coreset_starting_time = time.time()
        batch_size = self.sample_size*2
        curr_index = 0
        N = L.get_size()
        while curr_index < N:
#             if curr_index % 100 == 0:
#                 print(f"Lines read so far: {curr_index}")
            if batch_size > N - curr_index:
                self.add_to_tree(L.get_lines_at_indexes_interval(curr_index, N))
                break
            current_batch = L.get_lines_at_indexes_interval(curr_index, curr_index + batch_size)
            self.add_to_tree(current_batch)
            curr_index += batch_size

        # merge all nodes until single left
        while len(self.stack) > 1:
            node1 = self.stack.pop()
            node2 = self.stack.pop()
            new_node = self.merge_two_nodes(node1, node2)
            self.stack.append(new_node)
            
        C = self.stack[0].lines
        coreset_ending_time = time.time()
        return C, coreset_starting_time, coreset_ending_time

    ######################################################################

    def add_to_tree(self, L):
        """
            L is the current batch of lines
        """
        L_size = L.get_size()

        # compress only if batch size > samples
        if L_size > self.sample_size:
            coreset = CorsetForKMeansForLines(self.parameters_config).coreset(L=L, k=self.k, m=self.sample_size)
            current_node = TreeNode(coreset)
        else:
            current_node = TreeNode(L)
            
        if len(self.stack) == 0:
            self.stack.append(current_node)
            return

        stack_top_node = self.stack[-1]
        if stack_top_node.rank != current_node.rank:
            self.stack.append(current_node)
            return
        else:
            while stack_top_node.rank == current_node.rank: #TODO: take care for the case they are not equal, currently the node deosn't appanded to the tree
                self.stack.pop()
                current_node = self.merge_two_nodes(current_node, stack_top_node)
                if len(self.stack) == 0:
                    self.stack.append(current_node)
                    return
                stack_top_node = self.stack[-1]
                if stack_top_node.rank != current_node.rank:
                    self.stack.append(current_node)
                    return

    ######################################################################

    def merge_two_nodes(self, node1, node2):
        """
        Get nodes of the coreset tree, merge, and return the coreset of the merged nodes
        """
        L1 = node1.lines
        L2 = node2.lines
        L1.add_set_of_lines(L2)
        coreset = CorsetForKMeansForLines(self.parameters_config).coreset(L=L1, k=self.k, m=self.sample_size)
        return TreeNode(coreset, node1.rank+1)

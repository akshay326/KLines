#################################################################
#     Corset for Weighted centers of points                     #
#     Paper: http://people.csail.mit.edu/dannyf/outliers.pdf    #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################



from __future__ import division

import copy
import random

import numpy as np


class SetOfPoints:
    """
    Class that represent a set of weighted points in any d>0 dimensional space
    Attributes:
        points (ndarray) : The points in the set
        weights (ndarray) : The weights. weights[i] is the weight of the i-th point
        dim (integer): The dimension of the points
    """

    ##################################################################################

    def __init__(self, P=None, w=None, sen=None, indexes = None):
        """
        C'tor
        :param P: np.ndarray - set of points
        :param w: np.ndarray - set of weights
        :param sen: np.ndarray - set of sensitivities
        """
        #if (indexes != [] and len(P) == 0) or (indexes == [] and len(P) != 0):
        #    assert indexes != [] and len(P) != 0, "not indexes == [] and len(P) == 0"

        if P is None:
            P = []
        if w is None:
            w = []
        if sen is None:
            sen = []
        if indexes is None:
            indexes = []

        size = len(P)
        if size == 0:  # there is no points in the set we got
            self.points = []
            self.weights = []
            self.sensitivities = []
            self.dim = 0
            self.indexes = []
            return
        if np.ndim(P) == 1:  # there is only one point in the array
            Q = []
            Q.append(P)
            self.points = np.asarray(Q)
            if w == []:
                w = np.ones((1, 1), dtype=np.float)
            if sen == []:
                sen = np.ones((1, 1), dtype=np.float)
            self.weights = w
            self.sensitivities = sen
            [_, self.dim] = np.shape(self.points)
            self.indexes = np.zeros((1, 1), dtype=np.float)
            return
        else:
            self.points = np.asarray(P)
        [_, self.dim] = np.shape(self.points)
        if w == []:
            w = np.ones((size, 1), dtype=np.float)
        if sen == []:
            sen = np.ones((size, 1), dtype=np.float)
        self.weights = w
        self.sensitivities = sen
        if indexes == []:
            self.indexes = np.asarray(range(len(self.points))).reshape(-1)
        else:
            self.indexes = indexes.reshape(-1)

    ##################################################################################

    def get_sample_of_points(self, size_of_sample):
        """
        Args:
            size_of_sample (int) : the sample's size

        Returns:
            SetOfPoints: sample consist of size_of_sample points from the uniform distribution over the set
        """

        assert size_of_sample > 0, "size_of_sample <= 0"

        size = self.get_size()
        if size_of_sample >= size:
            return self
        else:
            all_indices = np.asarray(range(size))
            sample_indices = np.random.choice(all_indices, size_of_sample).tolist()
            sample_points = np.take(self.points, sample_indices, axis=0, out=None, mode='raise')
            sample_weights = np.take(self.weights, sample_indices, axis=0, out=None, mode='raise')
            sample_indexes = np.take(self.indexes, sample_indices, axis=0, out=None, mode='raise')
            return SetOfPoints(sample_points, sample_weights,indexes=sample_indexes)

    ##################################################################################

    def get_size(self):
        """
        Returns:
            int: number of points in the set
        """

        return np.shape(self.points)[0]

    ##################################################################################

    def get_points_from_indices(self, indices):
        """
        Args:
            indices (list of ints) : list of indices.

        Returns:
            SetOfPoints: a set of point that contains the points in the input indices
        """
        assert len(self.get_size()) > 0, "no points to select"
        assert len(indices) > 0, "indices length is zero"


        sample_points = self.points[indices]
        sample_weights = self.weights[indices]
        sample_indexes = self.indexes[indices]

        return SetOfPoints(sample_points, sample_weights, indexes=sample_indexes)


    def get_sum_of_weights(self):
        """
        Returns:
            float: the sum of wights in the set
        """

        assert self.get_size() > 0, "No points in the set"

        return np.sum(self.weights)

    ##################################################################################

    def add_set_of_points(self, P):
        """
        The method adds a set of weighted points to the set
        Args:
            P (SetOfPoints) : a set of points to add to the set

        Returns:
            ~
        """

        if P.get_size() == 0:
            return

        points = P.points
        weights = P.weights.reshape(-1, 1)
        sensitivities = P.sensitivities.reshape(-1, 1)
        indexes = P.indexes.reshape(-1)

        size = self.get_size()
        if size == 0 and self.dim == 0:
            self.dim = np.shape(points)[1]
            self.points = points
            self.weights = weights
            self.sensitivities = sensitivities
            self.indexes = indexes
            return

        self.points = np.append(self.points, points, axis=0)
        self.weights = np.append(self.weights, weights)
        self.sensitivities = np.append(self.sensitivities, sensitivities, axis=0)
        self.indexes = np.append(self.indexes, indexes, axis=0)


    def remove_from_set(self, C):
        """
        The method gets set of points C and remove each point in the set that also in C
        Args:
            C (SetOfPoints) : a set of points to remove from the set

        Returns:
            ~
        """

        indexes = []
        self_indexes = self.indexes
        for i in range(len(self_indexes)):
            index = self_indexes[i]
            if index in C.indexes:
                indexes.append(i)

        self.points = np.delete(self.points, indexes, axis=0)
        self.weights = np.delete(self.weights, indexes, axis=0)
        self.sensitivities = np.delete(self.sensitivities, indexes, axis=0)
        self.indexes = np.delete(self.indexes, indexes, axis=0)


    def get_sum_of_sensitivities(self):
        return sum(self.sensitivities)


    def     get_closest_points_to_set_of_points(self, P, m, type):
        """
        Args:
            P (SetOfPoints) : a set of points
            m (int): size of sample - may be percent or fixed number, depends on the parameter 'type'
            type (str): available values: "by number"/"by rate"

        Returns:
            SetOfPoints: the points that are closest to the given set of points, by rate or by
                         fixed number
        """

        assert type == "by number" or type == "by rate", "type undefined"
        if type == "by number":
            assert m <= self.get_size(), "(1) Number of points in query is larger than number of points in the set"
        if type == "by rate":
            assert m >= 0 and m <= 1, "(2) Number of points in query is larger than number of points in the set"

        self_size = self.get_size()
        self_points = np.asarray(self.points)
        self_weights = np.asarray(self.weights)
        P_size = P.get_size()
        P_points = np.asarray(P.points)
        P_weights = np.asarray(P.weights)

        self_points_repeat_each_point = np.repeat(self_points, repeats=P_size, axis=0) #this duplicate the self_point P_size times
        P_points_repeat_all = np.repeat(P_points.reshape(1, -1), repeats=self_size, axis=0).reshape(-1, self.dim) #this duplicate the P_points self_size times

        self_weights_repeat_each_point = np.repeat(self_weights, repeats=P_size, axis=0).reshape(-1)  # this duplicate the self_point P_size times
        P_weights_repeat_all = np.repeat(P_weights.reshape(1, -1), repeats=self_size,axis=0).reshape(-1)  # this duplicate the P_points self_size times

        self_points_repeat_each_point_minus_P_points_repeat_all = np.sum((self_points_repeat_each_point - P_points_repeat_all)** 2, axis=1)
        all_distances_unreshaped = self_points_repeat_each_point_minus_P_points_repeat_all * self_weights_repeat_each_point * P_weights_repeat_all
        all_distances_reshaped = all_distances_unreshaped.reshape(-1, P_size)
        all_distances = np.min(all_distances_reshaped, axis=1)

        if type == "by rate":
            m = int(m * self.get_size())  # number of points is m percents of n
        m_th_distance = np.partition(all_distances, m)[m]  # the m-th distance
        distances_smaller_than_median_indices = tuple(np.where(all_distances <= m_th_distance))  # all the m smallest distances indices in self.points
        P_subset = self_points[distances_smaller_than_median_indices]
        w_subset = self_weights[distances_smaller_than_median_indices]
        indexes_subset = self.indexes[distances_smaller_than_median_indices]
        return SetOfPoints(P_subset, w_subset, indexes=indexes_subset)


    def set_all_sensitivities(self, sensitivity):
        """
        The method gets a number and set all the sensitivities to be that number
        Args:
            sensitivity (float) : the sensitivity we set for all the points in the set

        Returns:
            ~
        """

        assert sensitivity > 0, "sensitivity is not positive"
        assert self.get_size() > 0, "set is empty"

        new_sensitivities = np.ones((self.get_size(), 1), dtype=np.float) * sensitivity
        self.sensitivities = new_sensitivities

    ######################################################################

    def set_weights(self, T, m):
        """
        The method sets the weights in the set to as described in line 10 int the main alg;
        Args:
            T (float) : sum of sensitivities
            m (int) : coreset size

        Returns:
            ~
        """

        assert self.get_size() > 0, "set is empty"

        numerator = self.weights.reshape(-1,1) * T #np.ones((self.get_size(), 1), dtype=np.float) * T
        denominator = self.sensitivities * m
        new_weights = numerator / denominator

        self.weights = new_weights

    #######################################################################

    def get_probabilites(self):
        """
        The method returns the probabilities to be choosen as described in line 9 in main alg
        Returns:
            np.ndarray: the probabilities to be choosen
        """

        T = self.get_sum_of_sensitivities()

        probs = self.sensitivities / T
        return probs

    #########################################################################

    def set_sensitivities(self, k):
        """
        The method set the sensitivities of the points in the set as decribed in line 5 in main alg.
        Args:
            k (int) : number of outliers

        Returns:
            ~
        """

        assert self.get_size() > 0, "set is empty"

        size = self.get_size()
        new_sensitivities = np.ones((self.get_size(), 1), dtype=np.float) * ((1*k)/size)
        self.sensitivities = new_sensitivities

    #########################################################################

    def get_arbitrary_sensitivity(self):
        """
        The method returns an arbitrary sensitivity from the set
        Returns:
            float: a random sensitivity from the set
        """

        assert self.get_size() > 0, "set is empty"

        num = random.randint(-1, self.get_size() - 1)
        return self.sensitivities[num]


    def sort_by_indexes(self):
        self_size = self.get_size()
        self_points = self.points
        self_weights = self.weights
        self_sensitivities = self.sensitivities
        self_indexes = self.indexes
        new_points = []
        new_weights = []
        new_sensitivities = []
        new_indexes = []
        for i in range(self_size):
            for j in range(self_size):
                if self.indexes[j] == i:
                    new_points.append(self_points[j])
                    new_weights.append(self_weights[j])
                    new_sensitivities.append(self_sensitivities[j])
                    new_indexes.append(self_indexes[j])
        self.points = np.asarray(new_points)
        self.weights = np.asarray(new_weights)
        self.sensitivities = np.asarray(new_sensitivities)
        self.indexes = np.asarray(new_indexes)

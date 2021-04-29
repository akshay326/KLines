#################################################################
#     Corset for k means for lines                              #
#     Paper: TBD                                                #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################
import copy

import numpy as np

class SetOfLines:
    """
    Class that represent a set of weighted lines in any d>0 dimensional space
    Attributes:
        spans (ndarray) : The spaning vectors. The i-th element is the spanning vector of the i-th line in the set
        displacements (ndarray) : The displacements. The i-th element is the displacement vector of the i-th line in the set
        dim (integer): The space's dimension
    """

    ##################################################################################

    def __init__(self, spans=None, displacements=None, weights=None, sen=None, lines=None, is_points=False):

        if spans is None:
            spans = []
        if displacements is None:
            displacements = []
        if weights is None:
            weights = []
        if sen is None:
            sen = []
        if lines is None:
            lines = []

        if is_points:
            self.dim = 2
            self.spans = []
            self.displacements = []
            for line in lines:
                v1 = np.asarray([line[0], line[1]])
                v2 = np.asarray([line[2], line[3]])
                span = v1 - v2
                displacement = v1
                self.spans.append(span)
                self.displacements.append(displacement)
            self.spans = np.asarray(self.spans)
            self.displacements = np.asarray(self.displacements)
            self.normalize_spans()
            self.weights = np.ones(len(lines)).reshape(-1)
            self.sensitivities = np.ones(len(lines))
        else:
            size = len(spans)
            if size == 0:  # there is no lines in the set we got
                self.spans = []
                self.displacements = []
                self.weights = []
                self.sensitivities = []
                self.dim = 0
                return
            [_, self.dim] = np.shape(spans)
            self.spans = spans
            self.normalize_spans()
            self.displacements = displacements
            self.weights = weights
            self.sensitivities = sen
  
    def get_all_intersection_points(self):
        """
        returns n(n-1) points, where each n-1 points in the n-1 points on each line that are closest to the rest n-1 lines.

        Returns:
            np.ndarray: all the "intersection" points
        """
        assert self.get_size() > 0, "set is empty"

        spans = self.spans
        displacements = self.displacements
        dim = self.dim
        size = self.get_size()
        t = range(size)
        indexes_repeat_all_but_one = np.array([[x for i, x in enumerate(t) if i != j] for j, j in enumerate(t)]).reshape(-1)

        spans_rep_each = spans[indexes_repeat_all_but_one]  # repeat of the spans, each span[i] is being repeated size times in a sequance
        spans_rep_all = np.repeat(spans.reshape(1, -1), size - 1, axis=0).reshape(-1, dim)  # repeat of the spans, all the spans block is repeated size-1 times
        disp_rep_each = displacements[indexes_repeat_all_but_one]  # repeat of the displacements, each span[i] is being repeated size times in a sequance
        disp_rep_all = np.repeat(displacements.reshape(1, -1), size - 1, axis=0).reshape(-1, dim)  # repeat of the displacements, all the spans block is repeated size-1 times

        W0 = disp_rep_each - disp_rep_all
        a = np.sum(np.multiply(spans_rep_each, spans_rep_each), axis=1)
        b = np.sum(np.multiply(spans_rep_each, spans_rep_all), axis=1)
        c = np.sum(np.multiply(spans_rep_all, spans_rep_all), axis=1)
        d = np.sum(np.multiply(spans_rep_each, W0), axis=1)
        e = np.sum(np.multiply(spans_rep_all, W0), axis=1)
        be = np.multiply(b, e)
        cd = np.multiply(c, d)
        be_minus_cd = be - cd
        ac = np.multiply(a, c)
        b_squared = np.multiply(b, b)
        ac_minus_b_squared = ac - b_squared
        s_c = be_minus_cd / ac_minus_b_squared
        
        s_c_repeated = np.repeat(s_c.reshape(-1, 1), dim, axis=1)
        G = disp_rep_each + np.multiply(s_c_repeated, spans_rep_each)

        b = np.where(np.isnan(G))
        c = np.where(np.isinf(G))
        G2 = np.delete(G, np.concatenate((b[0], c[0]), axis=0), axis=0).reshape(-1, dim)

        if len(G2) == 0:  # that means all the lines are parallel, take k random points from the displacements set
            return displacements
        return G2

  
    def get_size(self):
        return np.shape(self.spans)[0]


    def get_sample_of_lines(self, size_of_sample):
        """
        Args:
            size_of_sample (int) : the sample's size

        Returns:
            SetOfLines: sample consist of size_of_sample lines from the uniform distribution over the set
        """

        assert self.get_size() > 0, "set is empty"
        assert size_of_sample > 0, "size_of_sample <= 0"

        size = self.get_size()
        if size_of_sample >= size:
            return self
        else:
            all_indices = np.asarray(range(size))
            sample_indices = np.random.choice(all_indices, size_of_sample, False).tolist()
            sample_spans = np.take(self.spans, sample_indices, axis=0, out=None, mode='raise')
            sample_displacements = np.take(self.displacements, sample_indices, axis=0, out=None, mode='raise')
            try:
                sample_weights = np.take(self.weights, sample_indices, axis=0, out=None, mode='raise')
            except Exception as e:
                x = 2
        return SetOfLines(sample_spans, sample_displacements, sample_weights)

    ##################################################################################

    def get_indices_clusters(self, centers):
        """
        This method gets a set of k centers (points), and returns a size-dimensional row vector of indices in the range
        [0,k-1], where every number num in the i-th item indicates that centers[i] is the center that the i-th line was
        clustered into.

        Args:
            centers (SetOfPoints) : a set of centers

        Returns:
            np.ndarray: an array of n indices, where each index is in the range [0,k-1]
        """

        assert self.get_size() > 0, "set is empty"
        centers_size = centers.get_size()
        assert centers_size > 0, "no centers given"

        self_size = self.get_size()
        dim = self.dim
        self_displacements = self.displacements
        self_spans = self.spans
        self_weights = np.array(self.weights)
        centers_points = centers.points
        centers_weights = centers.weights

        # this is a size*k-simensional vector, where the i-th element is center[j], where j=i/k
        centers_points_repeat_each_row = np.repeat(centers_points, self_size, axis=0).reshape(-1, dim)  

        # repeating the displacement for the sum of squared distances calculation from each center for all the lines
        displacements_repeat_all = np.repeat(self_displacements.reshape(1, -1), centers_size, axis=0).reshape(-1, dim)  
        # repeating the displacement for the sum of squared distances calculation from each center for all the lines                                                                                                                                                                                                                        
        spans_repeat_all = np.repeat(self_spans.reshape(1, -1), centers_size, axis=0).reshape(-1,dim)  
        centers_minus_displacements = centers_points_repeat_each_row - displacements_repeat_all
        centers_minus_displacements_squared_norms = np.sum(np.multiply(centers_minus_displacements, centers_minus_displacements), axis=1)
        centers_minus_displacements_dot_spans = np.multiply(centers_minus_displacements, spans_repeat_all)
        centers_minus_displacements_dot_spans_squared_norms = np.sum(np.multiply(centers_minus_displacements_dot_spans, centers_minus_displacements_dot_spans), axis=1)
        
        all_unwighted_distances = centers_minus_displacements_squared_norms - centers_minus_displacements_dot_spans_squared_norms
        self_weights_repeat_all =         np.repeat(self_weights.reshape(-1, 1), centers_size, axis=0).reshape(-1, 1)
        centers_weights_repeat_each_row = np.repeat(centers_weights,             self_size,    axis=0).reshape(-1, 1)
        total_weights = np.multiply(self_weights_repeat_all, centers_weights_repeat_each_row)
        all_weighted_distances = np.multiply(all_unwighted_distances.reshape(-1, 1), total_weights.reshape(-1, 1))
        all_distances = (all_weighted_distances).reshape(-1, self_size)

        cluster_indices = np.argmin(all_distances.T, axis=1)  # the i-th element in this array contains the index of the cluster the i-th line was clusterd into.
        return cluster_indices

  
    def get_sum_of_distances_to_centers(self, centers):
        """
        This method gets a cet of k points and return the sum of squared distances from these points to the lines in
        the set

        Args:
            centers (SetOfPoints) : a set of k centers

        Returns:
            float: the sum of squared distances to the lines in the set from the centers
        """

        assert self.get_size() > 0, "set is empty"
        centers_size = centers.get_size()
        assert centers_size > 0, "no centers given"

        dim = self.dim
        self_size = self.get_size()
        self_displacements = self.displacements
        self_spans = self.spans
        self_weights = self.weights

        centers_points = centers.points
        centers_weights = centers.weights

        centers_points_repeat_each_row = np.repeat(centers_points, self_size, axis=0).reshape(-1,
                                                                                              dim)  # this is a k*size array where every k points were duplicated size times
        self_displacements_repeat_all = np.repeat(self_displacements.reshape(1, -1), centers_size, axis=0).reshape(-1,
                                                                                                                   dim)  # this is a size*k array where every size displacements were duplicated k times
        self_spans_repeat_all = np.repeat(self_spans.reshape(1, -1), centers_size, axis=0).reshape(-1,
                                                                                                   dim)  # this is a size*k array where every size spans were duplicated k times
        self_weights_repeat_all = np.repeat(self_weights.reshape(1, -1), centers_size,
                                            axis=0)  # this is a size*k array where every size spans were duplicated k times
        centers_weights_repeat_each_row = np.repeat(centers_weights, self_size, axis=0).reshape(-1,
                                                                                                1)  # this is a size*k array where every size spans were duplicated k times
        centers_points_repeat_each_row_minus_displacements_repeat_all = centers_points_repeat_each_row - self_displacements_repeat_all
        centers_points_minus_displacements_norm_squared = np.sum(
            centers_points_repeat_each_row_minus_displacements_repeat_all ** 2, axis=1)
        centers_points_minus_displacements_mul_spans_norm_squared = np.sum(
            np.multiply(centers_points_repeat_each_row_minus_displacements_repeat_all, self_spans_repeat_all) ** 2,
            axis=1)
        unweighted_all_distances = centers_points_minus_displacements_norm_squared.reshape(-1,
                                                                                           1) - centers_points_minus_displacements_mul_spans_norm_squared.reshape(
            -1, 1)

        total_weights = np.multiply(centers_weights_repeat_each_row.reshape(-1, 1),
                                    self_weights_repeat_all.reshape(-1, 1))
        all_weighted_distances = np.multiply(unweighted_all_distances.reshape(-1, 1), total_weights.reshape(-1, 1))
        all_distances = (all_weighted_distances).reshape(-1, self_size)

        all_distances_min = np.min(all_distances, axis=0)
        return np.sum(all_distances_min)

  
    def get_farthest_lines_to_centers(self, centers, m, type):
        """
        Args:
            centers (npndarray) : d-dimensional points centers
            m (int): size of sample - may be percent or fixed number, depends on the parameter 'type'
            type (str): available values: "by number"/"by rate"
        Returns:
            SetOfLines: the lines that are farthest to the given centers, by rate or by fixed number
        """

        assert type == "by number" or type == "by rate", "type undefined"
        if type == "by number":
            assert m <= self.get_size(), "(1) Number of lines in query is larger than number of lines in the set"
        if type == "by rate":
            assert m >= 0 and m <= 1, "(2) the rate invalid"

        self_spans = self.spans
        self_displacements = self.displacements
        self_weights = self.weights

        cluster_indices = self.get_indices_clusters(centers)
        centers_by_cluster_indices = centers.get_points_from_indices(
            cluster_indices)  # that is an array of size points, where the i-th element is the centers[cluster_indices[i]]

        centeres_clustered_points = centers_by_cluster_indices.points
        centeres_clustered_weights = centers_by_cluster_indices.weights

        centers_by_cluster_indices_minus_displacements = centeres_clustered_points - self_displacements
        centers_by_cluster_indices_minus_displacements_squared_norms = np.sum(
            np.multiply(centers_by_cluster_indices_minus_displacements, centers_by_cluster_indices_minus_displacements),
            axis=1)
        centers_mul_spans_squared_norms = np.sum(
            np.multiply(centers_by_cluster_indices_minus_displacements, self_spans) ** 2, axis=1)
        all_unweighted_distances = centers_by_cluster_indices_minus_displacements_squared_norms - centers_mul_spans_squared_norms
        total_weights = np.multiply(centeres_clustered_weights.reshape(-1, 1), self_weights.reshape(-1, 1)).reshape(-1)
        all_distances = np.multiply(all_unweighted_distances.reshape(-1, 1), total_weights.reshape(-1, 1)).reshape(-1)
        if type == "by rate":
            m = int(m * self.get_size())  # number of lines is m percents of size
        # m_th_distance = np.partition(all_distances, m)[m]  # the m-th distance
        # distances_higher_than_median_indices = np.where(all_distances >= m_th_distance)  # all the m highest distances indices in all_distances
        all_distances_mth_index_in_the_mth_place = np.argpartition(all_distances, m)
        if len(all_distances_mth_index_in_the_mth_place) % 2 == 0:
            first_m_smallest_distances_indices = all_distances_mth_index_in_the_mth_place[m:len(all_distances)]
        else:
            first_m_smallest_distances_indices = all_distances_mth_index_in_the_mth_place[m - 1:len(all_distances)]
        spans_subset = self.spans[first_m_smallest_distances_indices]
        displacements_subset = self.displacements[first_m_smallest_distances_indices]
        weights_subset = self.weights[first_m_smallest_distances_indices]
        return SetOfLines(spans_subset, displacements_subset, weights_subset)

    ##################################################################################

    def get_lines_at_indices(self, indices):
        """
        Args:
            indices (list of ints) : list of indices.

        Returns:
            SetOfLines: a set of lines that contains the points in the input indices
        """

        assert self.get_size() > 0, "set is empty"
        assert len(indices) > 0, "no indices given"

        new_spans = self.spans[indices]
        new_displacements = self.displacements[indices]
        new_weights = self.weights[indices]

        L = SetOfLines(new_spans, new_displacements, new_weights)
        return L

    ##################################################################################

    def add_set_of_lines(self, other):
        if self.get_size() == 0:
            self.dim = copy.deepcopy(other.dim)
            self.spans = copy.deepcopy(other.spans)
            self.weights = copy.deepcopy(other.weights)
            self.displacements = copy.deepcopy(other.displacements)
            return
        self.spans = np.concatenate((self.spans, other.spans))
        self.weights = np.concatenate((self.weights.reshape(-1, 1), other.weights.reshape(-1, 1)))
        self.displacements = np.concatenate((self.displacements, other.displacements))


    def get_lines_at_indexes_interval(self, start, end):
        """
        Args:
            start (int) : starting index
            end (end) : ending index

        Returns:
            SetOfLines: a set of lines that contains the points in the given range of indices
        """

        size = end - start
        indices = np.asarray(range(size)) + start

        spans_subset = self.spans[indices]
        displacements_subset = self.displacements[indices]
        weights_subset = self.weights[indices]
        return SetOfLines(spans_subset, displacements_subset, weights_subset)


    def get_sensitivities_first_argument_for_centers(self, B):
        """

        :param B (SetOfPoints) :  a set of centers to compute the sensitivities first arfument as in Alg.4 in the paper
        :return (ndarray) : an array of n numbers, where the i-th number is the sensitivity first arg of the i-th line
        """

        assert B.get_size() > 0, "The number of centers is zero"

        cost_to_B = self.get_sum_of_distances_to_centers(B)

        cluster_indexes = self.get_indices_clusters(B)
        clustered_points = B.get_points_from_indices(cluster_indexes)

        dim = self.dim
        self_size = self.get_size()
        self_displacements = self.displacements
        self_spans = self.spans
        self_weights = self.weights

        centers_points = clustered_points.points
        centers_weights = clustered_points.weights

        centers_points_repeat_each_row = np.repeat(centers_points, self_size, axis=0).reshape(-1, dim)
        centers_weights_repeat_each_row = np.repeat(centers_weights, self_size, axis=0).reshape(-1, 1)
        self_displacements_repeat_all = np.repeat(self_displacements.reshape(1, -1), self_size, axis=0).reshape(-1, dim)
        self_spans_repeat_all = np.repeat(self_spans.reshape(1, -1), self_size, axis=0).reshape(-1, dim)
        self_weights_repeat_all = np.repeat(self_weights.reshape(1, -1), self_size, axis=0)
        centers_points_repeat_each_row_minus_displacements_repeat_all = centers_points_repeat_each_row - self_displacements_repeat_all
        # centers_points_minus_displacements_norm_squared = np.sum(np.multiply(centers_points_repeat_each_row_minus_displacements_repeat_all,centers_points_repeat_each_row_minus_displacements_repeat_all), axis=1)
        centers_points_minus_displacements_norm_squared = np.sum(
            centers_points_repeat_each_row_minus_displacements_repeat_all ** 2, axis=1)
        try:
            centers_points_minus_displacements_mul_spans_norm_squared = np.sum(
                np.multiply(centers_points_repeat_each_row_minus_displacements_repeat_all, self_spans_repeat_all) ** 2,
                axis=1)
        except:
            self_spans_repeat_all_nan_indexes = np.where(np.isnan(self_spans_repeat_all))
            self_spans_repeat_all[self_spans_repeat_all_nan_indexes] = np.inf
            centers_points_minus_displacements_mul_spans_norm_squared = np.sum(
                np.multiply(centers_points_repeat_each_row_minus_displacements_repeat_all, self_spans_repeat_all) ** 2,
                axis=1)
        unweighted_all_distances = centers_points_minus_displacements_norm_squared - centers_points_minus_displacements_mul_spans_norm_squared
        less_than_zero_indexes = np.where(unweighted_all_distances < 0)
        is_nan_indexes = np.where(np.isnan(unweighted_all_distances))
        is_inf_indexes = np.where(np.isinf(unweighted_all_distances))
        less_than_zero_sum = np.sum(less_than_zero_indexes)
        is_nan_indexes_sum = np.sum(is_nan_indexes)
        is_inf_indexes_sum = np.sum(is_inf_indexes)
        if less_than_zero_sum + is_nan_indexes_sum + is_inf_indexes_sum > 0:
            print("less_than_zero_sum: ", less_than_zero_sum)

        total_weights = np.multiply(centers_weights_repeat_each_row.reshape(-1, 1),
                                    self_weights_repeat_all.reshape(-1, 1))
        all_weighted_distances = np.multiply(unweighted_all_distances.reshape(-1, 1), total_weights.reshape(-1, 1))
        all_distances = (all_weighted_distances).reshape(-1, self_size)
        all_distances_min = np.min(all_distances, axis=0)
        sensitivities_first_argument = all_distances_min / cost_to_B

        return sensitivities_first_argument

    def normalize_spans(self):
        spans_norm = np.sum(self.spans ** 2, axis=1) ** 0.5
        spans_norm_inv = 1 / spans_norm
        spans_norm_inv_repeated = np.repeat(spans_norm_inv.reshape(-1), self.dim).reshape(-1, self.dim)
        self.spans = np.multiply(self.spans, spans_norm_inv_repeated)

    def multiply_weights_by_value(self, val):
        self.weights = self.weights * val

    def get_projected_centers(self, centers):
        """
        This function gets a set of k centers, project each one of the centers onto its closest line in the ser and
        returns the n projected centers
        :param centers:
        :return:
        """

        spans = self.spans
        displacements = self.displacements
        dim = self.dim

        indices_cluster = self.get_indices_clusters(centers)
        centers_at_indices_cluster = centers.get_points_from_indices(indices_cluster)
        centers_points_at_indices_cluster = centers_at_indices_cluster.points
        centers_minus_displacements = centers_points_at_indices_cluster - displacements
        centers_minus_displacements_dot_spans = np.multiply(centers_minus_displacements, spans)
        projected_points = centers_minus_displacements_dot_spans + displacements
        return projected_points

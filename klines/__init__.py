import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs': np.get_include()})

from klines.set_of_lines import SetOfLines
from klines.set_of_points import SetOfPoints
from klines.coreset_for_k_means_for_lines import CorsetForKMeansForLines
from klines.coreset_for_weighted_centers import CoresetForWeightedCenters
from klines.coreset_streamer import CoresetStreamer, TreeNode

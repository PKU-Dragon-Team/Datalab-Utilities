import numpy as np
import typing as tg
import scipy.spatial.distance as spsd
import heapq as hq


def SimpleHierachicalCluster(X: np.ndarray, weight: tg.Union[np.ndarray]=None) -> np.ndarray:
    """My version of Hierachical Clustering, processing small amount of samples and give easily judgement of the hierachical tree
    Running time: O(N^2*D), N is n_samples, D is n_features
    """
    n_samples, n_features = X.shape

    if weight is None:
        weights = np.ones(n_samples)
    else:
        weights = weight.copy()

    output = np.zeros((n_samples - 1, 4))  # each row: (index1: int, index2:int, index_new:int, distance:float)
    output_index = 0

    nodes = X.copy()
    remaining_indexes = set(range(n_samples))
    distances = []
    calculated = set()
    for i in remaining_indexes:
        calculated.add(i)
        for j in remaining_indexes - calculated:
            hq.heappush(distances, (spsd.euclidean(nodes[i], nodes[j]), i, j))
    # now go with clustering
    while len(remaining_indexes) > 1:
        # drop merged ones
        min_d, index1, index2 = hq.heappop(distances)
        while not (index1 in remaining_indexes and index2 in remaining_indexes):
            min_d, index1, index2 = hq.heappop(distances)

        centroid = (weights[index1] * nodes[index1] + weights[index2] * nodes[index2]) / (weights[index1] + weights[index2])
        # now new centroid comes, drop i and j, and calculate new distances
        remaining_indexes.remove(index1)
        remaining_indexes.remove(index2)
        index_new = nodes.shape[0]
        for i in remaining_indexes:
            hq.heappush(distances, (spsd.euclidean(nodes[i], centroid), i, index_new))

        remaining_indexes.add(index_new)
        weights = np.hstack((weights, weights[index1] + weights[index2]))
        nodes = np.vstack((nodes, centroid))
        # forming output
        output[output_index] = (index1, index2, index_new, min_d)
        output_index += 1

    return output, nodes, weights

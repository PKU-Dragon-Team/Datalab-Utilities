import numpy as np
import typing as tg
import sklearn.cluster as sklc
import scipy.spatial.distance as spsd
import scipy.signal as spsn
import heapq as hq
import matplotlib.pyplot as plt


def SimpleHierachicalCluster(X: np.ndarray, weight: tg.Optional[np.ndarray]=None) -> np.ndarray:
    """My version of Hierachical Clustering, processing small amount of samples and give easily judgement of the hierachical tree
    Running time: O(N^2*D), N is n_samples, D is n_features
    """
    n_samples, n_features = X.shape

    if weight is None:
        weights = np.ones(n_samples, dtype=int)
    else:
        weights = weight.copy()

    hierachical = np.zeros((n_samples - 1, 3), dtype=int)  # each row: (index1: int, index2:int, index_new:int)
    distances = np.zeros(n_samples - 1)
    output_index = 0

    nodes = X.copy()
    remaining_indexes = set(range(n_samples))
    distance_heap = []
    calculated = set()
    for i in remaining_indexes:
        calculated.add(i)
        for j in remaining_indexes - calculated:
            hq.heappush(distance_heap, (spsd.euclidean(nodes[i], nodes[j]), i, j))
    # now go with clustering
    while len(remaining_indexes) > 1:
        # drop merged ones
        min_d, index1, index2 = hq.heappop(distance_heap)
        while not (index1 in remaining_indexes and index2 in remaining_indexes):
            min_d, index1, index2 = hq.heappop(distance_heap)

        centroid = (weights[index1] * nodes[index1] + weights[index2] * nodes[index2]) / (weights[index1] + weights[index2])
        # now new centroid comes, drop i and j, and calculate new distances
        remaining_indexes.remove(index1)
        remaining_indexes.remove(index2)
        index_new = nodes.shape[0]
        for i in remaining_indexes:
            hq.heappush(distance_heap, (spsd.euclidean(nodes[i], centroid), i, index_new))

        remaining_indexes.add(index_new)
        weights = np.hstack((weights, weights[index1] + weights[index2]))
        nodes = np.vstack((nodes, centroid))
        # forming output
        hierachical[output_index] = (index1, index2, index_new)
        distances[output_index] = min_d
        output_index += 1

    return hierachical, distances, nodes, weights


def LastLocalMinimumCluster(X: np.ndarray, weight: tg.Optional[np.ndarray]=None) -> tg.Tuple[np.ndarray, np.ndarray]:
    """Hierachical Cluster that pick the last local minimum of distance and cut the cluster tree into several clusters
    """
    n_samples, n_features = X.shape
    hierachical, distances, nodes, weights = SimpleHierachicalCluster(X, weight)
    # find local minimums
    extrema = spsn.argrelmin(distances)  # type: np.ndarray
    try:
        last_local_minimum = extrema[0][len(extrema[0]) - 1]
    except IndexError:  # no local_minimum, return all clustered nodes
        return nodes[n_samples:], weights[n_samples]

    merged_nodes = set(hierachical[:last_local_minimum + 1, 0:2].flat)
    post_cluster_nodes = set(hierachical[last_local_minimum + 1:, 2].flat)
    total_nodes = set(range(len(nodes)))
    cluster_centers = total_nodes - post_cluster_nodes - merged_nodes  # nodes that is not merged will be cluster_centers
    return nodes[list(cluster_centers)], weights[list(cluster_centers)]


def PercentageCluster(X: np.ndarray, weight: tg.Optional[np.ndarray]=None, percentage: float=0.5) -> tg.Tuple[np.ndarray, np.ndarray]:
    """Hierachical Cluster that cut the cluster tree into several clusters when distance is higher than the require percentage
    """
    n_samples, n_features = X.shape
    hierachical, distances, nodes, weights = SimpleHierachicalCluster(X, weight)
    # find cutting point
    max_d = np.max(distances)
    extrema = distances < percentage * max_d
    for i, x in enumerate(np.flipud(extrema)):
        if x:
            break

    last_less = len(extrema) - 1 - i

    merged_nodes = set(hierachical[:last_less + 1, 0:2].flat)
    post_cluster_nodes = set(hierachical[last_less + 1:, 2].flat)
    total_nodes = set(range(len(nodes)))
    cluster_centers = total_nodes - post_cluster_nodes - merged_nodes  # nodes that is not merged will be cluster_centers
    return nodes[list(cluster_centers)], weights[list(cluster_centers)]


def MeanLogWardTree(X: np.ndarray, weight: tg.Optional[np.ndarray]=None) -> np.ndarray:
    """Hierachical Cluster that pick the mean of log of ward_tree distance and cut the cluster tree into several clusters
    """
    if weight is None:
        connect = None
    else:
        connect = np.outer(weight, weight)
    n_samples, n_features = X.shape
    children, n_components, n_leaves, parents, distances = sklc.ward_tree(X, connectivity=connect, return_distance=True)
    c = children.shape[0]
    hierachical = np.hstack([children, np.arange(n_samples, n_samples + c).reshape(c, 1)])
    log_distance = np.log(np.sqrt(distances**2 / n_features))
    mean_log_distance = np.mean(log_distance)
    distance_mask = log_distance < mean_log_distance
    post_distance_mask = log_distance >= mean_log_distance

    merged_nodes = set(hierachical[distance_mask, 0:2].flat)
    post_cluster_nodes = set(hierachical[post_distance_mask, 2].flat)
    total_nodes = set(range(n_samples, c + n_samples))
    cluster_centers = total_nodes - post_cluster_nodes - merged_nodes  # nodes that is not merged will be cluster_centers
    print(merged_nodes)
    print(post_cluster_nodes)
    print(total_nodes)
    return hierachical[np.asarray(list(cluster_centers)) - n_samples]

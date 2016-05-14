import numpy as np
import typing as tg
import enum as em
import scipy.spatial.distance as spsd
import heapq as hq

# import sklearn.cluster as sklc
# from sklearn.cluster import _hierarchical
# from sklearn.cluster.hierarchical import _TREE_BUILDERS, _hc_cut
# from sklearn.externals import six
# from sklearn.externals.joblib import Memory
# from sklearn.utils import check_array


class SimpleHC_t(em.Enum):
    ndarray = 0
    tree = 1


def SimpleHierachicalCluster(X: np.ndarray, weight: np.ndarray=None, outputType: SimpleHC_t=SimpleHC_t.ndarray) -> tg.Union[np.ndarray, tg.Dict]:
    """My version of Hierachical Clustering, processing small amount of samples and give easily judgement of the hierachical tree
    Running time: O(N^2*D), N is n_samples, D is n_features
    """
    n_samples, n_features = X.shape

    if not weight:
        weights = np.ones(n_samples)
    else:
        weight = weights.copy()

    if outputType == SimpleHC_t.ndarray:
        output = np.zeros(n_samples - 1, 4)  # index1, index2, index_new, distance
        output_index = 0
    else:
        output = [{'index': i, 'weight': weights[i], 'left': None, 'right': None, 'distance': 0} for i in range(n_samples)]

    nodes = X.copy()
    remaining_indexes = list(range(nodes.shape[0]))
    merged_indexes = set()
    distances = []
    for ii in range(len(remaining_indexes)):
        for jj in range(ii + 1, remaining_indexes):
            hq.heappush(distances, (spsd.euclidean(nodes[remaining_indexes[ii]], nodes[remaining_indexes[jj]]), ii, jj))
    # now go with clustering
    while len(remaining_indexes) > 1:
        # drop merged ones
        d, ii, jj = hq.heappop(distances)
        while ii in merged_indexes or jj in merged_indexes:
            d, ii, jj = hq.heappop(distances)

        i = remaining_indexes[ii]  # index1
        j = remaining_indexes[jj]  # index2
        centroid = (weights[i] * X[i] + weights[j] * X[j]) / (weights[i] + weights[j])
        # now new centroid comes, drop i and j, and calculate new distances
        k = nodes.shape[0]  # index_new
        kk = len(remaining_indexes)
        del remaining_indexes[ii]
        del remaining_indexes[jj]
        merged_indexes.add(i)
        merged_indexes.add(j)
        for ii in remaining_indexes:
            hq.heappush(distances, (spsd.euclidean(nodes[remaining_indexes[ii]], centroid), ii, kk))
        remaining_indexes.append(k)
        weights.append(weights[i] + weights[j])
        nodes = np.vstack((nodes, centroid))
        # forming output
        if outputType == SimpleHC_t.ndarray:
            output[output_index] = (i, j, k, d)
            output_index += 1
        else:
            output.append({'index': k, 'weight': weights[k], 'left': i, 'right': j, 'distance': d})

    return output, nodes, weights


# class AgglomerativeClustering(sklc.AgglomerativeClustering):
#     """Rewrite fit function to pass parameters and get distances
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def fit(self, X: np.ndarray, y: np.ndarray=None, return_distances: bool=False) -> tg.Optional:
#         """Fit the hierarchical clustering on the data Parameter
#         ----------
#         X : array-like, shape = [n_samples, n_features] The samples a.k.a. observations.

#         Returns
#         -------
#         self
#         """
#         X = check_array(X, ensure_min_samples=2, estimator=self)
#         memory = self.memory
#         if isinstance(memory, six.string_types):
#             memory = Memory(cachedir=memory, verbose=0)

#         if self.n_clusters <= 0:
#             raise ValueError("n_clusters should be an integer greater than 0." " %s was provided." % str(self.n_clusters))

#         if self.linkage == "ward" and self.affinity != "euclidean":
#             raise ValueError("%s was provided as affinity. Ward can only " "work with euclidean distancess." % (self.affinity, ))

#         if self.linkage not in _TREE_BUILDERS:
#             raise ValueError("Unknown linkage type %s." "Valid options are %s" % (self.linkage, _TREE_BUILDERS.keys()))
#         tree_builder = _TREE_BUILDERS[self.linkage]

#         connectivity = self.connectivity
#         if self.connectivity is not None:
#             if callable(self.connectivity):
#                 connectivity = self.connectivity(X)
#             connectivity = check_array(connectivity, accept_sparse=['csr', 'coo', 'lil'])

#         n_samples = len(X)
#         compute_full_tree = self.compute_full_tree
#         if self.connectivity is None:
#             compute_full_tree = True
#         if compute_full_tree == 'auto':
#             # Early stopping is likely to give a speed up only for
#             # a large number of clusters. The actual threshold
#             # implemented here is heuristic
#             compute_full_tree = self.n_clusters < max(100, .02 * n_samples)
#         n_clusters = self.n_clusters
#         if compute_full_tree:
#             n_clusters = None
#         # Construct the tree
#         kwargs = {return_distances: return_distances}
#         if self.linkage != 'ward':
#             kwargs['linkage'] = self.linkage
#             kwargs['affinity'] = self.affinity

#         if return_distances:
#             self.children_, self.n_components_, self.n_leaves_, parents, self.distances_ = memory.cache(tree_builder)(X, connectivity, n_components=self.n_components, n_clusters=n_clusters, **kwargs)
#         else:
#             self.children_, self.n_components_, self.n_leaves_, parents = memory.cache(tree_builder)(X, connectivity, n_components=self.n_components, n_clusters=n_clusters, **kwargs)
#         # Cut the tree
#         if compute_full_tree:
#             self.labels_ = _hc_cut(self.n_clusters, self.children_, self.n_leaves_)
#         else:
#             labels = _hierarchical.hc_get_heads(parents, copy=False)
#             # copy to avoid holding a reference on the original array
#             labels = np.copy(labels[:n_samples])
#             # Reasign cluster numbers
#             self.labels_ = np.searchsorted(np.unique(labels), labels)
#         return self

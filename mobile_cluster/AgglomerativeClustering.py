import numpy as np
import typing as tg

import sklearn.cluster as sklc
from sklearn.cluster import _hierarchical
from sklearn.cluster.hierarchical import _TREE_BUILDERS, _hc_cut
from sklearn.externals import six
from sklearn.externals.joblib import Memory
from sklearn.utils import check_array


class AgglomerativeClustering(sklc.AgglomerativeClustering):
    """Rewrite fit function to pass parameters and get distance
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray=None, return_distance: bool=False) -> tg.Optional:
        """Fit the hierarchical clustering on the data Parameter
        ----------
        X : array-like, shape = [n_samples, n_features] The samples a.k.a. observations.

        Returns
        -------
        self
        """
        X = check_array(X, ensure_min_samples=2, estimator=self)
        memory = self.memory
        if isinstance(memory, six.string_types):
            memory = Memory(cachedir=memory, verbose=0)

        if self.n_clusters <= 0:
            raise ValueError("n_clusters should be an integer greater than 0." " %s was provided." % str(self.n_clusters))

        if self.linkage == "ward" and self.affinity != "euclidean":
            raise ValueError("%s was provided as affinity. Ward can only " "work with euclidean distances." % (self.affinity, ))

        if self.linkage not in _TREE_BUILDERS:
            raise ValueError("Unknown linkage type %s." "Valid options are %s" % (self.linkage, _TREE_BUILDERS.keys()))
        tree_builder = _TREE_BUILDERS[self.linkage]

        connectivity = self.connectivity
        if self.connectivity is not None:
            if callable(self.connectivity):
                connectivity = self.connectivity(X)
            connectivity = check_array(connectivity, accept_sparse=['csr', 'coo', 'lil'])

        n_samples = len(X)
        compute_full_tree = self.compute_full_tree
        if self.connectivity is None:
            compute_full_tree = True
        if compute_full_tree == 'auto':
            # Early stopping is likely to give a speed up only for
            # a large number of clusters. The actual threshold
            # implemented here is heuristic
            compute_full_tree = self.n_clusters < max(100, .02 * n_samples)
        n_clusters = self.n_clusters
        if compute_full_tree:
            n_clusters = None
        # Construct the tree
        kwargs = {return_distance: return_distance}
        if self.linkage != 'ward':
            kwargs['linkage'] = self.linkage
            kwargs['affinity'] = self.affinity
        
        if return_distance:
            self.children_, self.n_components_, self.n_leaves_, parents, self.distance_ = memory.cache(tree_builder)(X, connectivity, n_components=self.n_components, n_clusters=n_clusters, **kwargs)
        else:
            self.children_, self.n_components_, self.n_leaves_, parents = memory.cache(tree_builder)(X, connectivity, n_components=self.n_components, n_clusters=n_clusters, **kwargs)
        # Cut the tree
        if compute_full_tree:
            self.labels_ = _hc_cut(self.n_clusters, self.children_, self.n_leaves_)
        else:
            labels = _hierarchical.hc_get_heads(parents, copy=False)
            # copy to avoid holding a reference on the original array
            labels = np.copy(labels[:n_samples])
            # Reasign cluster numbers
            self.labels_ = np.searchsorted(np.unique(labels), labels)
        return self

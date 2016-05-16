"""The package to determination cluster number and position
"""

import numpy as np
import sklearn.cluster as sklc
import typing as tg

import math


def method_1994(X: np.matrix, Ra: float=2, Rb: float=3, epsilon_upper: float=0.5, epsilon_lower: float=0.15):
    """funtion that use the 1994 Fuzzy Model Identification Based on Cluster Estimation method
        return the index of Cluster point in X and the potential of them
    """

    def cal_potential(xi: int, X: np.matrix, alpha: float):
        """Function calculate the potential of point x become a cluster center
        """
        return np.exp(-alpha * (((X - X[xi, :])**2).sum(axis=1))).sum()

    def cal_new_potential(old_i: int, center: tg.Tuple[float, int], X: np.matrix, p: np.ndarray, beta: float):
        """Function calculate the updated potential of point x after a new cluster center occurs
        """
        return p[old_i] - center[0] * np.exp(-beta * (((X[old_i, :] - X[center[1], :])**2).sum()))

    def cal_d_min(xi: int, centers: tg.List[tg.Tuple[float, int]], X: np.matrix):
        """Function calculate the shortest distance between point X[xi, :] and all previous cluster centers
        """
        return math.sqrt(((X[tuple(zip(*centers))[1], :] - X[xi, :])**2).sum(axis=1).min())

    alpha = 4 / (Ra**2)
    beta = 4 / (Rb**2)

    # calc the first center
    potential = np.fromiter((cal_potential(i, X, alpha) for i in range(X.shape[0])), dtype=float)
    first_center_i = potential.argmax()
    first_center_p = potential[first_center_i]
    centers = [(float(first_center_p), int(first_center_i)), ]  # KDTree is not modifiable, so use plain method
    calculated_count = 1
    print("Center %d detected." % len(centers))
    potential = np.fromiter((cal_new_potential(i, (first_center_p, first_center_i), X, potential, beta) for i in range(X.shape[0])), dtype=float)
    while len(potential) > calculated_count:
        most_potential_i = potential.argmax()
        most_potential_p = potential[most_potential_i]
        calculated_count += 1
        while True:
            if most_potential_p > epsilon_upper * first_center_p:
                accepted = True
                break
            elif most_potential_p < epsilon_lower * first_center_p:
                accepted = False
                break
            elif cal_d_min(most_potential_i, centers, X) / Ra + most_potential_p / first_center_p > 1:
                accepted = True
                break
            else:
                potential[most_potential_i] = 0
                most_potential_i = potential.argmax()
                most_potential_p = potential[most_potential_i]
                calculated_count += 1
        if accepted:
            centers.append((float(most_potential_p), int(most_potential_i)))
            potential = np.fromiter((cal_new_potential(i, (most_potential_p, most_potential_i), X, potential, beta) for i in range(X.shape[0])), dtype=float)
            print("Center %d detected." % len(centers))
        else:
            break
    return tuple(zip(*centers))


def autoBirch(X: np.ndarray, start_threshold: float=0.5, target_range: tg.Tuple[int, int]=(3, 100)) -> tg.Tuple[sklc.Birch, np.ndarray, float]:
    """Method that use dihotomy to automatically select threshold to fit n_cluster into target_range
    """
    prev_t = 0.0
    t = start_threshold
    birch = sklc.Birch(threshold=t, n_clusters=None)
    labels = birch.fit_predict(X)
    sc = np.unique(labels).size

    while sc not in range(*target_range):
        if sc < target_range[0]:
            new_t = (prev_t + t) / 2
            prev_t = t
            t = new_t
            birch = sklc.Birch(threshold=t, n_clusters=None)
            labels = birch.fit_predict(X)
            sc = np.unique(labels).size
        elif sc > target_range[1]:
            new_t = t * 2
            prev_t = t
            t = new_t
            birch = sklc.Birch(threshold=t, n_clusters=None)
            labels = birch.fit_predict(X)
            sc = np.unique(labels).size
        else:
            break

    return birch, labels, t

"""The package to determination cluster number and position
"""

import numpy as np
import scipy.spatial as spatial
import typing as tg
import heapq


def method_1994(X: np.matrix, Ra: float=2.0, Rb: float=3.0, epsilon_upper: float=0.5, epsilon_lower: float=0.15) -> tg.Tuple[tg.List[int], tg.List[float]]:
    """funtion that use the 1994 Fuzzy Model Identification Based on Cluster Estimation method
        return the index of Cluster point in X and the potential of them
    """

    def cal_potential(xi: int, X: np.matrix, alpha: float=1.0) -> float:
        """Function calculate the potential of point x become a cluster center
        """
        return np.sum(np.exp(-alpha * np.power(X - X[xi, :], 2)))  # Here didn't exclude the xi row. The result is different but it dosn't affect the sequence.

    def cal_new_potential(old: tg.Tuple[np.double, int], center: tg.Tuple[np.double, int], X: np.matrix, beta: float=4.0 / 9) -> float:
        """Function calculate the updated potential of point x after a new cluster center occurs
        """
        return old[0] - center[0] * np.exp(-beta * spatial.distance.sqeuclidean(X[old[1], :], X[center[1], :]))

    def cal_d_min(xi: int, centers: tg.List[tg.Tuple[np.double, int]], X: np.matrix) -> float:
        """Function calculate the shortest distance between point X[xi, :] and all previous cluster centers
        """
        distance = []  # heapq
        for _, i in centers:
            heapq.heappush(distance, spatial.distance.euclidean(X[xi, :], X[i, :]))

        return distance[0]

    alpha = 4 / Ra**2
    beta = 4 / Rb**2

    potential = []  # heapq
    # calc the first center
    for i, x in enumerate(X):
        heapq.heappush(potential, (-cal_potential(i, X, alpha), i))  # heapq only provides min-heap, WTF
    first_center = heapq.heappop(potential)

    first_center_p = -first_center[0]  # SB heapq
    centers = [(first_center_p, first_center[1])]  # KDTree is not modifiable, so use plain method

    potential_clone = []
    for point in potential:
        heapq.heappush(potential_clone, (cal_new_potential(point, first_center, X, beta), point[1]))
    potential = potential_clone

    while len(potential) > 0:
        most_potential_p, most_potential_i = heapq.heappop(potential)

        while True:
            most_potential_p = -most_potential_p  # SB heapq again
            if most_potential_p > epsilon_upper * first_center_p:
                accepted = True
                break
            elif most_potential_p < epsilon_lower * first_center_p:
                accepted = False
                break
            elif cal_d_min(X[most_potential_i, :], centers, X) / Ra + most_potential_p / first_center_p > 1:
                accepted = True
                break
            else:
                most_potential_p, most_potential_i = heapq.heappushpop(potential, (0, most_potential_i))

        if accepted:
            centers.append((most_potential_p, most_potential_i))
            potential_clone = []

            for point in potential:
                heapq.heappush(potential_clone, (cal_new_potential(point, (most_potential_p, most_potential_i), X, beta), point[1]))
            potential = potential_clone
        else:
            break
    return tuple(zip(*centers))
    # Question: what is the return value?

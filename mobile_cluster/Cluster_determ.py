"""The package to determination cluster number and position
"""

import numpy as np
import scipy.spatial as spatial
import typing as tg
import heapq


def method_1994(X: tg.Dict[tg.Tuple, tg.List],
                Ra: float=2.0,
                Rb: float=3.0,
                epsilon_upper: np.double=np.double(0.5),
                epsilon_lower: np.double=np.double(0.15)) -> (int, tg.Dict):
    """funtion that use the 1994 Fuzzy Model Identification Based on Cluster Estimation method
    """

    def cal_potential(x: np.array,
                      X: np.matrix,
                      alpha: np.double=np.double(1.0)) -> np.double:
        """Function calculate the potential of point x become a cluster center
        """
        return np.sum(np.exp(-alpha * spatial.distance.sqeuclidean(X, x)))

    def cal_new_potential(old: tg.Tuple[np.double, int],
                          center: tg.Tuple[np.double, int],
                          X: np.matrix,
                          beta: np.double=np.double(4 / 9)) -> np.double:
        """Function calculate the updated potential of point x after a new cluster center occurs
        """
        return old[0] - center[0] * np.exp(
            -beta * spatial.distance.sqeuclidean(X[old[1], :],
                                                 X[center[1], :]))

    def cal_d_min(x: np.array, centers: tg.List[tg.Tuple[np.double, int]], X:
                  np.matrix) -> np.double:
        """Function calculate the shortest distance between point x and all previous cluster centers
        """
        distance = []  # heapq
        for _, i in centers:
            heapq.heappush(distance, spatial.distance.euclidean(x, X[i, :]))

        return distance[0]

    alpha = 4 / Ra**2
    beta = 4 / Rb**2

    names, positions = [x for x in zip(*X.items())]
    X = np.matrix(positions)

    potential = []  # heapq
    # calc the first center
    for i, x in enumerate(X):
        heapq.heappush(potential, (-cal_potential(
            np.array(x), X, alpha), i))  # heapq only provides min-heap, WTF
    first_center = heapq.heappop(potential)

    centers = [first_center, ]  # KDTree is not modifiable, so use plain method
    first_center_p = -first_center[0]
    calculated = set([first_center, ])

    while True:
        most_potential, most_potential_i = heapq.heappop(potential)

        while True:
            most_potential = -most_potential
            if len(centers) == 0 or most_potential > epsilon_upper * centers[
                    0][0]:
                accepted = True
                break
            elif most_potential < epsilon_lower * first_center_p:
                accepted = False
                break
            elif cal_d_min(X[most_potential_i, :], centers,
                           X) / Ra + most_potential / first_center_p > 1:
                accepted = True
                break
            else:
                most_potential, most_potential_i = heapq.heappushpop(
                    potential, (0, most_potential_i))

        if accepted:
            centers.append(most_potential)
            calculated.add(most_potential[1])
            potential_clone = []

            for point in potential:
                heapq.heappush(potential_clone,
                               (cal_new_potential(point, most_potential, X,
                                                  beta), point[1]))
            potential = potential_clone
        else:
            break
    # Question: what is the return value?

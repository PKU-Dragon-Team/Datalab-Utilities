"""The package to determination cluster number and position
"""

import numpy as np
import dask.array as da
import typing as tg
import heapq


def method_1994(X: da.core.Array, Ra: float=1, Rb: float=1.5, epsilon_upper: float=0.5, epsilon_lower: float=0.15) -> tg.Tuple[tg.List[float], tg.List[int]]:
    """funtion that use the 1994 Fuzzy Model Identification Based on Cluster Estimation method
        return the index of Cluster point in X and the potential of them
    """

    def cal_potential(xi: int, X: da.core.Array, alpha: float) -> float:
        """Function calculate the potential of point x become a cluster center
        """
        # Here didn't exclude the xi row. The result is different but it dosn't affect the sequence.
        return da.exp(-alpha * (X - X[xi, :])**2).sum().compute()

    def cal_new_potential(old: tg.Tuple[np.double, int], center: tg.Tuple[np.double, int], X: da.core.Array, beta: float) -> float:
        """Function calculate the updated potential of point x after a new cluster center occurs
        """
        return (-old[0] + center[0] * (-beta * da.linalg.lstsq(X[old[1], :], X[center[1], :]))**2).compute()

    def cal_d_min(xi: int, centers: tg.List[tg.Tuple[np.double, int]], X: da.core.Array) -> float:
        """Function calculate the shortest distance between point X[xi, :] and all previous cluster centers
        """
        distance = []  # heapq
        for _, i in centers:
            heapq.heappush(distance, da.linalg.lstsq(X[xi, :], X[i, :]).compute())

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
        heapq.heappush(potential_clone, (-cal_new_potential(point, first_center, X, beta), point[1]))
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

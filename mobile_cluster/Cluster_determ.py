"""The package to determination cluster number and position
"""

import numpy as np
import typing as tg
import heapq


def method_1994(X: tg.Dict, Ra: float=2.0, Rb: float=3.0) -> (int, tg.Dict):
    """funtion that use the 1994 Fuzzy Model Identification Based on Cluster Estimation method
    """

    def cal_potential(x: np.array,
                      X: np.matrix,
                      alpha: np.double=np.double(1.0)) -> np.double:
        """Function calculate the potential of point x become a cluster center
        """
        return np.sum(np.exp(-alpha * np.square(X - x)))

    alpha = 4 / Ra**2
    beta = 4 / Rb**2

    names, positions = [x for x in zip(*X.items())]
    X = np.matrix(positions)

    potential = []  # heapq
    centers = []
    calculated = set()
    flag = True
    while flag:
        for i, x in enumerate(X):
            if i not in calculated:
                heapq.heappush(potential,
                               (-cal_potential(
                                   np.array(x), X, alpha),
                                i))  # heapq only provides min-heap, WTF
        most_potential = heapq.heappop(potential)
        centers.append(most_potential)
        calculated.add(most_potential[1])
        # TODO: loop
    pass

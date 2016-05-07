"""The package to determination cluster number and position
"""

import numpy as np
import dask.array as da
import typing as tg


def method_1994(X: da.Array, Ra: float=1, Rb: float=1.5, epsilon_upper: float=0.5, epsilon_lower: float=0.15) -> tg.Tuple[tg.List[float], tg.List[int]]:
    """funtion that use the 1994 Fuzzy Model Identification Based on Cluster Estimation method
        return the index of Cluster point in X and the potential of them
    """

    def cal_potential(xi: int, X: da.Array, alpha: float):
        """Function calculate the potential of point x become a cluster center
        """
        return da.exp(-alpha * (X[da.fromfunction(lambda x: x != xi, chunks=(1024, 1024), shape=(X.shape[0], ), dtype=bool), :] - X[xi, :])**2).sum()

    def cal_new_potential(old_i: int, center_i: int, X: da.Array, p: da.Array, beta: float):
        """Function calculate the updated potential of point x after a new cluster center occurs
        """
        return (-p[old_i] + p[center_i] * (-beta * da.linalg.lstsq(X[old_i], X[center_i]))**2)

    def cal_d_min(xi: int, centers: tg.List[tg.Tuple[np.double, int]], X: da.Array):
        """Function calculate the shortest distance between point X[xi, :] and all previous cluster centers
        """
        return da.fromfunction(lambda i: da.linalg.lstsq(X[xi], X[i]), chunks=(1024, 1024), shape=(len(centers, )), dtype=np.double).min()

    alpha = 4 / Ra**2
    beta = 4 / Rb**2

    # calc the first center
    potential = da.fromfunction(lambda i: cal_potential(i, X, alpha), chunks=(1024, 1024), shape=(X.shape[0], ), dtype=np.double)
    first_center_i = potential.argmax().compute()
    first_center_p = potential[first_center_i]
    centers = [(first_center_p, first_center_i), ]  # KDTree is not modifiable, so use plain method
    calculated_count = 1

    potential = da.fromfunction(lambda i: cal_new_potential(i, first_center_i, X, potential, beta), chunks=(1024, 1024), shape=potential.shape, dtype=np.double)

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
            elif cal_d_min(X[most_potential_i, :], centers, X) / Ra + most_potential_p / first_center_p > 1:
                accepted = True
                break
            else:
                potential = da.fromfunction(lambda i: 0 if i == most_potential_i else potential[i], chunks=(1024, 1024), shape=potential.shape, dtype=np.double)  # da.Array is immutable
                most_potential_i = potential.argmax()
                most_potential_p = potential[most_potential_i]
                calculated_count += 1

        if accepted:
            centers.append((most_potential_p, most_potential_i))
            potential = da.fromfunction(lambda i: cal_new_potential(i, first_center_i, X, potential, beta), chunks=(1024, 1024), shape=potential.shape, dtype=np.double)
        else:
            break
    return tuple(zip(*centers))

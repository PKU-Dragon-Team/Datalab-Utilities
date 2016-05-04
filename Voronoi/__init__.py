"""Module to compute and show Voronoi figure
"""
import os
# import typing as tg

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import pandas as pd

__location__ = os.path.join(os.getcwd(), os.path.dirname(os.path.realpath(__file__)))

COUNT_LIMIT = None


def voronoi(cell_info: pd.DataFrame, show: bool=True, color_set: pd.DataFrame=None, sample_limit: int=20, target_axes: matplotlib.axes.Axes=None) -> None:
    """the function to compute and show voronoi figure
    """
    # ramdom sampling
    if sample_limit:
        ramdom_list = np.random.choice(len(cell_info), size=sample_limit)
        cells = np.array((cell_info[["x", "y"]].iloc[ramdom_list]))
        color = np.array((color_set.iloc[ramdom_list]))
    else:
        cells = np.array(cell_info[["x", "y"]])
        color = np.array(color_set)

# compute Voronoi tesselation
    vor = Voronoi(cells)

    # plot
    if target_axes:
        ax = target_axes
    else:
        ax = plt.gca()
    ax.set_aspect(1.)

    # colorize
    for i in range(len(cells)):
        region = vor.regions[vor.point_region[i]]
        if -1 not in region:
            polygon = [vor.vertices[j] for j in region]
            z = list(zip(*polygon))
            # plt.fill((x0, x1, x2, x3, ...), (y0, y1, y2, y3, ...), color=None)
            ax.fill(z[0], z[1], color=color[i])
    if show:
        plt.show()

if __name__ == '__main__':
    cell_info = pd.read_csv(os.path.join(__location__, 'cell_info.csv'), header=None, dtype={0: str, 1: float, 2: float}, nrows=COUNT_LIMIT)
    cell_info.columns = ("name", "x", "y")
    voronoi(cell_info)

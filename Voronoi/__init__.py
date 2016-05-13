"""Module to compute and show Voronoi figure
"""
import os
import typing as tg

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi

__location__ = os.path.join(os.getcwd(), os.path.dirname(os.path.realpath(__file__)))


def voronoi(cell_info: pd.DataFrame, color_set: tg.Sequence=None, target_axes: matplotlib.axes.Axes=None) -> None:
    """the function to compute and show voronoi figure
    cell_info is supposed to have column x and column y which contains the position of each point
    """
    cells = np.asarray(cell_info[["x", "y"]])
    color = np.asarray(color_set)

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
            ax.fill(z[0], z[1], color=color[i])

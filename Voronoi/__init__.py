"""Module to compute and show Voronoi figure
"""
import os
import typing as tg
import json

from .. import NumpyAndPandasEncoder

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi

__location__ = os.path.join(os.getcwd(), os.path.dirname(os.path.realpath(__file__)))


def voronoi_plot(X: np.ndarray, color_set: tg.Optional[tg.Sequence]=None, target_axes: tg.Optional[matplotlib.axes.Axes]=None) -> None:
    """the function to compute and show voronoi figure
    X[:, 0] and X[: 1] is consider as x and y
    """
    cells = np.asfarray(X)
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


def voronoi_dump(cell_info: pd.DataFrame, out_file=tg.TextIO, label: tg.Optional[tg.Sequence]=None, indent: int=4) -> None:
    cells = np.asarray(cell_info[['x', 'y']])
    if label is None:
        labels = np.zeros((cells.shape[0], 1), dtype=int)
    else:
        labels = np.asarray(label)

    vor = Voronoi(cells)

    clusters = np.unique(labels)

    output = {'clusters': clusters}
    for x in clusters:
        output[int(x)] = []

    for i in range(len(cells)):
        region = vor.regions[vor.point_region[i]]
        if -1 not in region:
            output[int(labels[i])].append([vor.vertices[j] for j in region])

    print("var data = ", file=out_file, end='')
    json.dump(output, out_file, cls=NumpyAndPandasEncoder, indent=indent)
    print(";", file=out_file)

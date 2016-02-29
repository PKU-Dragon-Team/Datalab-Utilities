import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas

import os

COUNT_LIMIT = None
SAMPLE_LIMIT = 20

cell_info = pandas.read_csv('cell_info.csv', header=None, dtype={0: str, 1: float, 2:float}, nrows=COUNT_LIMIT)
cell_info.columns = ("name", "x", "y")

# 随机抽取
if SAMPLE_LIMIT:
    cells = np.array([(cell_info["x"][i], cell_info["y"][i]) for i in np.random.choice(len(cell_info), size = SAMPLE_LIMIT)])
else:
    cells = np.array(cell_info["x", "y"])

# compute Voronoi tesselation
vor = Voronoi(cells)

# plot
fig = voronoi_plot_2d(vor)
fig.canvas.set_window_title("基站服务范围示意图")
title = fig.suptitle("cell_info")
ax = fig.gca()
ax.set_aspect(1.)

# colorize
for region in vor.regions:
    if -1 not in region:
        polygon = [vor.vertices[i] for i in region]
        _ = ax.fill(*zip(*polygon)) # plt.fill((x0, x1, x2, x3, ...), (y0, y1, y2, y3, ...), color=None)

plt.show()
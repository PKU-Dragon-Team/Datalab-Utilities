import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

import csv

COUNT_LIMIT = None
SAMPLE_LIMIT = 100

Points = []
with open('cell_info.csv', 'r', encoding='utf_8') as obj_file:
    csv_file = csv.reader(obj_file)
    for cnt, line in enumerate(csv_file):
        if COUNT_LIMIT and cnt >= COUNT_LIMIT:  # 只读取前 k 个点
            break
        Points.append([float(line[1]), float(line[2])])  # make up data points

# 随机抽取 n 个点
if SAMPLE_LIMIT:
    points = np.array([Points[i] for i in np.random.choice(len(Points), size = SAMPLE_LIMIT)])
else:
    points = np.array(Points)

# compute Voronoi tesselation
vor = Voronoi(points)

# plot
voronoi_plot_2d(vor)

# colorize
for region in vor.regions:
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon))

plt.show()
"""the package for clustering mobile_data
"""
import json
import math
import numbers
import os
import sys
import time
import typing as tg
from pathlib import Path
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
import skfuzzy.cluster as skfc
import sklearn.cluster as sklc
import sklearn.preprocessing as sklpp

__location__ = os.path.join(os.getcwd(), os.path.dirname(os.path.realpath(__file__)))
__path__ = Path(__location__)

sys.path.append(str(__path__.parent))

import Voronoi


class NumpyAndPandasEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.matrix)):
            return [self.default(x) for x in obj]
        if isinstance(obj, pd.DataFrame):
            return [self.default(series) for _, series in obj.iterrows()]
        if isinstance(obj, pd.Series):
            return [self.default(val) for _, val in obj.iteritems()]
        if isinstance(obj, numbers.Integral):
            return int(obj)
        if isinstance(obj, (numbers.Real, numbers.Rational)):
            return float(obj)
        return super().default(obj)


def gray2rgb(grayscale: float) -> tg.Tuple:
    """The function to change grayscale to an RGB tuple
    the grayscale is requeired to be in range of [0, 1]
    """
    return (grayscale, grayscale, grayscale)


def fuzzifier_determ(D: int, N: int) -> float:
    """The function to determ the fuzzifier paramter of fuzzy c-means
    """
    return 1 + (1418 / N + 22.05) * D**(-2) + (12.33 / N + 0.243) * D**(-0.0406 * math.log(N) - 0.1134)


def main() -> None:
    with open(os.path.join(__location__, "config.json"), 'r') as config:
        conf = json.load(config)
        HOST = conf['host']
        USER = conf['user']
        PASS = conf['pass']
        DATABASE = conf['database']
        CHARSET = conf['charset']

    connection = pymysql.connect(host=HOST, user=USER, password=PASS, db=DATABASE, charset=CHARSET, cursorclass=pymysql.cursors.DictCursor)

    print("Loding data from SQL…")
    sql = "SELECT usr_count_normalize, x, y FROM loc_week_count_reshape"
    dframe = pd.read_sql(sql, connection).sort_index()
    print("Data loded.")

    print("Building data structure…")
    dframe = dframe.assign(location=lambda X: list(zip(X.x, X.y)))
    dframe['usr_count_normalize'] = dframe['usr_count_normalize'].map(lambda row: json.loads(row))
    dframe.sort_values("location")
    location = dframe.location
    print("Data structure builded.")

    # with open(os.path.join(__location__, "cluster_centers.json"), 'r') as f:
    #     _, cluster_centers = json.load(f)

    print("Preprocessing data…")
    X = sklpp.maxabs_scale(np.asfarray(dframe['usr_count_normalize'].values.tolist()), copy=False)  # type: np.ndarray
    # sample_mask = np.random.choice(X.shape[0], size=1000, replace=False)
    with open(os.path.join(__location__, "middle_dump.json"), "r") as f:
        middle_dump = json.load(f)  # type: tg.Dict["mask", "centers"]
        sample_mask = np.array(middle_dump["mask"])
    # X_sample = X[sample_mask, :]
    print("Data preprocessed.")

    print("Birch model fitting…")
    t0 = time.time()
    birch = sklc.Birch(threshold=0.28)
    birch.fit(X)
    cluster_centers = birch.subcluster_centers_
    cluster_labels = birch.subcluster_labels_
    labels = birch.labels_
    c = len(cluster_centers)
    t1 = time.time()
    print("Birch model fitted.")
    print("%d cluster centers dectected." % c)
    print("Total time: %.2f" % (t1 - t0))

    # dump middle result
    with open(os.path.join(__location__, "middle_dump.json"), 'w') as f:
        json.dump({'mask': sample_mask, 'centers': cluster_centers}, f, cls=NumpyAndPandasEncoder)

    m = fuzzifier_determ(X.shape[1], X.shape[0])

    # use Fuzzy c-means predict to build the initial_matrix
    # cmeans accept (D, N) rather than (N, D)
    # pdb.set_trace()
    u, u0, d, jm, p, fpc = skfc.cmeans_predict(X.T, cluster_centers, m, 1e-3, 100)  # in the function will rotate X WTF

    print("Fuzzy c-means model fitting…")
    t2 = time.time()
    cntr, u, u0, d, jm, p, fpc = skfc.cmeans(X.T, c, m, 1e-3, 100, u)
    t3 = time.time()
    print("Fuzzy c-means model fitted.")
    print("Total time: %.2f" % (t3 - t2))

    print("Plotting result…")
    result = u.T.argmax(axis=1)
    color = (result + 1) / (c + 1)
    rgb_color = [gray2rgb(x) for x in color]
    ax = plt.subplot(1, 1, 1)
    ax.set_autoscale_on(False)
    ax.set_xlim(115.8, 116.9)
    ax.set_ylim(39.6, 40.3)
    Voronoi.voronoi(location, color_set=pd.DataFrame.from_records(rgb_color), target_axes=ax, sample_limit=0, show=False)
    print("Result plotted.")
    plt.show()
    # handles = [mpatches.Patch(color=gray2rgb((i + 1) / (c + 1)), label='cluster %d' % (i + 1)) for i in range(c)]
    # ax.legend(handles=handles, loc=0, borderaxespad=0.0)

    # print("Dumping result…")
    # with open(os.path.join(__location__, 'cluster_centers.json'), 'w', encoding='utf_8') as f:
    #     json.dump(dframe.iloc[sample_mask[cluster_centers], :], f, cls=NumpyAndPandasEncoder)
    # print("Result dumped.")

    connection.close()

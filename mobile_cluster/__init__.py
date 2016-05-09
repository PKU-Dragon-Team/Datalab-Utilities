"""the package for clustering mobile_data
"""
import numpy as np
import pandas as pd
import sklearn.cluster as sklc
import skfuzzy.cluster as skfc
import sklearn.preprocessing as sklpp
import pymysql
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import os
import json
import math
import decimal
import typing as tg
import sys
import time
import numbers
from pathlib import Path

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
    sql = "SELECT usr_count_json, x, y FROM `loc_week_count_reshape`"
    data = pd.read_sql(sql, connection)
    print("Data loded.")

    print("Building data structure…")
    dset = {}
    for i in range(len(data)):
        row = data.iloc[i]
        dset[(row.x, row.y)] = json.loads(row.usr_count_json)

    dframe = pd.DataFrame.from_dict(dset, orient='index')
    location = pd.DataFrame.from_records([row for row in dframe.index])
    location.columns = ("x", "y")
    print("Data structure builded.")

    # with open(os.path.join(__location__, "cluster_centers.json"), 'r') as f:
    #     _, cluster_centers = json.load(f)

    print("Preprocessing data…")
    X = sklpp.maxabs_scale(np.matrix(dframe, dtype=np.double), copy=False)
    sample_mask = np.random.choice(X.shape[0], size=1000, replace=False)
    X_sample = X[sample_mask, :]
    print("Data preprocessed.")

    print("Affinity propagation model fitting…")
    t0 = time.time()
    ap = sklc.AffinityPropagation(damping=.99, max_iter=10000).fit(X_sample)
    cluster_centers = ap.cluster_centers_indices_
    c = len(cluster_centers)
    t1 = time.time()
    print("Affinity propagation model fitted.")
    print("%d cluster centers dectected." % c)
    print("Total time: %.2f" % (t1 - t0))

    ap_result = ap.predict(X)
    init_matrix = np.fromfunction(lambda c, n: ap_result[n] == c, (c, X.shape[0]), dtype=int)

    # print("DBSCAN propagation model fitting…")
    # t0 = time.time()
    # ap = sklc.DBSCAN().fit(X)
    # # result = ap.predict(X)
    # cluster_centers = ap.core_sample_indices_
    # c = len(cluster_centers)
    # t1 = time.time()
    # print("DBSCAN propagation model fitted.")
    # print("%d cluster centers dectected." % c)
    # print("Total time: %.2f" % (t1 - t0))

    print("Fuzzy c-means model fitting…")
    t2 = time.time()
    # centers = np.matrix(X[sample_mask[cluster_centers, :]])
    m = fuzzifier_determ(X.shape[1], X.shape[0])
    fcm = {}
    fcm['cntr'], fcm['u'], fcm['u0'], fcm['d'], fcm['jm'], fcm['p'], fcm['fpc'] = skfc.cmeans(X.T, c, m, 1e-3, 100, init_matrix)  # cmeans accept (D, N) rather than (N, D)
    t3 = time.time()
    print("Fuzzy c-means model fitted.")
    print("Total time: %.2f" % (t3 - t2))
    # kmeans = sklc.MiniBatchKMeans(n_clusters=c, init=centers, max_iter=1e8)
    # result = kmeans.fit_predict(dframe)

    print("Plotting result…")
    result = fcm['u'].T.argmax(axis=1)
    color = (result + 1) / (c + 1)
    rgb_color = [gray2rgb(x) for x in color]
    ax = plt.subplot(1, 1, 1)
    ax.set_autoscale_on(False)
    ax.set_xlim(115.8, 116.9)
    ax.set_ylim(39.6, 40.3)
    Voronoi.voronoi(location, color_set=pd.DataFrame.from_records(rgb_color), target_axes=ax, sample_limit=0, show=False)
    plt.show()
    # handles = [mpatches.Patch(color=gray2rgb((i + 1) / (c + 1)), label='cluster %d' % (i + 1)) for i in range(c)]
    # ax.legend(handles=handles, loc=0, borderaxespad=0.0)
    print("Result plotted.")

    # print("Dumping result…")
    # with open(os.path.join(__location__, 'cluster_centers.json'), 'w', encoding='utf_8') as f:
    #     json.dump(dframe.iloc[sample_mask[cluster_centers], :], f, cls=NumpyAndPandasEncoder)
    # print("Result dumped.")

    connection.close()

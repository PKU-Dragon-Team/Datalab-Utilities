"""the package for clustering mobile_data
"""
import pandas as pd
import sklearn
import sklearn.cluster
import pymysql
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import os
import json
import decimal
import typing as tg
import sys
from pathlib import Path

__location__ = os.path.join(os.getcwd(), os.path.dirname(os.path.realpath(__file__)))
__path__ = Path(__location__)

sys.path.append(str(__path__.parent))

import Voronoi


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def gray2rgb(grayscale: float) -> tg.Tuple:
    """The function to change grayscale to an RGB tuple
    the grayscale is requeired to be in range of [0, 1]
    """
    return (grayscale, grayscale, grayscale)


def main() -> None:
    with open(os.path.join(__location__, "config.json"), 'r') as config:
        conf = json.load(config)
        HOST = conf['host']
        USER = conf['user']
        PASS = conf['pass']
        DATABASE = conf['database']
        CHARSET = conf['charset']

    connection = pymysql.connect(host=HOST, user=USER, password=PASS, db=DATABASE, charset=CHARSET, cursorclass=pymysql.cursors.DictCursor)

    sql = "SELECT location, day_type, shour, usr_count, x, y FROM `loc_week_count`"
    data = pd.read_sql(sql, connection)

    dset = {}
    for i in range(len(data)):
        row = data.iloc[i]
        try:
            dset[(row.x, row.y)]
        except KeyError:
            dset[(row.x, row.y)] = [0] * 2 * 24
        if row.day_type == "weekday":
            dset[(row.x, row.y)][row.shour] = row.usr_count
        elif row.day_type == "weekend":
            dset[(row.x, row.y)][row.shour + 24] = row.usr_count

    dframe = pd.DataFrame.from_dict(dset, orient='index')
    location = pd.DataFrame.from_records([row for row in dframe.index])
    location.columns = ("x", "y")

    cluster_range = range(5, 11)
    for i, c in enumerate(cluster_range, start=1):
        ax = plt.subplot(4, 2, i)
        kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=c)
        result = kmeans.fit_predict(dframe)
        color = (result + 1) / (c + 1)
        rgb_color = [gray2rgb(x) for x in color]
        handles = [mpatches.Patch(color=gray2rgb((i + 1) / (c + 1)), label='cluster %d' % (i + 1)) for i in range(c)]
        if i % 2 != 0:
            anchor = (-0.3, 0.5)
            loc = 7
        else:
            anchor = (1.05, 0.5)
            loc = 6
        ax.legend(handles=handles, bbox_to_anchor=anchor, loc=loc, borderaxespad=0.0)
        ax.set_autoscale_on(False)
        ax.set_xlim(115.8, 116.9)
        ax.set_ylim(39.6, 40.3)
        Voronoi.voronoi(location, color_set=pd.DataFrame.from_records(rgb_color), target_axes=ax, show=False)
    plt.show()

    connection.close()

"""The script to calc cluster center and store in json file
"""

import numpy as np
import sklearn.preprocessing as sklpp
import pandas as pd
import dask.array as da
import pymysql

import os
import json
import decimal
import typing as tg
import sys
from pathlib import Path

import Cluster_determ as cd

__location__ = os.path.join(os.getcwd(), os.path.dirname(os.path.realpath(__file__)))
__path__ = Path(__location__)

sys.path.append(str(__path__.parent))


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


if __name__ == '__main__':
    with open(os.path.join(__location__, "config.json"), 'r') as config:
        conf = json.load(config)
        HOST = conf['host']
        USER = conf['user']
        PASS = conf['pass']
        DATABASE = conf['database']
        CHARSET = conf['charset']

    connection = pymysql.connect(host=HOST, user=USER, password=PASS, db=DATABASE, charset=CHARSET, cursorclass=pymysql.cursors.DictCursor)

    sql = "SELECT usr_count_json, x, y FROM `loc_week_count_reshape`"
    data = pd.read_sql(sql, connection)

    dset = {}
    for i in range(len(data)):
        row = data.iloc[i]
        dset[(row.x, row.y)] = json.loads(row.usr_count_json)

    dframe = pd.DataFrame.from_dict(dset, orient='index')
    # location = pd.DataFrame.from_records([row for row in dframe.index])
    # location.columns = ("x", "y")

    X = da.from_array(sklpp.minmax_scale(np.matrix(dframe, dtype=np.double), copy=False), (1000, 1000))

    result = cd.method_1994(X)

    with open(os.path.join(__location__, 'cluster_centers.json'), 'w', encoding='utf_8') as f:
        json.dump(result, f, indent=4)

    connection.close()

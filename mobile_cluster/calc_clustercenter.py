"""The script to calc cluster center and store in json file
"""

import numpy as np
import sklearn.preprocessing as sklpp
import pandas as pd
import pymysql

import os
import json
from pathlib import Path
import time

import Cluster_determ as cd

__location__ = os.path.join(os.getcwd(), os.path.dirname(os.path.realpath(__file__)))
__path__ = Path(__location__)

if __name__ == '__main__':
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
    # location = pd.DataFrame.from_records([row for row in dframe.index])
    # location.columns = ("x", "y")
    print("Data structure builded.")

    print("Preprocessing data…")
    X = sklpp.minmax_scale(np.matrix(dframe, dtype=np.double), copy=False)
    print("Data preprocessed.")

    print("Calculating cluster centers…")
    t0 = time.time()
    result = cd.method_1994(X)
    t1 = time.time()
    print("Cluster centers alculated.")
    print("Total time: %.2f" % (t1 - t0))

    print("Dumping result…")
    with open(os.path.join(__location__, 'cluster_centers.json'), 'w', encoding='utf_8') as f:
        json.dump(result, f, indent=4)
    print("Result dumped.")

    connection.close()

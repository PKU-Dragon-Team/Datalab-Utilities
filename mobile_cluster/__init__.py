from .. import Voronoi

# import numpy as np
import pandas as pd
import sklearn
import sklearn.cluster
import pymysql

import os
import json
import decimal
import typing as tg
__location__ = os.path.join(os.getcwd(),
                            os.path.dirname(os.path.realpath(__file__)))


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

    connection = pymysql.connect(host=HOST,
                                 user=USER,
                                 password=PASS,
                                 db=DATABASE,
                                 charset=CHARSET,
                                 cursorclass=pymysql.cursors.DictCursor)

    sql = "SELECT location, day_type, shour, usr_count, x, y FROM `loc_week_count_copy`"
    data = pd.read_sql(sql, connection)

    dset = {}
    for i in range(len(data)):
        row = data.iloc[i]
        try:
            _ = dset[(row.x, row.y)]
        except KeyError:
            dset[(row.x, row.y)] = [0] * 2 * 24
        if row.day_type == "weekday":
            dset[(row.x, row.y)][row.shour] = row.usr_count
        elif row.day_type == "weekend":
            dset[(row.x, row.y)][row.shour + 24] = row.usr_count

    dframe = pd.DataFrame.from_dict(dset, orient='index')
    location = pd.DataFrame.from_records([row for row in dframe.index])
    location.columns = ("x", "y")

    kmeans = sklearn.cluster.MiniBatchKMeans()
    result = kmeans.fit_predict(dframe)
    color = (result + 1) / 8

    rgb_color = []
    for x in color:
        rgb_color.append(gray2rgb(x))

    Voronoi.voronoi(location, color_set=pd.DataFrame.from_records(rgb_color))

    connection.close()

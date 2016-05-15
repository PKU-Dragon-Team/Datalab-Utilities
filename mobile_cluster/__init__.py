"""the package for clustering mobile_data
"""
import json
import math
import numbers
import typing as tg

import numpy as np
import pandas as pd
import pymysql


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


def dataloder(database_conf: tg.Dict, select_sql: str) -> pd.DataFrame:
    HOST = database_conf['host']
    USER = database_conf['user']
    PASS = database_conf['pass']
    DATABASE = database_conf['database']
    CHARSET = database_conf['charset']
    connection = pymysql.connect(host=HOST, user=USER, password=PASS, db=DATABASE, charset=CHARSET, cursorclass=pymysql.cursors.DictCursor)
    dframe = pd.read_sql(select_sql, connection)
    connection.close()
    return dframe

"""the package for clustering mobile_data
"""
import math
import typing as tg

import numpy as np
import pandas as pd
import pymysql


def gray2rgb(grayscale: float) -> tg.Tuple:
    """The function to change grayscale to an RGB tuple
    the grayscale is requeired to be in range of [0, 1]
    """
    return (grayscale, grayscale, grayscale)


def fuzzifier_determ(D: int, N: int) -> float:
    """The function to determ the fuzzifier paramter of fuzzy c-means
    """
    return 1 + (1418 / N + 22.05) * D**(-2) + (12.33 / N + 0.243) * D**(-0.0406 * math.log(N) - 0.1134)


def data_loder(database_conf: tg.Dict, select_sql: str) -> pd.DataFrame:
    connection = pymysql.connect(host=database_conf['host'], user=database_conf['user'], password=database_conf['pass'], db=database_conf['database'], charset=database_conf['charset'], cursorclass=pymysql.cursors.DictCursor)
    dframe = pd.read_sql(select_sql, connection)
    connection.close()
    return dframe

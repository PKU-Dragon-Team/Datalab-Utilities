"""plot dataset
"""
import numpy as np
import pandas as pd
import sklearn.preprocessing as sklpp
import pymysql
import matplotlib.pyplot as plt

import os
import json

__location__ = os.path.join(os.getcwd(), os.path.dirname(os.path.realpath(__file__)))

if __name__ == "__main__":
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
    location = pd.DataFrame.from_records([row for row in dframe.index])
    location.columns = ("x", "y")

    X = np.array(dframe, dtype=np.double)
    X_scale = sklpp.maxabs_scale(X)

    ax = plt.subplot(3, 2, 1)
    plt.boxplot(X)
    plt.grid(True)
    ax.set_xlim(-1, 49)
    ax.set_title('box')

    ax = plt.subplot(3, 2, 2)
    plt.boxplot(X_scale)
    plt.grid(True)
    ax.set_xlim(-1, 49)
    ax.set_title('box (scaled)')

    ax = plt.subplot(3, 2, 3)
    x = np.arange(0, X.shape[1])
    y = X.mean(axis=0)
    plt.errorbar(x, y, yerr=0.25 * y)
    y = np.median(X, axis=0)
    plt.errorbar(x, y, yerr=0.25 * y)
    plt.grid(True)
    ax.set_xlim(-1, 49)
    ax.set_title('mean, median')

    ax = plt.subplot(3, 2, 4)
    x = np.arange(0, X_scale.shape[1])
    y = X_scale.mean(axis=0)
    plt.errorbar(x, y, yerr=0.25 * y)
    y = np.median(X_scale, axis=0)
    plt.errorbar(x, y, yerr=0.25 * y)
    plt.grid(True)
    ax.set_xlim(-1, 49)
    ax.set_title('mean, median (scaled)')

    ax = plt.subplot(3, 2, 5)
    x = np.arange(0, X.shape[1])
    y = X.std(axis=0)
    plt.errorbar(x, y, yerr=0.25 * y)
    plt.grid(True)
    ax.set_xlim(-1, 49)
    ax.set_title('std')

    ax = plt.subplot(3, 2, 6)
    x = np.arange(0, X_scale.shape[1])
    y = np.std(X_scale, axis=0)
    plt.errorbar(x, y, yerr=0.25 * y)
    plt.grid(True)
    ax.set_xlim(-1, 49)
    ax.set_title('std (scaled)')

    plt.show()
    connection.close()

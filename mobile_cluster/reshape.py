import pandas as pd
import pymysql

import os
import json

__location__ = os.path.join(os.getcwd(),
                            os.path.dirname(os.path.realpath(__file__)))

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

sql = "SELECT location, day_type, shour, usr_count, x, y FROM `loc_week_count`"
data = pd.read_sql(sql, connection)

dset = {}
for i in range(len(data)):
    row = data.iloc[i]
    try:
        dset[(row.x, row.y)]
    except KeyError:
        dset[(row.x, row.y)] = {'location': row.location,
                                'usr_count': [0] * 2 * 24}
    if row.day_type == "weekday":
        dset[(row.x, row.y)]['usr_count'][row.shour] = row.usr_count
    elif row.day_type == "weekend":
        dset[(row.x, row.y)]['usr_count'][row.shour + 24] = row.usr_count

with connection.cursor() as cursor:
    insert_sql = "INSERT INTO loc_week_count_reshape (location, usr_count_json, x, y) VALUES %s;"
    value_template = "('%s', '%s', %8.5f, %8.5f)"
    value_list = []
    n = 0
    for key, val in dset.items():
        if n < 100:
            value_list.append(value_template %
                              (val['location'],
                               json.dumps([int(x) for x in val['usr_count']]),
                               key[0], key[1]))
            n += 1
        else:
            cursor.execute(insert_sql % ", ".join(value_list))
            value_list.clear()
            n = 0

connection.commit()
connection.close()

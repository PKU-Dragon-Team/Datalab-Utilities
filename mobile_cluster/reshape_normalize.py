import pandas as pd
import pymysql

import os
import json

__location__ = os.path.join(os.getcwd(), os.path.dirname(os.path.realpath(__file__)))

with open(os.path.join(__location__, "config.json"), 'r') as config:
    conf = json.load(config)
    HOST = conf['host']
    USER = conf['user']
    PASS = conf['pass']
    DATABASE = conf['database']
    CHARSET = conf['charset']

connection = pymysql.connect(host=HOST, user=USER, password=PASS, db=DATABASE, charset=CHARSET, cursorclass=pymysql.cursors.DictCursor)

sql = "SELECT location, usr_count_json FROM loc_week_count_reshape"
data = pd.read_sql(sql, connection)
data['usr_count_json'] = data['usr_count_json'].map(lambda row: json.dumps(list(pd.Series(json.loads(row)) / pd.Series([22] * 24 + [9] * 24))))

with connection.cursor() as cursor:
    update_sql = "UPDATE loc_week_count_reshape SET usr_count_normalize=%s WHERE location=%s;" % ('"%s"', '"%s"')

    for row in data.iterrows():
        cursor.execute(update_sql % (row[1]['usr_count_json'], row[1]['location']))

connection.commit()
connection.close()

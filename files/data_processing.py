import pandas as pd
import datetime
import csv
import re
import matplotlib.pyplot as plt

start = datetime.date(2018,12,2)

df = pd.read_json("crypto_data.json")
df.to_csv('crypto_data.csv')
#pd.set_option('display.max_rows',622)

date_list = []
with open("crypto_data.csv") as f:
    for date in csv.reader(f):
        date_list.append(str(date[1]))

date_list.pop(0)

processed_date_list = []

for date in date_list:
    dt = datetime.datetime.fromisoformat(date[:-2]).date()
    processed_date_list.append(dt)


df_new = df.rename(index=lambda s: processed_date_list[s])
df_new.to_csv("crypto_data2.csv")

#print(df_new)
"""
data = pd.read_csv('crypto_data2.csv', index_col=0)
data["Date"] = pd.to_datetime(data["Date"])
data_i = data.set_index('Date')
"""
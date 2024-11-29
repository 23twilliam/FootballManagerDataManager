import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import re
plt.ion()

#df = pd.read_csv('T10GKs.csv')
#df = pd.read_csv('T5DMs.csv')
df = pd.read_csv('AllPlayers.csv')
df.fillna(0, inplace=True)
#-----
#This code snippet excludes players that are "Not for Sale"
mask = df['Transfer Value'] == 'Not for Sale'
df = df[~mask]
mask = df['AP'] == 'Â£0'
df = df[~mask]
mask = df['AP'] == 0
df = df[~mask]

df = df.replace({'Â':''}, regex=True)
df = df.replace({'-':''}, regex=True)
#-----
def value_to_float(x):
    x = re.sub(r"[£]", "", x)
    res = [i for j in x.split() for i in (j, ' ')][:-1]
    if type(x) == float or type(x) == int:
        return x
    if (len(res) == 3):
        if 'K' in res[0] and 'M' in res[2]:
            res[0] = re.sub(r"[K]", "", res[0])
            res[2] = re.sub(r"[M]", "", res[2])
            x = ( (float(res[0]) * 1000 ) + (float(res[2]) * 1000000) ) / 2
            return x
    if 'K' in x:
        if len(x) > 1:
            res[0] = re.sub(r"[K]", "", res[0])
            if len(res) == 3:
                res[2] = re.sub(r"[K]", "", res[2])
                x = (float(res[0]) + float(res[2])) / 2
            else:
                x = float(res[0])
            return x * 1000
        return 1000.0
    if 'M' in x:
        if len(x) > 1:
            res[0] = re.sub(r"[M]", "", res[0])
            if len(res) == 3:
                res[2] = re.sub(r"[M]", "", res[2])
                x = (float(res[0]) + float(res[2])) / 2
            else:
                x = float(res[0])
            return x * 1000000
        return 1000000.0
    return 0.0

def AP_to_float(x):
    x = re.sub(r"[£]", "", x)
    if type(x) == float or type(x) == int:
        return x
    if 'K' in x:
        if len(x) > 1:
            x = re.sub(r"[K]", "", x)
            return float(x) * 1000
        return 1000.0
    if 'M' in x:
        if len(x) > 1:
            x = re.sub(r"[M]", "", x)
            return float(x) * 1000000
        return 1000000.0
    return 0.0

df['Transfer Value'] = df['Transfer Value'].apply(value_to_float)
df = df.sort_values(by=['Transfer Value'], ascending=False)
df['AP'] = df['AP'].apply(AP_to_float)
mask = df['Transfer Value'] == 0.0
df = df[~mask]

print(df['Transfer Value'].corr(df['AP']))
plt.scatter(df['AP'], df['Transfer Value'], s=1, c='Red', marker='.')
plt.show(block=True)
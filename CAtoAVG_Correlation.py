import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import re
plt.ion()
import matplotlib.colors as colors
import math
from scipy import stats
#df = pd.read_csv('T10GKs.csv')
#df = pd.read_csv('T5DMs.csv')
df = pd.read_csv('T5CBwCA.csv')
df.fillna(0, inplace=True)
#-----
#This code snippet excludes players that are "Not for Sale"
mask = df['Transfer Value'] == 'Not for Sale'
df = df[~mask]




df_obj = df.select_dtypes('object')
df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip('%'))
#df['Dist/90'] = df['Dist/90'].apply(lambda x: x.strip('km'))
#df['Dist/90'] = pd.to_numeric(df['Dist/90'], errors='coerce')
df['Pas %'] = pd.to_numeric(df['Pas %'], errors='coerce')
#df['xSv %'] = pd.to_numeric(df['xSv %'], errors='coerce')
#df['Sv %'] = pd.to_numeric(df['Sv %'], errors='coerce')
df['Hdr %'] = pd.to_numeric(df['Hdr %'], errors='coerce')
#df['Pens Saved Ratio'] = pd.to_numeric(df['Pens Saved Ratio'], errors='coerce')
def CB(df):
    #mask = df['Transfer Value'] > valMax
    df2 = df
    #Positives
    PrPsAVG = np.nanmean(df['Pr passes/90'])
    PsWnAVG = np.nanmean(df['Poss Won/90'])
    PsCmpAVG = np.nanmean(df['Ps C/90'])
    DrbAVG = np.nanmean(df['Drb/90'])

    SprntAVG = np.nanmean(df['Sprints/90'])
    HdrAVG = np.nanmean(df['Hdr %'])
    PasAVG = np.nanmean(df['Pas %'])
    IntAVG = np.nanmean(df['Int/90'])

    PrPsImpct = df['Pr passes/90'].corr(df['Pts/Gm'])
    PsWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])
    PsCmpImpct = df['Ps C/90'].corr(df['Pts/Gm'])
    DrbImpct = df['Drb/90'].corr(df['Pts/Gm'])

    SprntImpct = df['Sprints/90'].corr(df['Pts/Gm'])
    HdrsImpct = df['Hdr %'].corr(df['Pts/Gm'])
    PasImpct = df['Pas %'].corr(df['Pts/Gm'])
    IntImpct = df['Int/90'].corr(df['Pts/Gm'])

    TotalPImpact = PrPsImpct + PsWnImpct + PsCmpImpct + DrbImpct + SprntImpct + HdrsImpct + PasImpct + IntImpct

    #Negatives

    PosLstAVG = np.nanmean(df['Poss Lost/90'])

    PosLstImpct = df['Poss Lost/90'].corr(df['Pts/Gm'])

    #AVGs

    avg = (((((df2['Pr passes/90']/PrPsAVG) * PrPsImpct) + ((df2['Poss Won/90']/PsWnAVG) * PsWnImpct) +
           ((df2['Ps C/90']/PsCmpAVG) * PsCmpImpct) + ((df2['Drb/90']/DrbAVG) * DrbImpct) +
           ((df2['Sprints/90']/SprntAVG) * SprntImpct) + ((df2['Hdr %']/HdrAVG) * HdrsImpct) +
           ((df2['Pas %']/PasAVG) * PasImpct)) + ((df2['Int/90']/IntAVG) * IntImpct) / TotalPImpact) +
           ((df2['Poss Lost/90'])/PosLstAVG) * PosLstImpct)
    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg
avg = CB(df)

plt.scatter(df['CA'],avg)
print(df['CA'].corr(avg))
#plt.scatter(150, (-0.65 + (0.02*150)), color="red")



#polynomial fit with degree = 2
model = np.poly1d(np.polyfit(df['CA'], avg, 2))

#add fitted polynomial line to scatterplot
polyline = np.linspace(100, 200, 50)
plt.plot(polyline, model(polyline))
print(model)
plt.show(block=True)

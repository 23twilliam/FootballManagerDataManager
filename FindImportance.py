import matplotlib.pyplot as plt
import pandas as pd

plt.ion()

df = pd.read_csv('Defenders.csv')
df.fillna(0, inplace=True)
# -----
# This code snippet excludes players that are "Not for Sale"
# mask = df['Transfer Value'] == 'Not for Sale'
# df = df[~mask]
# -----

df_obj = df.select_dtypes('object')

df[df_obj.columns] = df_obj.apply(lambda x: x.str.rstrip('%'))

df_obj = df.select_dtypes('object')
df[df_obj.columns] = df_obj.apply(lambda x: x.str.rstrip('km'))

df['Dist/90'] = pd.to_numeric(df['Dist/90'], errors='coerce')
df['Pas %'] = pd.to_numeric(df['Pas %'], errors='coerce')
#df['Mins/Gl'] = pd.to_numeric(df['Mins/Gl'], errors='coerce')

#df['xSv %'] = pd.to_numeric(df['xSv %'], errors='coerce')
#df['Sv %'] = pd.to_numeric(df['Sv %'], errors='coerce')
#df['Pens Saved Ratio'] = pd.to_numeric(df['Pens Saved Ratio'], errors='coerce')

plt.bar('Aer A/90', df['Pts/Gm'].corr(df['Aer A/90']))
plt.bar('Asts/90', df['Pts/Gm'].corr(df['Asts/90']))
plt.bar('Blk/90', df['Pts/Gm'].corr(df['Blk/90']))
plt.bar('Ch C/90', df['Pts/Gm'].corr(df['Ch C/90']))
plt.bar('Clr/90', df['Pts/Gm'].corr(df['Clr/90']))
plt.bar('Crs A/90', df['Pts/Gm'].corr(df['Crs A/90']))
plt.bar('Cr C/90', df['Pts/Gm'].corr(df['Cr C/90']))
plt.bar('Dist/90', df['Pts/Gm'].corr(df['Dist/90']))
plt.bar('Drb/90', df['Pts/Gm'].corr(df['Drb/90']))
#plt.bar('Hdrs L/90', df['Pts/Gm'].corr(df['Hdrs L/90']))
plt.bar('Hdrs W/90', df['Pts/Gm'].corr(df['Hdrs W/90']))
plt.bar('Sprints/90', df['Pts/Gm'].corr(df['Sprints/90']))
plt.bar('Int/90', df['Pts/Gm'].corr(df['Int/90']))
plt.bar('K Hdrs/90', df['Pts/Gm'].corr(df['K Hdrs/90']))
plt.bar('K Ps/90', df['Pts/Gm'].corr(df['K Ps/90']))
plt.bar('K Tck/90', df['Pts/Gm'].corr(df['K Tck/90']))
plt.bar('OPCr %', df['Pts/Gm'].corr(df['OPCr %']))
#plt.bar('OPCrs A/90', df['Pts/Gm'].corr(df['OPCrs A/90']))
#plt.bar('OPCrs C/90', df['Pts/Gm'].corr(df['OPCrs C/90']))
plt.bar('OPKP/90', df['Pts/Gm'].corr(df['OPKP/90']))
plt.bar('Ps A/90', df['Pts/Gm'].corr(df['Ps A/90']))
plt.bar('Ps C/90', df['Pts/Gm'].corr(df['Ps C/90']))
plt.bar('Poss Lost/90', df['Pts/Gm'].corr(df['Poss Lost/90']))
plt.bar('Pas %', df['Pts/Gm'].corr(df['Pas %']))
plt.bar('Poss Won/90', df['Pts/Gm'].corr(df['Poss Won/90']))
plt.bar('Pres A/90', df['Pts/Gm'].corr(df['Pres A/90']))
plt.bar('Pres C/90', df['Pts/Gm'].corr(df['Pres C/90']))
plt.bar('Pr passes/90', df['Pts/Gm'].corr(df['Pr passes/90']))
plt.bar('Shts Blckd/90', df['Pts/Gm'].corr(df['Shts Blckd/90']))
plt.bar('Shots Outside Box/90', df['Pts/Gm'].corr(df['Shots Outside Box/90']))
plt.bar('ShT/90', df['Pts/Gm'].corr(df['ShT/90']))
plt.bar('Shot/90', df['Pts/Gm'].corr(df['Shot/90']))
plt.bar('Tck/90', df['Pts/Gm'].corr(df['Tck/90']))
plt.bar('Tck R', df['Pts/Gm'].corr(df['Tck R']))
plt.bar('Hdr %', df['Pts/Gm'].corr(df['Hdr %']))
#plt.bar('Gls/90', df['Pts/Gm'].corr(df['Gls/90']))
#plt.bar('NPxG/90', df['Pts/Gm'].corr(df['NPxG/90']))
plt.axhline(y=0.1, color='r', linestyle='-')

plt.show(block=True)

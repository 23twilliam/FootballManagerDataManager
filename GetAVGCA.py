import numpy as np
import pandas as pd

avg = np.array(float)

str = ('Austria'+str(i)+'.csv')
df = pd.read_csv(str)
AustriaAVG = np.nanmean(df['CA'])
avg = np.append(avg,AustriaAVG)

df = pd.read_csv('Belarus.csv')
BelarusAVG = np.nanmean(df['CA'])
avg = np.append(avg,BelarusAVG)

df = pd.read_csv('Belgium.csv')
BelgiumAVG = np.nanmean(df['CA'])
avg = np.append(avg,BelgiumAVG)

df = pd.read_csv('Bulgaria.csv')
BulgariaAVG = np.nanmean(df['CA'])
avg = np.append(avg,BulgariaAVG)

df = pd.read_csv('Croatia.csv')
CroatiaAVG = np.nanmean(df['CA'])
avg = np.append(avg,CroatiaAVG)

df = pd.read_csv('Czechia.csv')
CzechiaAVG = np.nanmean(df['CA'])
avg = np.append(avg,CzechiaAVG)

df = pd.read_csv('Denmark.csv')
DenmarkAVG = np.nanmean(df['CA'])
avg = np.append(avg,DenmarkAVG)

df = pd.read_csv('England.csv')
EnglandAVG = np.nanmean(df['CA'])
avg = np.append(avg,EnglandAVG)

df = pd.read_csv('Finland.csv')
FinlandAVG = np.nanmean(df['CA'])
avg = np.append(avg,FinlandAVG)

df = pd.read_csv('France.csv')
FranceAVG = np.nanmean(df['CA'])
avg = np.append(avg,FranceAVG)

df = pd.read_csv('Germany.csv')
GermanyAVG = np.nanmean(df['CA'])
avg = np.append(avg,GermanyAVG)

df = pd.read_csv('Gibraltar.csv')
GibraltarAVG = np.nanmean(df['CA'])
avg = np.append(avg,GibraltarAVG)

df = pd.read_csv('Greece.csv')
GreeceAVG = np.nanmean(df['CA'])
avg = np.append(avg,GreeceAVG)

df = pd.read_csv('Hungary.csv')
HungaryAVG = np.nanmean(df['CA'])
avg = np.append(avg,HungaryAVG)

df = pd.read_csv('Iceland.csv')
IcelandAVG = np.nanmean(df['CA'])
avg = np.append(avg,IcelandAVG)

df = pd.read_csv('Israel.csv')
IsraelAVG = np.nanmean(df['CA'])
avg = np.append(avg,IsraelAVG)

df = pd.read_csv('Italy.csv')
ItalyAVG = np.nanmean(df['CA'])
avg = np.append(avg,ItalyAVG)

df = pd.read_csv('Latvia.csv')
LatviaAVG = np.nanmean(df['CA'])
avg = np.append(avg,LatviaAVG)

df = pd.read_csv('Netherlands.csv')
NetherlandsAVG = np.nanmean(df['CA'])
avg = np.append(avg,NetherlandsAVG)

df = pd.read_csv('NorthernIreland.csv')
NorthernIrelandAVG = np.nanmean(df['CA'])
avg = np.append(avg,NorthernIrelandAVG)

df = pd.read_csv('Norway.csv')
NorwayAVG = np.nanmean(df['CA'])
avg = np.append(avg,NorwayAVG)

df = pd.read_csv('Poland.csv')
PolandAVG = np.nanmean(df['CA'])
avg = np.append(avg,PolandAVG)

df = pd.read_csv('Portugal.csv')
PortugalAVG = np.nanmean(df['CA'])
avg = np.append(avg,PortugalAVG)

df = pd.read_csv('RepublicOfIreland.csv')
RepublicOfIrelandAVG = np.nanmean(df['CA'])
avg = np.append(avg,RepublicOfIrelandAVG)

df = pd.read_csv('Romania.csv')
RomaniaAVG = np.nanmean(df['CA'])
avg = np.append(avg,RomaniaAVG)

df = pd.read_csv('Scotland.csv')
ScotlandAVG = np.nanmean(df['CA'])
avg = np.append(avg,ScotlandAVG)

df = pd.read_csv('Serbia.csv')
SerbiaAVG = np.nanmean(df['CA'])
avg = np.append(avg,SerbiaAVG)

df = pd.read_csv('Slovakia.csv')
SlovakiaAVG = np.nanmean(df['CA'])
avg = np.append(avg,SlovakiaAVG)

df = pd.read_csv('Slovenia.csv')
SloveniaAVG = np.nanmean(df['CA'])
avg = np.append(avg,SloveniaAVG)

df = pd.read_csv('Spain.csv')
SpainAVG = np.nanmean(df['CA'])
avg = np.append(avg,SpainAVG)

df = pd.read_csv('Sweden.csv')
SwedenAVG = np.nanmean(df['CA'])
avg = np.append(avg,SwedenAVG)

df = pd.read_csv('Switzerland.csv')
SwitzerlandAVG = np.nanmean(df['CA'])
avg = np.append(avg,SwitzerlandAVG)

df = pd.read_csv('Turkey.csv')
TurkeyAVG = np.nanmean(df['CA'])
avg = np.append(avg,TurkeyAVG)

df = pd.read_csv('Ukraine.csv')
UkraineAVG = np.nanmean(df['CA'])
avg = np.append(avg,UkraineAVG)

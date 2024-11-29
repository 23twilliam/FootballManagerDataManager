import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import re
plt.ion()
import matplotlib.colors as colors
import math
from scipy import stats

df = pd.read_csv('leaguesTesting.csv')
df2 = pd.read_csv('Coeffs/2025.csv')
# USE PLAYER ID NOT NAME
currentDiv = (df.loc[df['Name'].eq('Hernán Guillén'), 'Division']).item()
DivisionNames = {'Austrian Premier Division': 'Austria',
               'Belarusian Highest League': 'Belarus',
               'Jupiler Pro League': 'Belgium',
               'Bulgarian First League': 'Bulgaria',
               'Croatian First League': 'Croatia',
               'Czech First Division': 'Czechia',
               '3F Superliga': 'Denmark',
               'English Premier Division': 'England',
               'Finnish Premier League': 'Finland',
               'Ligue 1 Uber Eats': 'France',
               'Bundesliga': 'Germany',
               'Gibraltar Football League': 'Gibraltar',
               'Greek Super League 1': 'Greece',
               'Hungary Division I': 'Hungary',
               'Icelandic Premier division': 'Iceland',
               'Irish Premier Division': 'Republic of Ireland',
               'Israeli Premier League': 'Israel',
               'Italian Serie A': 'Italy',
               'Optibet Virsliga': 'Latvia',
               'Sports Direct Premiership': 'Northern Ireland',
               'Eredivisie': 'Netherlands',
               'Norwegian Premier Division': 'Norway',
               'PKO Bank Polski Ekstraklasa': 'Poland',
               'Portugal Premier League': 'Portugal',
               'Romanian First League': 'Romania',
               'Russian Premier League': 'Russia',
               'cinch Premiership': 'Scotland',
               'Serbian SuperLeague': 'Serbia',
               'Slovak First Division': 'Slovakia',
               'Slovenian First League': 'Slovenia',
               'Spanish First Division': 'Spain',
               'Swedish Premier Division': 'Sweden',
               'Swiss Super League': 'Switzerland',
               'Turkish Super League': 'Turkey',
               'Ukrainian Premier League': 'Ukraine',
               'JD Cymru Premier': 'Wales',
               }
print(DivisionNames[currentDiv])
currentCountry = DivisionNames[currentDiv]
currentCoeff = (df2.loc[df2['Nation'].eq(currentCountry), 'EUFA']).item()
currentCA = (df2.loc[df2['Nation'].eq(currentCountry), 'CA']).item()

print(currentCoeff)
print(currentCA)

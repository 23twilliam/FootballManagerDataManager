import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Book1.csv')
df = df.sort_values(by=['Nation'])
df.reset_index(drop = True, inplace = True)
df = df['CA']

df2 = pd.read_csv('CoeffY1.csv')
df4 = pd.read_csv('CoeffY2.csv')
df5 = pd.read_csv('CoeffY3.csv')

# here df3 gets the resulting merged DataFrame
df2 = df2[['Nation', 'Coef']]
df4 = df4[['Nation', 'Coef']]
df5 = df5[['Nation', 'Coef']]

df3 = pd.concat([df2, df4, df5, df])
print(df3)

# Remove Unwanted Nations
df3 = df3[df3.Nation != 'Albania']
df3 = df3[df3.Nation != 'Andorra']
df3 = df3[df3.Nation != 'Armenia']
df3 = df3[df3.Nation != 'Azerbaijan']
df3 = df3[df3.Nation != 'Bosnia and Herzegovina']
df3 = df3[df3.Nation != 'Cyprus']
df3 = df3[df3.Nation != 'Estonia']
df3 = df3[df3.Nation != 'Faroe Islands']
df3 = df3[df3.Nation != 'Georgia']
df3 = df3[df3.Nation != 'Kazakhstan']
df3 = df3[df3.Nation != 'Kosovo']
df3 = df3[df3.Nation != 'Liechtenstein']
df3 = df3[df3.Nation != 'Lithuania']
df3 = df3[df3.Nation != 'Luxembourg']
df3 = df3[df3.Nation != 'Malta']
df3 = df3[df3.Nation != 'Moldova']
df3 = df3[df3.Nation != 'Montenegro']
df3 = df3[df3.Nation != 'North Macedonia']
df3 = df3[df3.Nation != 'San Marino']
df3 = df3[df3.Nation != 'Wales']
df3 = df3.reset_index(drop=True)
df3 = df3.merge(df, how='outer', left_index=True, right_index=True)
df3 = df3.drop(['CA_x'], axis=1)
df3.rename(columns={'CA_y': 'CA'}, inplace=True)
df3.to_csv('CoeffTotal.csv')
print(df3)
df8 = pd.read_csv('Book1.csv')
plt.scatter(df8['EUFA'], df8['CA'])
xlog_data = np.log(df8['EUFA'])
curve = np.polyfit(xlog_data, df8['CA'], 1)
print(curve)
y = 21.822925 * xlog_data + 38.19678636
plt.plot(df8['EUFA'], y)
plt.show(block=True)


DivisionNames = {
    'Austrain Premier Division': 'Austria',
    'Belarus Premier Division': 'Belarus',
    'Jupiler Pro League': 'Belgium',
    'Bulgarian First League': 'Bulgaria',
    'Croatian First League': 'Croatia',
    'Czech First Division': 'Czechia',
    '3F Superliga': 'Denmark',
    'England Premier Division': 'England',
    'Finnish Premier Division': 'Finland',
    'Ligue 1 Uber Eats': 'France',
    'Bundesliga': 'Germany',
    'Gibraltar Football League': 'Gibraltar',
    'Greek Super League 1': 'Greece',
    'Hungary Division I': 'Hungary',
    'Icelandic Premier Division': 'Iceland',
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
    'Russia Premier League': 'Russia',
    'Cinch Premiership': 'Scotland',
    'Serbian SuperLeague': 'Serbia',
    'Slovak First Division': 'Slovakia',
    'Slovenian First League': 'Slovenia',
    'Spanish First Division': 'Spain',
    'Swedish Premier Division': 'Swedish',
    'Swiss Super League': 'Switzerland',
    'Turkish Super League': 'Turkey',
    'Ukrainian Premier League': 'Ukraine',
    'JD Cymru Premier': 'Wales',
}
print(DivisionNames["JD Cymru Premier"])
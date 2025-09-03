import numpy as np

def defensive_midfielder(df, valMax):
    mask = df['Transfer Value'] > valMax
    df2 = df[~mask]

    # Positives

    # Averages

    PsWnAVG = np.nanmean(df['Poss Won/90'])
    AstsAVG = np.nanmean(df['Asts/90'])
    DrbAVG = np.nanmean(df['Drb/90'])
    OPKPAVG = np.nanmean(df['OP-KP/90'])

    PsCmpAVG = np.nanmean(df['Ps C/90'])
    ShtOBAVG = np.nanmean(df['Shots Outside Box/90'])
    ShTAVG = np.nanmean(df['ShT/90'])
    OPCrsAVG = np.nanmean(df['OP-Crs C/90'])

    # Averages

    PsWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])
    AstsImpct = df['Asts/90'].corr(df['Pts/Gm'])
    DrbImpct = df['Drb/90'].corr(df['Pts/Gm'])
    OPKPImpct = df['OP-KP/90'].corr(df['Pts/Gm'])

    PsCmpImpct = df['Ps C/90'].corr(df['Pts/Gm'])
    ShtOBImpct = df['Shots Outside Box/90'].corr(df['Pts/Gm'])
    ShTImpct = df['ShT/90'].corr(df['Pts/Gm'])
    OPCrsImpct = df['OP-Crs C/90'].corr(df['Pts/Gm'])

    DMImpct = PsWnImpct + AstsImpct + DrbImpct + OPKPImpct + PsCmpImpct + ShtOBImpct + ShTImpct + OPCrsImpct

    avg = (((df2['Poss Won/90'] / PsWnAVG) * PsWnImpct) + ((df2['Asts/90'] / AstsAVG) * AstsImpct) +
           ((df2['Drb/90'] / DrbAVG) * DrbImpct) + ((df['OP-KP/90'] / OPKPAVG) * OPKPImpct) +
           ((df2['Ps C/90'] / PsCmpAVG) * PsCmpImpct) + ((df2['Shots Outside Box/90'] / ShtOBAVG) * ShtOBImpct) +
           ((df2['ShT/90'] / ShTAVG) * ShTImpct) + ((df2['OP-Crs C/90'] / OPCrsAVG) * OPCrsImpct)) / DMImpct
    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg
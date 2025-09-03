import numpy as np

def defender(df, valMax):
    mask = df['Transfer Value'] > valMax
    df2 = df[~mask]
    #Positives
    PrPsAVG = np.nanmean(df['Pr passes/90'])
    PsWnAVG = np.nanmean(df['Poss Won/90'])
    #PsCmpAVG = np.nanmean(df['Ps C/90'])
    DrbAVG = np.nanmean(df['Drb/90'])

    SprntAVG = np.nanmean(df['Sprints/90'])
    HdrAVG = np.nanmean(df['Hdr %'])
    PasAVG = np.nanmean(df['Pas %'])
    IntAVG = np.nanmean(df['Int/90'])

    PrPsImpct = df['Pr passes/90'].corr(df['Pts/Gm'])
    PsWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])
    #PsCmpImpct = df['Ps C/90'].corr(df['Pts/Gm'])
    DrbImpct = df['Drb/90'].corr(df['Pts/Gm'])

    SprntImpct = df['Sprints/90'].corr(df['Pts/Gm'])
    HdrsImpct = df['Hdr %'].corr(df['Pts/Gm'])
    PasImpct = df['Pas %'].corr(df['Pts/Gm'])
    IntImpct = df['Int/90'].corr(df['Pts/Gm'])

    TotalPImpact = PrPsImpct + PsWnImpct + DrbImpct + SprntImpct + HdrsImpct + PasImpct + IntImpct

    #Negatives

    PosLstAVG = np.nanmean(df['Poss Lost/90'])

    PosLstImpct = df['Poss Lost/90'].corr(df['Pts/Gm'])

    #AVGs

    avg = (((((df2['Pr passes/90'] / PrPsAVG) * PrPsImpct) + ((df2['Poss Won/90'] / PsWnAVG) * PsWnImpct) +
             ((df2['Drb/90'] / DrbAVG) * DrbImpct) +
             ((df2['Sprints/90'] / SprntAVG) * SprntImpct) + ((df2['Hdr %'] / HdrAVG) * HdrsImpct) +
             ((df2['Pas %'] / PasAVG) * PasImpct)) + ((df2['Int/90'] / IntAVG) * IntImpct) / TotalPImpact) +
           ((df2['Poss Lost/90']) / PosLstAVG) * PosLstImpct)
    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg
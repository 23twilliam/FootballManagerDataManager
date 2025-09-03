import numpy as np

def fullback(df, valMax):
    mask = df['Transfer Value'] > valMax
    df2 = df[~mask]

    # Positives

    # Averages

    PsWnAVG = np.nanmean(df['Poss Won/90'])
    PrPsAVG = np.nanmean(df['Pr passes/90'])
    PsCmpAVG = np.nanmean(df['Pas %'])
    ShTAVG = np.nanmean(df['ShT/90'])

    ChCAVG = np.nanmean(df['Ch C/90'])
    AstsAVG = np.nanmean(df['Asts/90'])
    GlsAVG = np.nanmean(df['Gls/90'])

    # Impacts

    PsWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])
    PrPsImpct = df['Pr passes/90'].corr(df['Pts/Gm'])
    PsCmpImpct = df['Pas %'].corr(df['Pts/Gm'])
    ShTImpct = df['ShT/90'].corr(df['Pts/Gm'])

    ChCImpct = df['Ch C/90'].corr(df['Pts/Gm'])
    AstsImpct = df['Asts/90'].corr(df['Pts/Gm'])
    GlsImpct = df['Gls/90'].corr(df['Pts/Gm'])

    TotalPImpact = PsWnImpct + PrPsImpct + PsCmpImpct + ChCImpct + AstsImpct + GlsImpct + ShTImpct

    # Negatives

    # Averages

    PossLostAVG = np.nanmean(df['Poss Lost/90'])

    # Impacts

    PossLostImpct = df['Poss Lost/90'].corr(df['Pts/Gm'])

    # Avg

    avg = ((((df2['Asts/90'] / AstsAVG) * AstsImpct) + ((df2['Ch C/90'] / ChCAVG) * ChCImpct) +
            ((df2['Poss Won/90'] / PsWnAVG) * PsWnImpct) + ((df2['Pr passes/90'] / PrPsAVG) * PrPsImpct) +
            ((df2['Pas %'] / PsCmpAVG) * PsCmpImpct) + ((df2['ShT/90'] / ShTAVG) * ShTImpct) +
            ((df2['Gls/90'] / GlsAVG) * GlsImpct) / TotalPImpact) +
           ((df2['Poss Lost/90'] / PossLostAVG) * PossLostImpct))

    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg
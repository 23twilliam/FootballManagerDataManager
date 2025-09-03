import numpy as np

def wingback(df, valMax):
    mask = df['Transfer Value'] > valMax
    df2 = df[~mask]

    # Positives

    # Averages

    PsWnAVG = np.nanmean(df['Poss Won/90'])
    PrPsAVG = np.nanmean(df['Pr passes/90'])
    PsCmpAVG = np.nanmean(df['Pas %'])
    HdrCmpAVG = np.nanmean(df['Hdr %'])

    ChCAVG = np.nanmean(df['Ch C/90'])
    AstsAVG = np.nanmean(df['Asts/90'])
    TckAVG = np.nanmean(df['Tck/90'])

    # Impacts

    PsWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])
    PrPsImpct = df['Pr passes/90'].corr(df['Pts/Gm'])
    PsCmpImpct = df['Pas %'].corr(df['Pts/Gm'])
    HdrCmpImpct = df['Hdr %'].corr(df['Pts/Gm'])

    ChCImpct = df['Ch C/90'].corr(df['Pts/Gm'])
    AstsImpct = df['Asts/90'].corr(df['Pts/Gm'])
    TckImpct = df['Tck/90'].corr(df['Pts/Gm'])

    TotalPImpact = PsWnImpct + PrPsImpct + PsCmpImpct + HdrCmpImpct + ChCImpct + AstsImpct + TckImpct

    # Negatives

    # Averages

    PossLostAVG = np.nanmean(df['Poss Lost/90'])

    # Impacts

    PossLostImpct = df['Poss Lost/90'].corr(df['Pts/Gm'])

    # Avg

    avg = ((((df2['Asts/90'] / AstsAVG) * AstsImpct) + ((df2['Ch C/90'] / ChCAVG) * ChCImpct) +
             ((df2['Poss Won/90'] / PsWnAVG) * PsWnImpct) + ((df2['Pr passes/90'] / PrPsAVG) * PrPsImpct) +
             ((df2['Pas %'] / PsCmpAVG) * PsCmpImpct) + ((df2['Hdr %'] / HdrCmpAVG) * HdrCmpImpct) +
             ((df2['Tck/90'] / TckAVG) * TckImpct) / TotalPImpact) +
             ((df2['Poss Lost/90'] / PossLostAVG) * PossLostImpct))

    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg
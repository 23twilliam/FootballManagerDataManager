import numpy as np

def attacking_midfielder(df, valMax):
    mask = df['Transfer Value'] > valMax
    df2 = df[~mask]

    # Positives

    # Averages

    ShTAVG = np.nanmean(df['ShT/90'])
    GlsAVG = np.nanmean(df['Gls/90'])
    PossWnAVG = np.nanmean(df['Poss Won/90'])
    OPKPAVG = np.nanmean(df['OP-KP/90'])

    HdrCmpAVG = np.nanmean(df['Hdr %'])
    ChCAVG = np.nanmean(df['Ch C/90'])
    AstsAVG = np.nanmean(df['Asts/90'])
    SprntAVG = np.nanmean(df['Sprints/90'])

    DrbAVG = np.nanmean(df['Drb/90'])

    # Impacts

    ShTImpct = df['ShT/90'].corr(df['Pts/Gm'])
    GlsImpct = df['Gls/90'].corr(df['Pts/Gm'])
    PossWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])
    OPKPImpct = df['OP-KP/90'].corr(df['Pts/Gm'])

    HdrCmpImpct = df['Hdr %'].corr(df['Pts/Gm'])
    ChCImpct = df['Ch C/90'].corr(df['Pts/Gm'])
    AstsImpct = df['Asts/90'].corr(df['Pts/Gm'])
    SprntImpct = df['Sprints/90'].corr(df['Pts/Gm'])

    DrbImpct = df['Drb/90'].corr(df['Pts/Gm'])

    TotalPImpact = ShTImpct + GlsImpct + PossWnImpct + OPKPImpct + HdrCmpImpct + DrbImpct + ChCImpct + AstsImpct + SprntImpct

    # Negatives

    # Averages

    PossLostAVG = np.nanmean(df['Poss Lost/90'])

    # Impacts

    PossLostImpct = df['Poss Lost/90'].corr(df['Pts/Gm'])

    # Overall avg

    avg = (((((df2['Asts/90'] / AstsAVG) * AstsImpct) + ((df2['Ch C/90'] / ChCAVG) * ChCImpct) +
             ((df2['Gls/90'] / GlsAVG) * GlsImpct) + ((df2['Poss Won/90'] / PossWnAVG) * PossWnImpct) +
             ((df2['Hdr %'] / HdrCmpAVG) * HdrCmpImpct) + ((df2['ShT/90'] / ShTAVG) * ShTImpct) +
             ((df2['Drb/90'] / DrbAVG) * DrbImpct) + ((df2['OP-KP/90'] / OPKPAVG) * OPKPImpct) +
            ((df2['Sprints/90'] / SprntAVG) * SprntImpct)) / TotalPImpact) +
           ((df2['Poss Lost/90'] / PossLostAVG) * PossLostImpct))
    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg
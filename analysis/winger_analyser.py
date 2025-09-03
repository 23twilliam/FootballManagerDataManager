import numpy as np

def winger(df, valMax):
    mask = df['Transfer Value'] > valMax
    df2 = df[~mask]

    # Positives

    # Averages

    PossWnAVG = np.nanmean(df['Poss Won/90'])
    ChCAVG = np.nanmean(df['Ch C/90'])
    AstsAVG = np.nanmean(df['Asts/90'])
    SprintAVG = np.nanmean(df['Sprints/90'])

    DrbAVG = np.nanmean(df['Drb/90'])
    PrPassAVG = np.nanmean(df['Pr passes/90'])
    ShTAVG = np.nanmean(df['ShT/90'])
    OPKPAVG = np.nanmean(df['OP-KP/90'])

    OPCrAVG = np.nanmean(df['OP-Cr %'])
    PresAAvg = np.nanmean(df['Pres A/90'])

    # Impacts

    PossWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])
    ChCImpct = df['Ch C/90'].corr(df['Pts/Gm'])
    SprintImpct = df['Sprints/90'].corr(df['Pts/Gm'])
    AstsImpct = df['Asts/90'].corr(df['Pts/Gm'])

    DrbImpct = df['Drb/90'].corr(df['Pts/Gm'])
    PrPassImpct = df['Pr passes/90'].corr(df['Pts/Gm'])
    ShTImpct = df['ShT/90'].corr(df['Pts/Gm'])
    OPKPImpct = df['OP-KP/90'].corr(df['Pts/Gm'])

    OPCrImpct = df['OP-Cr %'].corr(df['Pts/Gm'])
    PresAImpct = df['Pres A/90'].corr(df['Pts/Gm'])

    TotalPImpact = PossWnImpct + ChCImpct + SprintImpct + AstsImpct + DrbImpct + PrPassImpct + ShTImpct + OPKPImpct + OPCrImpct + PresAImpct

    # Negatives

    # Averages

    PossLostAVG = np.nanmean(df['Poss Lost/90'])

    # Impacts

    PossLostImpct = df['Poss Lost/90'].corr(df['Pts/Gm'])

    avg = (((((df2['Asts/90'] / AstsAVG) * AstsImpct) + ((df2['Ch C/90'] / ChCAVG) * ChCImpct) +
             ((df2['Sprints/90'] / SprintAVG) * SprintImpct) + ((df2['Poss Won/90'] / PossWnAVG) * PossWnImpct) +
             ((df2['Drb/90'] / DrbAVG) * DrbImpct) + ((df2['ShT/90'] / ShTAVG) * ShTImpct) +
             ((df2['Pres A/90'] / PresAAvg) * PresAImpct) + ((df2['Pr passes/90'] / PrPassAVG) * PrPassImpct) +
             ((df2['OP-KP/90'] / OPKPAVG) * OPKPImpct)) + ((df2['OP-Cr %'] / OPCrAVG) * OPCrImpct) / TotalPImpact) +
           ((df2['Poss Lost/90'] / PossLostAVG) * PossLostImpct))
    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg
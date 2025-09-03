import numpy as np

def central_midfielder(df, valMax):
    mask = df['Transfer Value'] > valMax
    df2 = df[~mask]

    # Positives

    # Averages
    AstsAVG = np.nanmean(df['Asts/90'])
    ChCAVG = np.nanmean(df['Ch C/90'])
    SprntAVG = np.nanmean(df['Sprints/90'])
    PossWnAVG = np.nanmean(df['Poss Won/90'])

    ShTAVG = np.nanmean(df['ShT/90'])
    PresAAVG = np.nanmean(df['Pres A/90'])
    DrbAVG = np.nanmean(df['Drb/90'])
    IntAVG = np.nanmean(df['Int/90'])

    OPKPAVG = np.nanmean(df['OP-KP/90'])

    # Impacts

    AstsImpct = df['Asts/90'].corr(df['Pts/Gm'])
    ChCImpct = df['Ch C/90'].corr(df['Pts/Gm'])
    SprntImpct = df['Sprints/90'].corr(df['Pts/Gm'])
    PossWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])

    ShTImpct = df['ShT/90'].corr(df['Pts/Gm'])
    PresAImpct = df['Pres A/90'].corr(df['Pts/Gm'])
    DrbImpct = df['Drb/90'].corr(df['Pts/Gm'])
    IntImpct = df['Int/90'].corr(df['Pts/Gm'])

    OPKPImpct = df['OP-KP/90'].corr(df['Pts/Gm'])

    TotalPImpact = AstsImpct + ChCImpct + SprntImpct + PossWnImpct + ShTImpct + PresAImpct + DrbImpct + IntImpct + OPKPImpct

    # Negatives

    # Averages

    PossLostAVG = np.nanmean(df['Poss Lost/90'])

    # Impact

    PossLostImpct = df['Poss Lost/90'].corr(df['Pts/Gm'])

    #AVGs

    avg = (((((df2['Asts/90'] / AstsAVG) * AstsImpct) + ((df2['Ch C/90'] / ChCAVG) * ChCImpct) +
             ((df2['Sprints/90'] / SprntAVG) * SprntImpct) + ((df2['Poss Won/90'] / PossWnAVG) * PossWnImpct) +
             ((df2['Drb/90'] / DrbAVG) * DrbImpct) + ((df2['ShT/90'] / ShTAVG) * ShTImpct) +
             ((df2['Pres A/90'] / PresAAVG) * PresAImpct) + ((df2['Int/90'] / IntAVG) * IntImpct) +
             ((df2['OP-KP/90'] / OPKPAVG) * OPKPImpct)) / TotalPImpact) +
           ((df2['Poss Lost/90'] / PossLostAVG) * PossLostImpct))
    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import re

plt.ion()
import math
from scipy import stats

#df = pd.read_csv('T10GKs.csv')
#df = pd.read_csv('T5DMs.csv')
#df = pd.read_csv('T10CBs.csv')



# -----
def value_to_float(x):
    x = re.sub(r"[£]", "", x)
    res = [i for j in x.split() for i in (j, ' ')][:-1]
    if type(x) == float or type(x) == int:
        return x
    if (len(res) == 3):
        if 'K' in res[0] and 'M' in res[2]:
            res[0] = re.sub(r"[K]", "", res[0])
            res[2] = re.sub(r"[M]", "", res[2])
            x = ((float(res[0]) * 1000) + (float(res[2]) * 1000000)) / 2
            return x
    if 'K' in x:
        if len(x) > 1:
            res[0] = re.sub(r"[K]", "", res[0])
            if len(res) == 3:
                res[2] = re.sub(r"[K]", "", res[2])
                x = (float(res[0]) + float(res[2])) / 2
            else:
                x = float(res[0])
            return x * 1000
        return 1000.0
    if 'M' in x:
        if len(x) > 1:
            res[0] = re.sub(r"[M]", "", res[0])
            if len(res) == 3:
                res[2] = re.sub(r"[M]", "", res[2])
                x = (float(res[0]) + float(res[2])) / 2
            else:
                x = float(res[0])
            return x * 1000000
        return 1000000.0
    return 0.0






#df['Pens Saved Ratio'] = pd.to_numeric(df['Pens Saved Ratio'], errors='coerce')


def keeper(df):

    mask = df['Transfer Value'] > valMax
    df2 = df[~mask]

    # sweeper keeper stats
    PosWnAVG = np.nanmean(df['Poss Won/90'])
    PsCmpAVG = np.nanmean(df['Ps C/90'])
    PsCmpRtAVG = np.nanmean(df['Pas %'])

    PosWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])
    PsCmpImpct = df['Ps C/90'].corr(df['Pts/Gm'])
    PsCmpRtImpct = df['Pas %'].corr(df['Pts/Gm'])

    # reg keeper stats
    XgPrevAVG = np.nanmean(df['xGP/90'])
    PenSvAVG = np.nanmean(df['Pens Saved Ratio'])

    SvAVG = np.nanmean(df['Sv %']) / np.nanmean(df['xSv %'])
    SvRatio = df['Sv %'] / df['xSv %']
    df = df.assign(SvRatio=SvRatio)

    XgPrevImpct = df['xGP/90'].corr(df['Pts/Gm'])
    PenSvImpct = df['Pens Saved Ratio'].corr(df['Pts/Gm'])
    SvRatioImpct = df['SvRatio'].corr(df['Pts/Gm'])

    GKImpct = PosWnImpct + PsCmpImpct + PsCmpRtImpct + XgPrevImpct + PenSvImpct + SvRatioImpct

    avg = (((df2['SvRatio'] / SvAVG) * SvRatioImpct) + ((df2['xGP/90'] / XgPrevAVG) * XgPrevImpct) +
           ((df2['Pens Saved Ratio'] / PenSvAVG) * PenSvImpct) + ((df2['Poss Won/90'] / PosWnAVG) * PosWnImpct) +
           ((df2['Ps C/90'] / PsCmpAVG) * PsCmpImpct) + ((df2['Pas %'] / PsCmpRtAVG) * PsCmpRtImpct)) / GKImpct
    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg

def defender(df):
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

def fullback(df):
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

def wingback(df):
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

def defensiveMidfielder(df):
    mask = df['Transfer Value'] > valMax
    df2 = df[~mask]

    # Positives

    # Averages

    PsWnAVG = np.nanmean(df['Poss Won/90'])
    AstsAVG = np.nanmean(df['Asts/90'])
    DrbAVG = np.nanmean(df['Drb/90'])
    OPKPAVG = np.nanmean(df['OPKP/90'])

    PsCmpAVG = np.nanmean(df['Ps C/90'])
    ShtOBAVG = np.nanmean(df['Shots Outside Box/90'])
    ShTAVG = np.nanmean(df['ShT/90'])
    OPCrsAVG = np.nanmean(df['OPCrs C/90'])

    # Averages

    PsWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])
    AstsImpct = df['Asts/90'].corr(df['Pts/Gm'])
    DrbImpct = df['Drb/90'].corr(df['Pts/Gm'])
    OPKPImpct = df['OPKP/90'].corr(df['Pts/Gm'])

    PsCmpImpct = df['Ps C/90'].corr(df['Pts/Gm'])
    ShtOBImpct = df['Shots Outside Box/90'].corr(df['Pts/Gm'])
    ShTImpct = df['ShT/90'].corr(df['Pts/Gm'])
    OPCrsImpct = df['OPCrs C/90'].corr(df['Pts/Gm'])

    DMImpct = PsWnImpct + AstsImpct + DrbImpct + OPKPImpct + PsCmpImpct + ShtOBImpct + ShTImpct + OPCrsImpct

    avg = (((df2['Poss Won/90'] / PsWnAVG) * PsWnImpct) + ((df2['Asts/90'] / AstsAVG) * AstsImpct) +
           ((df2['Drb/90'] / DrbAVG) * DrbImpct) + ((df['OPKP/90'] / OPKPAVG) * OPKPImpct) +
           ((df2['Ps C/90'] / PsCmpAVG) * PsCmpImpct) + ((df2['Shots Outside Box/90'] / ShtOBAVG) * ShtOBImpct) +
           ((df2['ShT/90'] / ShTAVG) * ShTImpct) + ((df2['OPCrs C/90'] / OPCrsAVG) * OPCrsImpct)) / DMImpct
    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg

def midfielder(df):
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

    OPKPAVG = np.nanmean(df['OPKP/90'])

    # Impacts

    AstsImpct = df['Asts/90'].corr(df['Pts/Gm'])
    ChCImpct = df['Ch C/90'].corr(df['Pts/Gm'])
    SprntImpct = df['Sprints/90'].corr(df['Pts/Gm'])
    PossWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])

    ShTImpct = df['ShT/90'].corr(df['Pts/Gm'])
    PresAImpct = df['Pres A/90'].corr(df['Pts/Gm'])
    DrbImpct = df['Drb/90'].corr(df['Pts/Gm'])
    IntImpct = df['Int/90'].corr(df['Pts/Gm'])

    OPKPImpct = df['OPKP/90'].corr(df['Pts/Gm'])

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
             ((df2['OPKP/90'] / OPKPAVG) * OPKPImpct)) / TotalPImpact) +
           ((df2['Poss Lost/90'] / PossLostAVG) * PossLostImpct))
    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg

def attackingMidfielder(df):
    mask = df['Transfer Value'] > valMax
    df2 = df[~mask]

    # Positives

    # Averages

    ShTAVG = np.nanmean(df['ShT/90'])
    GlsAVG = np.nanmean(df['Gls/90'])
    PossWnAVG = np.nanmean(df['Poss Won/90'])
    OPKPAVG = np.nanmean(df['OPKP/90'])

    HdrCmpAVG = np.nanmean(df['Hdr %'])
    ChCAVG = np.nanmean(df['Ch C/90'])
    AstsAVG = np.nanmean(df['Asts/90'])
    SprntAVG = np.nanmean(df['Sprints/90'])

    DrbAVG = np.nanmean(df['Drb/90'])

    # Impacts

    ShTImpct = df['ShT/90'].corr(df['Pts/Gm'])
    GlsImpct = df['Gls/90'].corr(df['Pts/Gm'])
    PossWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])
    OPKPImpct = df['OPKP/90'].corr(df['Pts/Gm'])

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
             ((df2['Drb/90'] / DrbAVG) * DrbImpct) + ((df2['OPKP/90'] / OPKPAVG) * OPKPImpct) +
            ((df2['Sprints/90'] / SprntAVG) * SprntImpct)) / TotalPImpact) +
           ((df2['Poss Lost/90'] / PossLostAVG) * PossLostImpct))
    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg

def winger(df):
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
    OPKPAVG = np.nanmean(df['OPKP/90'])

    OPCrAVG = np.nanmean(df['OPCr %'])
    PresAAvg = np.nanmean(df['Pres A/90'])

    # Impacts

    PossWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])
    ChCImpct = df['Ch C/90'].corr(df['Pts/Gm'])
    SprintImpct = df['Sprints/90'].corr(df['Pts/Gm'])
    AstsImpct = df['Asts/90'].corr(df['Pts/Gm'])

    DrbImpct = df['Drb/90'].corr(df['Pts/Gm'])
    PrPassImpct = df['Pr passes/90'].corr(df['Pts/Gm'])
    ShTImpct = df['ShT/90'].corr(df['Pts/Gm'])
    OPKPImpct = df['OPKP/90'].corr(df['Pts/Gm'])

    OPCrImpct = df['OPCr %'].corr(df['Pts/Gm'])
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
             ((df2['OPKP/90'] / OPKPAVG) * OPKPImpct)) + ((df2['OPCr %'] / OPCrAVG) * OPCrImpct) / TotalPImpact) +
           ((df2['Poss Lost/90'] / PossLostAVG) * PossLostImpct))
    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg

def striker(df):
    mask = df['Transfer Value'] > valMax
    df2 = df[~mask]

    # Positives

    # Averages

    ShTAVG = np.nanmean(df['ShT/90'])
    GlsAVG = np.nanmean(df['Gls/90'])
    PossWnAVG = np.nanmean(df['Poss Won/90'])
    OPKPAVG = np.nanmean(df['OPKP/90'])

    HdrCmpAVG = np.nanmean(df['Hdr %'])
    DistAVG = np.nanmean(df['Dist/90'])
    ChCAVG = np.nanmean(df['Ch C/90'])
    AstsAVG = np.nanmean(df['Asts/90'])

    # Impacts

    ShTImpct = df['ShT/90'].corr(df['Pts/Gm'])
    GlsImpct = df['Gls/90'].corr(df['Pts/Gm'])
    PossWnImpct = df['Poss Won/90'].corr(df['Pts/Gm'])
    OPKPImpct = df['OPKP/90'].corr(df['Pts/Gm'])

    HdrCmpImpct = df['Hdr %'].corr(df['Pts/Gm'])
    DistImpct = df['Dist/90'].corr(df['Pts/Gm'])
    ChCImpct = df['Ch C/90'].corr(df['Pts/Gm'])
    AstsImpct = df['Asts/90'].corr(df['Pts/Gm'])

    TotalPImpact = ShTImpct + GlsImpct + PossWnImpct + OPKPImpct + HdrCmpImpct + DistImpct + ChCImpct + AstsImpct

    # Negatives

    # Averages

    PossLostAVG = np.nanmean(df['Poss Lost/90'])

    # Impacts

    PossLostImpct = df['Poss Lost/90'].corr(df['Pts/Gm'])

    #Overall avg

    avg = (((((df2['Asts/90'] / AstsAVG) * AstsImpct) + ((df2['Ch C/90'] / ChCAVG) * ChCImpct) +
            ((df2['Gls/90'] / GlsAVG) * GlsImpct) + ((df2['Poss Won/90'] / PossWnAVG) * PossWnImpct) +
            ((df2['Hdr %'] / HdrCmpAVG) * HdrCmpImpct) + ((df2['ShT/90'] / ShTAVG) * ShTImpct) +
            ((df2['Dist/90'] / DistAVG) * DistImpct) + ((df2['OPKP/90'] / OPKPAVG) * OPKPImpct)) / TotalPImpact) +
           ((df2['Poss Lost/90'] / PossLostAVG) * PossLostImpct))
    avg.columns = ['avg']
    avg.fillna(0, inplace=True)
    return avg

fileLocation = str(input('Enter file location: '))

df = pd.read_csv(fileLocation)
df.fillna(0, inplace=True)
# -----
# This code snippet excludes players that are "Not for Sale"
mask = df['Transfer Value'] == 'Not for Sale'
df = df[~mask]
df['Transfer Value'] = df['Transfer Value'].apply(value_to_float)
df = df.sort_values(by=['Transfer Value'], ascending=False)


df_obj = df.select_dtypes('object')
df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip('%'))
df['Dist/90'] = df['Dist/90'].apply(lambda x: x.strip('km'))
df['Dist/90'] = pd.to_numeric(df['Dist/90'], errors='coerce')
df['Pas %'] = pd.to_numeric(df['Pas %'], errors='coerce')
df['OPCr %'] = pd.to_numeric(df['OPCr %'], errors='coerce')
#df['xSv %'] = pd.to_numeric(df['xSv %'], errors='coerce')
#df['Sv %'] = pd.to_numeric(df['Sv %'], errors='coerce')
df['Hdr %'] = pd.to_numeric(df['Hdr %'], errors='coerce')

# mask df here and add df2 to input to functions
position = int(input('Enter position: \n '
                     '1. Keeper \n '
                     '2. Halfback \n '
                     '3. Fullback \n '
                     '4. Wingback \n '
                     '5. Defensive Midfielder \n '
                     '6. Midfielder \n '
                     '7. Attacking Midfielder \n '
                     '8. Winger \n '
                     '9. Striker \n '))

valMax = int(input("What is the maximum value you would consider"))
mask = df['Transfer Value'] > valMax
df2 = df[~mask]

match position:
    case 1:
        avg = keeper(df)
    case 2:
        avg = defender(df)
    case 3:
        avg = fullback(df)
    case 4:
        avg = wingback(df)
    case 5:
        avg = defensiveMidfielder(df)
    case 6:
        avg = midfielder(df)
    case 7:
        avg = attackingMidfielder(df)
    case 8:
        avg = winger(df)
    case 9:
        avg = striker(df)
    case _:
        avg = None  # Default case if position doesn't match any

avg.columns = ['avg']
r = np.nanmean(avg)
df2.loc[:, 'avg'] = avg.values

df2_obj = df2.select_dtypes('object')
df2[df2_obj.columns] = df2_obj.apply(lambda x: x.str.strip('%'))
df2['Hdr %'] = pd.to_numeric(df2['Hdr %'], errors='coerce')
df2['Pas %'] = pd.to_numeric(df2['Pas %'], errors='coerce')

fig, ax = plt.subplots()

x = df2['Transfer Value'].values
y = df2['avg'].values
labels = df2['Name'].values

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
        }
#cdict = ["#ff5454","#cc3300"]
#my_cmap = colors.ListedColormap(cdict)
cmap = plt.get_cmap('RdYlGn')
norm = plt.Normalize(y.min(), y.max())
line_colors = cmap(norm(y))

for i, label in enumerate(df2['Name']):
    plt.scatter(x[i], y[i], label=labels[i], s=50, c=line_colors[i], marker='.')

plt.ylim(0, (max(avg) + max(avg) / 20))
plt.tight_layout()

c2 = mplcursors.cursor(hover=True)
plt.setp(ax.spines.values(), color="white")
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Transfer Value')
ax.xaxis.label.set_color('white')


@c2.connect("add")
def _(sel):
    sel.annotation.get_bbox_patch().set(color="#FFFFFF", ec="#999999", snap=True, boxstyle="round,pad=0.3")
    #arrowstyle="-", color='#333333',fc="white",
    sel.annotation.arrow_patch.set(visible=False)
    sel.annotation.set_text(sel.artist.get_label())


c3 = mplcursors.cursor()


@c3.connect("add")
def _1(sel):
    label = sel.artist.get_label()
    #'Transfer Value'
    plt.close(fig)
    fig2, ax2 = plt.subplots()
    plt.bar('Progressive Passes',
            100,
            color='#333333')
    plt.bar('Progressive Passes',
            stats.percentileofscore(df['Pr passes/90'], (df2.loc[df2['Name'] == label, 'Pr passes/90']), kind='rank'),
            color='lightblue')
    plt.annotate(
        math.trunc(
            stats.percentileofscore(df['Pr passes/90'], (df2.loc[df2['Name'] == label, 'Pr passes/90']), kind='rank')[
                0]),
        ('Progressive Passes',
         stats.percentileofscore(df['Pr passes/90'], (df2.loc[df2['Name'] == label, 'Pr passes/90']), kind='rank')),
        ha='center',
        color='white',
        xytext=(
            'Progressive Passes',
            stats.percentileofscore(df['Pr passes/90'], (df2.loc[df2['Name'] == label, 'Pr passes/90']),
                                    kind='rank') + 1),
        fontsize=12)

    plt.bar('Possession Won',
            stats.percentileofscore(df['Poss Won/90'], (df2.loc[df2['Name'] == label, 'Poss Won/90']), kind='rank'),
            color='lightblue')
    plt.annotate(
        math.trunc(
            stats.percentileofscore(df['Poss Won/90'], (df2.loc[df2['Name'] == label, 'Poss Won/90']), kind='rank')[0]),
        ('Possession Won',
         stats.percentileofscore(df['Poss Won/90'], (df2.loc[df2['Name'] == label, 'Poss Won/90']), kind='rank')),
        ha='center',
        color='white',
        xytext=(
            'Possession Won',
            stats.percentileofscore(df['Poss Won/90'], (df2.loc[df2['Name'] == label, 'Poss Won/90']),
                                    kind='rank') + 1),
        fontsize=12)

    plt.bar('Sprints',
            stats.percentileofscore(df['Sprints/90'], (df2.loc[df2['Name'] == label, 'Sprints/90']), kind='rank'),
            color='lightblue')
    plt.annotate(math.trunc(
        stats.percentileofscore(df['Sprints/90'], (df2.loc[df2['Name'] == label, 'Sprints/90']), kind='rank')[0]),
                 ('Sprints', stats.percentileofscore(df['Sprints/90'], (df2.loc[df2['Name'] == label, 'Sprints/90']),
                                                     kind='rank')),
                 ha='center',
                 color='white',
                 xytext=('Sprints',
                         stats.percentileofscore(df['Sprints/90'], (df2.loc[df2['Name'] == label, 'Sprints/90']),
                                                 kind='rank') + 1),
                 fontsize=12)

    plt.bar('Dribbles', stats.percentileofscore(df['Drb/90'], (df2.loc[df2['Name'] == label, 'Drb/90']), kind='rank'),
            color='lightblue')
    plt.annotate(
        math.trunc(stats.percentileofscore(df['Drb/90'], (df2.loc[df2['Name'] == label, 'Drb/90']), kind='rank')[0]),
        ('Dribbles', stats.percentileofscore(df['Drb/90'], (df2.loc[df2['Name'] == label, 'Drb/90']), kind='rank')),
        ha='center',
        color='white',
        xytext=(
        'Dribbles', stats.percentileofscore(df['Drb/90'], (df2.loc[df2['Name'] == label, 'Drb/90']), kind='rank') + 1),
        fontsize=12)

    ax.set_yticks(range(0, 100, 10))

    plt.bar('Header Win %', stats.percentileofscore(df['Hdr %'], (df2.loc[df2['Name'] == label, 'Hdr %']), kind='rank'),
            color='lightblue')
    plt.annotate(
        math.trunc(stats.percentileofscore(df['Hdr %'], (df2.loc[df2['Name'] == label, 'Hdr %']), kind='rank')[0]),
        ('Header Win %', stats.percentileofscore(df['Hdr %'], (df2.loc[df2['Name'] == label, 'Hdr %']), kind='rank')),
        ha='center',
        color='white',
        xytext=(
            'Header Win %',
            stats.percentileofscore(df['Hdr %'], (df2.loc[df2['Name'] == label, 'Hdr %']), kind='rank') + 1),
        fontsize=12)

    plt.bar('Pass Completion %',
            stats.percentileofscore(df['Pas %'], (df2.loc[df2['Name'] == label, 'Pas %']), kind='rank'),
            color='lightblue')
    plt.annotate(
        math.trunc(stats.percentileofscore(df['Pas %'], (df2.loc[df2['Name'] == label, 'Pas %']), kind='rank')[0]),
        ('Pass Completion %',
         stats.percentileofscore(df['Pas %'], (df2.loc[df2['Name'] == label, 'Pas %']), kind='rank')),
        ha='center',
        color='white',
        xytext=(
            'Pass Completion %',
            stats.percentileofscore(df['Pas %'], (df2.loc[df2['Name'] == label, 'Pas %']), kind='rank') + 1),
        fontsize=12)

    plt.xticks(rotation=90)

    plt.title(label, color='white')
    plt.tight_layout()
    plt.setp(ax2.spines.values(), color="white")
    plt.ylabel('Percentile', color='white')
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.xaxis.label.set_color('white')
    ax2.set_facecolor('#333333')
    fig2.set_facecolor('#333333')
    plt.show(block=True)


header = ["Transfer Value", "avg"]
df2.to_csv("output.csv", index=False, columns=header)
ax.set_facecolor('#333333')
fig.set_facecolor('#333333')

plt.show(block=True)

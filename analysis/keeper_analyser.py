import numpy as np

def keeper(df, valMax):

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
import pandas as pd
from utils.value_conversion import value_to_float

def preprocess(fileLocation):
    df = pd.read_csv(fileLocation)
    df.fillna(0, inplace=True)
    # This code excludes players that are "Not for Sale"
    mask = df['Transfer Value'] == 'Not for Sale'
    df = df[~mask]
    df['Transfer Value'] = df['Transfer Value'].apply(value_to_float)
    df = df.sort_values(by=['Transfer Value'], ascending=False)
    df_obj = df.select_dtypes('object')
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip('%'))
    df_obj = df.select_dtypes('object')
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip('km'))

    numeric_columns = [
        'Pts/Gm', 'xGP/90', 'Hdrs L/90', 'Sv %', 'xSv %', 'Aer A/90', 'OP-Crs C/90', 'OP-Crs A/90',
        'Conv %', 'Mins/Gl', 'xA/90', 'Saves/90', 'xG/90', 'Gls/90', 'NP-xG/90', 'xG/shot', 'Shot %',
        'Hdrs W/90', 'K Hdrs/90', 'Cr C/A', 'Crs A/90', 'Cr C/90', 'OP-Cr %', 'Dist/90', 'Drb/90',
        'Ps A/90', 'Ps C/90', 'Sprints/90', 'Poss Lost/90', 'Pr passes/90', 'Shots Outside Box/90',
        'Shot/90', 'ShT/90', 'Blk/90', 'Clr/90', 'Int/90', 'Poss Won/90', 'OP-KP/90', 'Pres C/90',
        'Shts Blckd/90', 'Hdr %', 'Tck/90', 'Tck R', 'Pas %', 'Pres A/90', 'K Tck/90', 'Ch C/90',
        'K Ps/90', 'Asts/90']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
import io

import numpy as np
import pandas as pd

from utils.preprocessing import preprocess

CSV = (
    "Name,League,Transfer Value,Pts/Gm,Dist/90,Pas %,Sv %,xSv %,Asts/90\n"
    "Mark,Prem,£1.5M,2.1,10.5km,85%,70%,68%,0.4\n"
    "Kim,Prem,Not for Sale,1.9,9.2km,90%,72%,70%,0.2\n"
    "Frank,Liga,£750K,1.4,11.0km,77%,60%,0%,0.1\n"
)


def _load(**kwargs):
    return preprocess(io.StringIO(CSV), **kwargs)


def test_names_are_not_truncated():
    """str.strip('km') took a character set, turning 'Mark' into 'Mar'."""
    assert set(_load()['Name']) == {'Mark', 'Frank'}


def test_not_for_sale_players_are_dropped():
    assert 'Kim' not in set(_load()['Name'])


def test_units_are_stripped_and_columns_are_float64():
    df = _load()
    assert df.loc[df['Name'] == 'Mark', 'Dist/90'].iloc[0] == 10.5
    assert df.loc[df['Name'] == 'Mark', 'Pas %'].iloc[0] == 85.0
    for col in ('Transfer Value', 'Dist/90', 'Pas %'):
        assert df[col].dtype == np.dtype('float64')


def test_derived_save_ratio_survives_zero_expected_saves():
    df = _load()
    assert df.loc[df['Name'] == 'Mark', 'SvRatio'].iloc[0] == 70 / 68
    assert np.isnan(df.loc[df['Name'] == 'Frank', 'SvRatio'].iloc[0])


def test_league_scaling_applies_to_stat_columns():
    df = _load(league_scaling={'Prem': 1.0, 'Liga': 0.5})
    assert df.loc[df['Name'] == 'Frank', 'Asts/90'].iloc[0] == 0.05
    assert df.loc[df['Name'] == 'Mark', 'Asts/90'].iloc[0] == 0.4


def test_transfer_value_is_not_scaled_by_league():
    df = _load(league_scaling={'Liga': 0.5})
    assert df.loc[df['Name'] == 'Frank', 'Transfer Value'].iloc[0] == 750_000

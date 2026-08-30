"""Load a Football Manager CSV export into a clean, numeric DataFrame."""
import numpy as np
import pandas as pd

from config import (DERIVED_COLS, LEAGUE_COL, NOT_FOR_SALE, NUMERIC_COLUMNS,
                    VALUE_COL)
from utils.value_conversion import value_to_float


def preprocess(file_location, league_scaling=None):
    """Read `file_location` and return an analysis-ready DataFrame.

    league_scaling: optional {league name: multiplier} used to put stats from
    different divisions on a comparable footing. Applied to every numeric stat
    column before any average or correlation is taken, so baselines and player
    scores are scaled consistently. Leagues absent from the mapping scale by 1.0.
    """
    df = pd.read_csv(file_location)

    if VALUE_COL not in df.columns:
        raise KeyError(f"{VALUE_COL!r} column missing from {file_location}")

    # Drop players who cannot be signed at any price.
    df = df.loc[df[VALUE_COL] != NOT_FOR_SALE].copy()

    df[VALUE_COL] = df[VALUE_COL].map(value_to_float)

    # Coerce only the columns we know are numeric. The previous approach ran
    # .str.strip('km') across every object column, which silently truncated
    # names ending in 'k' or 'm' ("Mark" -> "Mar") because strip() takes a
    # character set, not a suffix.
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    if league_scaling:
        df = apply_league_scaling(df, league_scaling)

    df = _add_derived_columns(df)
    return df.sort_values(by=VALUE_COL, ascending=False)


def to_numeric(series):
    """Strip FM's unit suffixes and coerce to float; unparseable becomes NaN.

    NaN is left in place rather than filled with 0 -- the scorer skips missing
    values when averaging, and a 0 would drag a player's baseline down as if
    they had genuinely recorded nothing.
    """
    if series.dtype.kind in 'if':
        return series
    cleaned = (series.astype('string')
                     .str.replace(',', '', regex=False)
                     .str.replace('%', '', regex=False)
                     .str.replace(r'\s*km\s*$', '', regex=True, case=False)
                     .str.strip())
    # astype('string') yields pandas' nullable dtypes; force plain float64
    # so pd.NA becomes np.nan and numpy/corr behave predictably.
    return pd.to_numeric(cleaned, errors='coerce').astype('float64')


def _add_derived_columns(df):
    """Add stats computed from other stats rather than read from the export."""
    if {'Sv %', 'xSv %'}.issubset(df.columns):
        expected = df['xSv %'].replace(0, np.nan)  # avoid divide-by-zero
        df['SvRatio'] = df['Sv %'] / expected
    return df


def apply_league_scaling(df, league_scaling):
    """Multiply numeric stat columns by a per-league difficulty factor."""
    if LEAGUE_COL not in df.columns:
        raise KeyError(
            f"league_scaling given but {LEAGUE_COL!r} column is missing")

    factors = df[LEAGUE_COL].map(league_scaling).fillna(1.0)
    scalable = [c for c in NUMERIC_COLUMNS
                if c in df.columns and c not in DERIVED_COLS]
    df = df.copy()
    df[scalable] = df[scalable].mul(factors, axis=0)
    return df

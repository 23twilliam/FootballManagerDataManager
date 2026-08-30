"""Chart maths, with no rendering attached.

Everything here is a pure function over arrays and DataFrames, so the matplotlib
plot and the Streamlit app draw the same curve from the same numbers rather than
each growing its own version.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from config import LOWER_IS_BETTER

# How many players to draw. Plotting every row makes a dense export unreadable
# and slow, so we show sqrt(n) * 25 -- growing with the dataset but sub-linearly
# -- capped where the scatter stops being legible.
POINTS_PER_ROOT_PLAYER = 25
MAX_POINTS_PLOTTED = 1500

# Transfer values span five orders of magnitude and pile up at zero, so a linear
# axis crushes almost every player into the left edge. symlog keeps a linear
# stretch near zero -- free transfers are real and must stay visible -- and goes
# logarithmic above it.
VALUE_LINTHRESH = 100_000

# Tracing what a given fee normally buys: a rolling median over players sorted by
# price. Not fixed bands -- a band's point has to sit at the band's median price,
# so a banded curve can never reach the cheapest or dearest players.
CURVE_POINTS = 60      # how many places along the price range to evaluate
CURVE_WINDOW_FRAC = 8  # window holds 1/8 of the players
MIN_WINDOW = 20        # ...but never fewer than this many
MIN_FOR_CURVE = 60     # below this, a median curve is noise

# Free transfers are a different population -- released players and expiring
# contracts -- so their median ability says nothing about what a fee buys. The
# curve is fitted over priced players only and its gradient across this reference
# window extended down to zero. The window shrinks if the data stops short.
FREE_REF_LOW = 50_000
FREE_REF_HIGH = 500_000
MIN_FOR_EXTRAPOLATION = 15


def points_to_plot(total: int) -> int:
    """Sub-linear cap on how many players to draw."""
    return min(int(math.sqrt(total) * POINTS_PER_ROOT_PLAYER),
               MAX_POINTS_PLOTTED, total)


def money(value, _pos=None) -> str:
    """120000000 -> '120M'. Axis ticks, so brevity beats precision."""
    value = float(value)
    for scale, suffix in ((1e9, 'B'), (1e6, 'M'), (1e3, 'K')):
        if abs(value) >= scale:
            trimmed = value / scale
            places = 0 if abs(trimmed) >= 10 else 1
            return f'{trimmed:.{places}f}{suffix}'
    return f'{value:.0f}'


def symlog(value, linthresh: float = VALUE_LINTHRESH):
    """The axis's own coordinate: linear to `linthresh`, decades above it.

    Extrapolating toward a price of zero has to happen in this space, not in raw
    pounds or in log pounds -- log(0) is undefined, and a straight line in pounds
    is a curve on screen.
    """
    value = np.asarray(value, dtype=float)
    small = np.abs(value) <= linthresh
    scaled = np.divide(np.abs(value), linthresh,
                       out=np.ones_like(value), where=~small)
    return np.where(small, value / linthresh,
                    np.sign(value) * (1 + np.log10(scaled)))


def extrapolate_to_free(centres, medians):
    """Value at a price of zero, read off the trend rather than measured.

    Returns None when the reference window holds too little of the curve to give
    a gradient worth trusting.
    """
    centres, medians = np.asarray(centres), np.asarray(medians)
    low, high = FREE_REF_LOW, min(FREE_REF_HIGH, centres.max())
    if high <= low:
        low, high = centres.min(), centres.max()
    window = (centres >= low) & (centres <= high)
    if window.sum() < 2:
        return None

    positions = symlog(centres[window])
    if np.ptp(positions) == 0:
        return None
    gradient, intercept = np.polyfit(positions, medians[window], 1)
    return float(intercept + gradient * symlog(0.0))


def value_curve(x, y):
    """Median score at each price level, or None if the prices cannot support it.

    Returns (prices, medians). The reference the chart needs is not a flat
    average -- only the strongest half of a shortlist is ever drawn, so a
    whole-population line sits at or below the bottom of the cloud every time.
    What matters is what a given fee normally buys: above the curve is good
    value, below it is not.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < MIN_FOR_CURVE or np.ptp(x) == 0:
        return None

    # Fit on priced players only; the free cluster is handled by extrapolation.
    priced = x > 0
    has_free = bool((~priced).any())
    if priced.sum() >= MIN_FOR_CURVE and np.ptp(x[priced]) > 0:
        x, y = x[priced], y[priced]
    else:
        has_free = False

    order = np.argsort(x, kind='stable')
    xs, ys = x[order], y[order]
    count = len(xs)
    window = max(MIN_WINDOW, count // CURVE_WINDOW_FRAC)
    half = window // 2

    ranks = np.unique(np.linspace(0, count - 1, CURVE_POINTS).astype(int))
    centres, medians = [], []
    for rank in ranks:
        # Shrink the window symmetrically at the edges rather than sliding it
        # inward. Sliding leaves every point near an end sharing one window, so
        # the curve flatlines over the cheapest and dearest players -- exactly
        # where it needs to keep moving.
        reach = max(min(half, rank, count - 1 - rank), MIN_WINDOW // 2)
        low, high = max(0, rank - reach), min(count, rank + reach + 1)
        # Widen to cover every player sharing this price. Free transfers all sit
        # at exactly zero and the sort orders those ties arbitrarily, so a window
        # clipping through the cluster would take an arbitrary handful of them.
        low = min(low, int(np.searchsorted(xs, xs[rank], 'left')))
        high = max(high, int(np.searchsorted(xs, xs[rank], 'right')))
        centres.append(xs[rank])
        medians.append(np.median(ys[low:high]))

    centres, medians = np.asarray(centres), np.asarray(medians)
    keep = np.concatenate(([True], np.diff(centres) > 0))
    centres, medians = centres[keep], medians[keep]
    if len(centres) < 3:
        return None

    if has_free and len(centres) >= MIN_FOR_EXTRAPOLATION:
        free_value = extrapolate_to_free(centres, medians)
        if free_value is not None:
            centres = np.concatenate(([0.0], centres))
            medians = np.concatenate(([free_value], medians))
    return centres, medians


def percentile_of(population: pd.Series, value):
    """Percentile rank of `value` within `population`, or None if unavailable."""
    clean = population.dropna()
    if clean.empty or value is None or value != value:  # NaN check
        return None
    return float(scipy_stats.percentileofscore(clean, value, kind='rank'))


def percentile_table(df: pd.DataFrame, player: pd.Series, stats) -> pd.DataFrame:
    """One row per stat: the player's value and how good it is, 0-100.

    Percentiles for stats in LOWER_IS_BETTER are flipped, so every bar reads
    the same way round: high is good. Without it a player who almost never
    gives the ball away shows a red bar at the 10th percentile, which says the
    opposite of what is true. `raw_percentile` keeps the unflipped rank and
    `inverted` marks which rows were turned around.

    Passing the player's row rather than looking them up by name avoids picking
    the wrong player when two share a name, which FM exports routinely contain.
    """
    rows = []
    for stat in stats:
        if stat not in df.columns or stat not in player.index:
            continue
        raw = percentile_of(df[stat], player[stat])
        inverted = stat in LOWER_IS_BETTER
        rows.append({'stat': stat,
                     'value': player[stat],
                     'raw_percentile': raw,
                     'percentile': None if raw is None else
                                   (100.0 - raw if inverted else raw),
                     'inverted': inverted})
    return pd.DataFrame(
        rows, columns=['stat', 'value', 'raw_percentile', 'percentile',
                       'inverted'])

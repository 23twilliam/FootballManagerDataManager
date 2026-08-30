"""Scatter plot of shortlisted players: transfer value against score.

Hovering shows a name; clicking opens that player's percentile breakdown.
The plot knows nothing about positions or scoring methods -- it takes a frame
carrying a score column and a list of stats to chart, so the CA model and the
older correlation scorer can both drive it.
"""
import math

import numpy as np
import mplcursors

from config import NAME_COL, SCORE_COL, VALUE_COL
from visualisation.percentile_chart import show_percentile_chart
from matplotlib.ticker import FuncFormatter

from visualisation.style import (ANNOTATION_EDGE, ANNOTATION_FACE, BACKGROUND,
                                 FOREGROUND, SCORE_CMAP, new_dark_figure, plt)

# How many players to draw. Plotting every row makes a dense CSV unreadable and
# slow to render, so we show sqrt(n) * 25 -- growing with the dataset but
# sub-linearly -- capped at a point where the scatter is still legible.
POINTS_PER_ROOT_PLAYER = 25
MAX_POINTS_PLOTTED = 1500

# Marker size shrinks as the plot fills up, down to a readable floor.
MAX_MARKER_SIZE = 50
MIN_MARKER_SIZE = 10
MARKER_SHRINK_PER_PLAYER = 1 / 100

Y_PADDING = 0.05  # headroom above and below the plotted score range

# Transfer values span five orders of magnitude and pile up at zero, so a
# linear axis crushes almost every player into the left edge. symlog keeps a
# linear stretch near zero -- free transfers are real and must stay visible --
# and goes logarithmic above it.
VALUE_LINTHRESH = 100_000


def plot_shortlist(scored, df, title, chart_stats, reference=None,
                   score_label='Score', reference_label='median'):
    """Plot the top of `scored`; clicking a point opens its breakdown.

    scored:      DataFrame carrying SCORE_COL, NAME_COL and VALUE_COL.
    df:          the full population, the comparison set for percentiles.
    chart_stats: which stats the click-through breakdown shows.
    reference:   optional y value to mark with a dashed line.
    """
    if scored.empty:
        raise ValueError("Nothing to plot -- no players matched the value ceiling")

    scored = scored.sort_values(SCORE_COL, ascending=False)
    shown = scored.head(_points_to_plot(len(scored)))
    print(f"Plotting {len(shown)} of {len(scored)} affordable players.")

    fig, ax = new_dark_figure()
    x = shown[VALUE_COL].to_numpy()
    y = shown[SCORE_COL].to_numpy()
    names = shown[NAME_COL].to_numpy()

    colours = plt.get_cmap(SCORE_CMAP)(plt.Normalize(y.min(), y.max())(y))
    scatter = ax.scatter(x, y, s=_marker_size(len(shown)), c=colours, marker='.')

    _scale_value_axis(ax, x)
    ax.set_xlabel('Transfer Value')
    ax.set_ylabel(score_label)
    ax.set_title(title, color=FOREGROUND)
    _set_score_limits(ax, y, reference)
    if not _draw_value_curve(ax, x, y):
        # Not enough price variation to fit a curve; fall back to a flat line
        # through the middle of what is actually on screen.
        _draw_reference_line(ax, np.median(y) if reference is None else reference,
                             reference_label)

    _attach_hover(scatter, names)
    _attach_click(fig, scatter, shown, df, chart_stats)

    fig.tight_layout()
    plt.show(block=True)


def _scale_value_axis(ax, x):
    """Put transfer value on a symlog axis with readable money labels."""
    if np.nanmax(x) > VALUE_LINTHRESH * 10:
        ax.set_xscale('symlog', linthresh=VALUE_LINTHRESH)
    ax.xaxis.set_major_formatter(FuncFormatter(_money))


def _money(value, _pos=None):
    """120000000 -> '120M'. Axis ticks, so brevity beats precision."""
    value = float(value)
    for scale, suffix in ((1e9, 'B'), (1e6, 'M'), (1e3, 'K')):
        if abs(value) >= scale:
            trimmed = value / scale
            places = 0 if abs(trimmed) >= 10 else 1
            return f'{trimmed:.{places}f}{suffix}'
    return f'{value:.0f}'


def _points_to_plot(total):
    """Sub-linear cap on how many players to draw."""
    return min(int(math.sqrt(total) * POINTS_PER_ROOT_PLAYER), MAX_POINTS_PLOTTED, total)


def _marker_size(count):
    return max(MIN_MARKER_SIZE, MAX_MARKER_SIZE - count * MARKER_SHRINK_PER_PLAYER)


def _set_score_limits(ax, y, reference=None):
    """Frame the actual score range.

    The old code floored the axis at 0, which hid every below-average player and
    inverted the axis entirely when all scores were negative.
    """
    values = list(y) + ([reference] if reference is not None
                        and np.isfinite(reference) else [])
    low, high = float(np.min(values)), float(np.max(values))
    span = high - low or abs(high) or 1.0
    ax.set_ylim(low - span * Y_PADDING, high + span * Y_PADDING)


# Tracing what a given fee normally buys. A rolling median over players sorted
# by price, rather than fixed bands: a band's point has to sit at the band's
# median price, so a banded curve could never reach the cheapest or dearest
# players and always stopped short at both ends.
CURVE_POINTS = 60      # how many places along the price range to evaluate
CURVE_WINDOW_FRAC = 8  # window holds 1/8 of the players
MIN_WINDOW = 20        # ...but never fewer than this many
MIN_FOR_CURVE = 60     # below this, a median curve is noise

# Free transfers are a different population -- released players and expiring
# contracts -- so their median ability says nothing about what a fee buys.
# Rather than letting them kink the left-hand end, the curve is fitted over
# priced players only and its gradient across this reference window is
# extended down to zero. The window shrinks to fit if the data does not reach
# that far.
FREE_REF_LOW = 50_000
FREE_REF_HIGH = 500_000
MIN_FOR_EXTRAPOLATION = 15


def _symlog(value, linthresh=VALUE_LINTHRESH):
    """The axis's own coordinate: linear to `linthresh`, decades above it.

    Extrapolating toward a price of zero has to happen in this space, not in
    raw pounds or in log pounds -- log(0) is undefined, and a straight line in
    pounds is a curve on screen.
    """
    value = np.asarray(value, dtype=float)
    small = np.abs(value) <= linthresh
    scaled = np.divide(np.abs(value), linthresh,
                       out=np.ones_like(value), where=~small)
    return np.where(small, value / linthresh,
                    np.sign(value) * (1 + np.log10(scaled)))


def _extrapolate_to_free(centres, medians):
    """Value at a price of zero, read off the trend rather than measured.

    Returns None when the reference window holds too little of the curve to
    give a gradient worth trusting.
    """
    low, high = FREE_REF_LOW, min(FREE_REF_HIGH, centres.max())
    if high <= low:
        low, high = centres.min(), centres.max()
    window = (centres >= low) & (centres <= high)
    if window.sum() < 2:
        return None

    positions = _symlog(centres[window])
    if np.ptp(positions) == 0:
        return None
    gradient, intercept = np.polyfit(positions, medians[window], 1)
    return float(intercept + gradient * _symlog(0.0))


def _draw_value_curve(ax, x, y):
    """Trace the median score at each price level. Returns True if drawn.

    A flat line at the population average is useless here: the plot only shows
    the strongest half of the shortlist, so a whole-population reference sits at
    or below the bottom of the cloud every time. What the chart is for is
    spotting players who beat their price, so the reference is what that price
    normally buys -- above the curve is good value, below it is not.

    A rolling median over price-sorted players, evaluated at real data points
    from the cheapest to the dearest, so the curve spans the whole axis
    rather than stopping short of both ends.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < MIN_FOR_CURVE or np.ptp(x) == 0:
        return False

    # Fit on priced players only; the free cluster is handled separately.
    priced = x > 0
    has_free = (~priced).any()
    if priced.sum() >= MIN_FOR_CURVE and np.ptp(x[priced]) > 0:
        x, y = x[priced], y[priced]
    else:
        has_free = False

    order = np.argsort(x, kind='stable')
    xs, ys = x[order], y[order]
    count = len(xs)
    window = max(MIN_WINDOW, count // CURVE_WINDOW_FRAC)

    # Evaluate at ranks spanning 0..count-1 so both extremes are included.
    ranks = np.unique(np.linspace(0, count - 1, CURVE_POINTS).astype(int))
    half = window // 2
    centres, medians = [], []
    for rank in ranks:
        # Shrink the window symmetrically at the edges rather than sliding it
        # inward. Sliding leaves every point near an end sharing one window, so
        # the curve flatlines over the cheapest and dearest players -- exactly
        # where it needs to keep moving. Narrower means noisier out there, which
        # is the honest trade.
        reach = max(min(half, rank, count - 1 - rank), MIN_WINDOW // 2)
        low, high = max(0, rank - reach), min(count, rank + reach + 1)
        # Widen to cover every player sharing this price. Free transfers all
        # sit at exactly zero and the sort orders those ties arbitrarily, so
        # a window clipping through the cluster would take an arbitrary
        # handful of them and spike the left-hand end of the curve.
        low = min(low, int(np.searchsorted(xs, xs[rank], 'left')))
        high = max(high, int(np.searchsorted(xs, xs[rank], 'right')))
        centres.append(xs[rank])
        medians.append(np.median(ys[low:high]))

    # Collapse duplicate prices (a big cluster of free players) to one point.
    centres, medians = np.asarray(centres), np.asarray(medians)
    keep = np.concatenate(([True], np.diff(centres) > 0))
    centres, medians = centres[keep], medians[keep]
    if len(centres) < 3:
        return False

    if has_free and len(centres) >= MIN_FOR_EXTRAPOLATION:
        free_value = _extrapolate_to_free(centres, medians)
        if free_value is not None:
            centres = np.concatenate(([0.0], centres))
            medians = np.concatenate(([free_value], medians))

    ax.plot(centres, medians, color=FOREGROUND, linestyle='--',
            linewidth=1.4, alpha=0.85, zorder=4,
            label='typical for this price')
    legend = ax.legend(loc='upper left', frameon=True, fontsize=8)
    legend.get_frame().set(facecolor=BACKGROUND, edgecolor=FOREGROUND,
                           alpha=0.75)
    for text in legend.get_texts():
        text.set_color(FOREGROUND)
    return True


def _draw_reference_line(ax, reference, label='median'):
    """Flat fallback when the prices cannot support a curve."""
    if reference is not None and np.isfinite(reference):
        ax.axhline(reference, color=FOREGROUND, linestyle='--',
                   linewidth=0.8, alpha=0.5)
        ax.annotate(label, (0.99, reference), xycoords=('axes fraction', 'data'),
                    ha='right', va='bottom', color=FOREGROUND, alpha=0.6, fontsize=8)


def _attach_hover(scatter, names):
    """Show a player's name on hover."""
    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect('add')
    def _(selection):
        selection.annotation.set_text(names[selection.index])
        selection.annotation.get_bbox_patch().set(
            color=ANNOTATION_FACE, ec=ANNOTATION_EDGE, snap=True,
            boxstyle='round,pad=0.3')
        if selection.annotation.arrow_patch is not None:
            selection.annotation.arrow_patch.set(visible=False)

    return cursor


def _attach_click(fig, scatter, shown, df, chart_stats):
    """Open a percentile breakdown for the clicked player."""
    cursor = mplcursors.cursor(scatter, hover=False)

    @cursor.connect('add')
    def _(selection):
        player = shown.iloc[selection.index]
        name = player[NAME_COL]
        plt.close(fig)
        try:
            show_percentile_chart(df, player, chart_stats, name)
        except (ValueError, KeyError) as exc:
            print(f"Could not chart {name}: {exc}")

    return cursor

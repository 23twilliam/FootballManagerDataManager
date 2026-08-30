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
from visualisation.style import (ANNOTATION_EDGE, ANNOTATION_FACE, FOREGROUND,
                                 SCORE_CMAP, new_dark_figure, plt)

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


def plot_shortlist(scored, df, title, chart_stats, reference=None,
                   score_label='Score'):
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

    ax.set_xlabel('Transfer Value')
    ax.set_ylabel(score_label)
    ax.set_title(title, color=FOREGROUND)
    _set_score_limits(ax, y, reference)
    _draw_reference_line(ax, reference)

    _attach_hover(scatter, names)
    _attach_click(fig, scatter, shown, df, chart_stats)

    fig.tight_layout()
    plt.show(block=True)


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


def _draw_reference_line(ax, reference):
    """Mark a reference score, e.g. the league or dataset average."""
    if reference is not None and np.isfinite(reference):
        ax.axhline(reference, color=FOREGROUND, linestyle='--',
                   linewidth=0.8, alpha=0.5)
        ax.annotate('average', (0.99, reference), xycoords=('axes fraction', 'data'),
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

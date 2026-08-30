"""Shared dark-theme styling for every chart.

Previously 16 identical lines of axis styling were copy-pasted into each of the
ten on-click modules; they live here once instead.
"""
import os

import matplotlib

# Interactive backend, chosen before pyplot is imported. Respect an explicit
# MPLBACKEND so the charts can be exercised headlessly (e.g. under CI).
if not os.environ.get('MPLBACKEND'):
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt  # noqa: E402

BACKGROUND = '#333333'
FOREGROUND = 'white'
ANNOTATION_FACE = '#FFFFFF'
ANNOTATION_EDGE = '#999999'
BAR_COLOUR = 'lightblue'
SCORE_CMAP = 'RdYlGn'


def new_dark_figure():
    """Create a figure and axes already styled for the dark theme."""
    fig, ax = plt.subplots()
    style_axes(fig, ax)
    return fig, ax


def style_axes(fig, ax):
    """Apply the dark theme to an existing figure/axes pair."""
    fig.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    plt.setp(ax.spines.values(), color=FOREGROUND)
    ax.tick_params(axis='x', colors=FOREGROUND)
    ax.tick_params(axis='y', colors=FOREGROUND)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.label.set_color(FOREGROUND)
    ax.yaxis.label.set_color(FOREGROUND)
    ax.title.set_color(FOREGROUND)
    return ax

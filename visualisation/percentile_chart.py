"""Per-player percentile breakdown -- one module replacing ten near-identical ones.

The caller chooses which stats to chart, so the breakdown reflects whatever
actually drove the score -- the model's top-weighted features for the CA model.
Previously each on-click module kept its own hand-maintained list, and several
had drifted out of sync with the analyser they were paired with.
"""
import math

from scipy import stats as scipy_stats

from config import stats_dict
from visualisation.style import BAR_COLOUR, FOREGROUND, new_dark_figure, plt


def show_percentile_chart(df, player, stats, name):
    """Plot `player`'s percentile in each of `stats` against `df`.

    player: a Series -- one row of the scored frame. Passing the row itself
    rather than looking it up by name avoids picking the wrong player when two
    share a name, which FM exports routinely contain.
    stats:  which columns to chart. The model path passes its top-weighted
    features, so the breakdown reflects what actually drove the score.
    """
    stats = list(stats)
    charted = [s for s in stats if s in df.columns and s in player.index]
    if not charted:
        raise ValueError(f"None of {stats} are present to chart")

    fig, ax = new_dark_figure()

    for stat in charted:
        percentile = _percentile(df[stat], player[stat])
        label = stats_dict.get(stat, stat)
        if percentile is None:
            # Missing value: leave a visible gap rather than implying zero.
            ax.bar(label, 0, color='none')
            continue
        ax.bar(label, percentile, color=BAR_COLOUR)
        ax.annotate(math.trunc(percentile), (label, percentile),
                    ha='center', va='bottom', color=FOREGROUND, fontsize=12)

    ax.set_yticks(range(0, 101, 10))
    ax.set_ylim(0, 105)
    ax.set_ylabel('Percentile', color=FOREGROUND)
    ax.set_title(name, color=FOREGROUND)
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    plt.show(block=True)


def _percentile(population, value):
    """Percentile rank of `value` within `population`, or None if unavailable."""
    clean = population.dropna()
    if clean.empty or value is None or value != value:  # NaN check
        return None
    return scipy_stats.percentileofscore(clean, value, kind='rank')

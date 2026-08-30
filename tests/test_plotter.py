import numpy as np
import pandas as pd
import pytest

from visualisation import plotter, shortlist
from visualisation.style import plt


@pytest.fixture
def axes():
    figure, ax = plt.subplots()
    yield ax
    plt.close(figure)


@pytest.mark.parametrize('value, expected', [
    (0, '0'), (750, '750'), (40_000, '40K'), (100_000, '100K'),
    (2_550_000, '2.5M'), (12_500_000, '12M'), (260_000_000, '260M'),
])
def test_money_formatter(value, expected):
    assert shortlist.money(value) == expected


def _cloud(n=600, seed=0):
    """Prices spanning five orders of magnitude, ability rising with price."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.zeros(n // 6), rng.lognormal(13, 2, n - n // 6)])
    y = 100 + np.log1p(x) * 2.5 + rng.normal(0, 6, len(x))
    return x, y


def test_value_curve_is_drawn_and_sits_inside_the_cloud(axes):
    """A flat population average lands at the bottom edge: the plot only shows
    the strongest half of the shortlist."""
    x, y = _cloud()
    assert plotter._draw_value_curve(axes, shortlist.value_curve(x, y)) is True
    curve = axes.get_lines()[0].get_ydata()
    assert y.min() < min(curve) and max(curve) < y.max()


def test_value_curve_splits_the_players_roughly_in_half(axes):
    x, y = _cloud()
    plotter._draw_value_curve(axes, shortlist.value_curve(x, y))
    line = axes.get_lines()[0]
    above = (y > np.interp(x, line.get_xdata(), line.get_ydata())).mean()
    assert 0.4 < above < 0.6


def test_value_curve_rises_with_price(axes):
    """The reference is what a fee normally buys, so it must track price."""
    x, y = _cloud()
    plotter._draw_value_curve(axes, shortlist.value_curve(x, y))
    curve = axes.get_lines()[0].get_ydata()
    assert curve[-1] > curve[0]


def test_no_curve_when_every_player_costs_the_same(axes):
    x = np.zeros(400)
    y = np.random.default_rng(0).normal(120, 10, 400)
    assert plotter._draw_value_curve(axes, shortlist.value_curve(x, y)) is False


def test_no_curve_with_too_few_players(axes):
    x, y = _cloud(n=20)
    assert plotter._draw_value_curve(axes, shortlist.value_curve(x, y)) is False


def test_flat_reference_line_is_the_fallback(axes):
    plotter._draw_reference_line(axes, 120.0, 'median of those shown')
    assert any(np.allclose(line.get_ydata(), 120.0)
               for line in axes.get_lines() + list(axes.get_children())
               if hasattr(line, 'get_ydata'))


def test_value_axis_goes_symlog_when_prices_span_orders_of_magnitude(axes):
    plotter._scale_value_axis(axes, np.array([0.0, 5e3, 1e6, 2.6e8]))
    assert axes.get_xscale() == 'symlog'


def test_value_axis_stays_linear_for_a_narrow_price_range(axes):
    plotter._scale_value_axis(axes, np.array([0.0, 1e3, 5e3]))
    assert axes.get_xscale() == 'linear'


def test_score_limits_frame_the_data_not_zero(axes):
    """The old code floored the axis at 0, hiding every below-average player."""
    y = np.array([102.0, 140.0, 200.0])
    plotter._set_score_limits(axes, y)
    low, high = axes.get_ylim()
    assert low > 50 and high > 200


def test_curve_reaches_the_cheapest_and_dearest_players(axes):
    """Banded medians could only span the 4th-96th percentile, so the curve
    stopped short at both ends -- no line below ~50K or above ~50M."""
    x, y = _cloud()
    plotter._draw_value_curve(axes, shortlist.value_curve(x, y))
    curve_x = axes.get_lines()[0].get_xdata()
    assert curve_x.min() <= x.min()
    assert curve_x.max() >= x.max()


def test_curve_keeps_moving_at_the_ends(axes):
    """Sliding the window inward left every edge point sharing one window, which
    flatlined the curve exactly where it still needed to rise."""
    x, y = _cloud()
    plotter._draw_value_curve(axes, shortlist.value_curve(x, y))
    curve = axes.get_lines()[0].get_ydata()
    assert curve[0] != curve[3], 'left end is flat'
    assert curve[-1] != curve[-4], 'right end is flat'


def test_tied_prices_are_averaged_together_not_sampled(axes):
    """Free transfers all sit at exactly zero and sort order among ties is
    arbitrary, so a clipped window would spike the left-hand end."""
    rng = np.random.default_rng(1)
    free_y = rng.normal(110, 25, 200)          # wide spread, all priced at zero
    paid_x = rng.lognormal(14, 1.5, 400)
    x = np.concatenate([np.zeros(200), paid_x])
    y = np.concatenate([free_y, 100 + np.log1p(paid_x) * 2])

    firsts = []
    for seed in range(5):                       # reshuffle the ties each time
        shuffle = np.random.default_rng(seed).permutation(len(x))
        axes.clear()
        plotter._draw_value_curve(axes, shortlist.value_curve(x[shuffle], y[shuffle]))
        firsts.append(axes.get_lines()[0].get_ydata()[0])
    assert np.ptp(firsts) < 1.0, (
        f'first point swings by {np.ptp(firsts):.1f} depending on tie order')


def test_curve_tracks_a_rising_price_relationship_end_to_end(axes):
    x, y = _cloud()
    plotter._draw_value_curve(axes, shortlist.value_curve(x, y))
    curve = axes.get_lines()[0].get_ydata()
    # Cheapest third should sit clearly below the dearest third.
    third = len(curve) // 3
    assert curve[:third].mean() < curve[-third:].mean() - 5


@pytest.mark.parametrize('value, expected', [
    (0, 0.0), (50_000, 0.5), (100_000, 1.0), (1_000_000, 2.0), (100_000_000, 4.0),
])
def test_symlog_matches_the_axis_coordinate(value, expected):
    """Extrapolating toward zero has to happen in the axis's own space: log(0)
    is undefined and a straight line in pounds is a curve on screen."""
    assert float(shortlist.symlog(value)) == pytest.approx(expected, abs=1e-6)


def _cloud_with_odd_free_players(free_level, n=800, seed=2):
    """Priced players on a clean trend, plus free players at a chosen level."""
    rng = np.random.default_rng(seed)
    paid_x = rng.lognormal(13.5, 1.8, n)
    paid_y = 100 + shortlist.symlog(paid_x) * 12 + rng.normal(0, 4, n)
    free_x = np.zeros(40)
    free_y = np.full(40, free_level, dtype=float) + rng.normal(0, 2, 40)
    return (np.concatenate([free_x, paid_x]), np.concatenate([free_y, paid_y]))


def test_zero_point_comes_from_the_trend_not_the_free_players(axes):
    """Free transfers are released players and expiring contracts -- a different
    population, whose median says nothing about what a fee buys. The zero point
    must sit on the priced trend regardless of how odd they are."""
    x, y = _cloud_with_odd_free_players(175.0)
    plotter._draw_value_curve(axes, shortlist.value_curve(x, y))
    line = axes.get_lines()[0]
    assert line.get_xdata()[0] == 0
    at_zero = line.get_ydata()[0]

    # What the priced players alone imply at a price of zero.
    axes.clear()
    priced = x > 0
    plotter._draw_value_curve(axes, shortlist.value_curve(x[priced], y[priced]))
    paid = axes.get_lines()[0]
    expected = shortlist.extrapolate_to_free(paid.get_xdata(), paid.get_ydata())
    assert at_zero == pytest.approx(expected, abs=1.0)
    assert abs(at_zero - 175.0) > 20, 'zero point tracked the free players'


def test_zero_point_is_stable_however_odd_the_free_players_are(axes):
    values = []
    for level in (60.0, 100.0, 175.0):
        x, y = _cloud_with_odd_free_players(level)
        axes.clear()
        plotter._draw_value_curve(axes, shortlist.value_curve(x, y))
        values.append(axes.get_lines()[0].get_ydata()[0])
    assert np.ptp(values) < 2.0, f'zero point swings by {np.ptp(values):.1f}'


def test_zero_point_continues_the_gradient_downward(axes):
    """It should sit below the cheapest priced players, not spike above them."""
    x, y = _cloud_with_odd_free_players(175.0)
    plotter._draw_value_curve(axes, shortlist.value_curve(x, y))
    curve = axes.get_lines()[0].get_ydata()
    assert curve[0] < curve[1] < curve[2]


def test_no_zero_point_when_nobody_is_free(axes):
    x, y = _cloud()
    x = x[x > 0]
    y = y[-len(x):]
    plotter._draw_value_curve(axes, shortlist.value_curve(x, y))
    assert axes.get_lines()[0].get_xdata()[0] > 0


def test_reference_window_shrinks_when_prices_never_reach_it(axes):
    """--max-value 200000 leaves nothing in the 50K-500K window."""
    rng = np.random.default_rng(3)
    paid_x = rng.uniform(1_000, 200_000, 500)
    paid_y = 90 + shortlist.symlog(paid_x) * 8 + rng.normal(0, 3, 500)
    x = np.concatenate([np.zeros(40), paid_x])
    y = np.concatenate([np.full(40, 150.0), paid_y])
    assert plotter._draw_value_curve(axes, shortlist.value_curve(x, y)) is True
    curve = axes.get_lines()[0]
    assert curve.get_xdata()[0] == 0
    assert curve.get_ydata()[0] < 120, 'fell back to the free players median'


# --- negative stats --------------------------------------------------------

def _population(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'Poss Lost/90': rng.uniform(0, 30, n),
        'Pas %': rng.uniform(50, 95, n),
        'Hdrs L/90': rng.uniform(0, 8, n),
        'Asts/90': rng.uniform(0, 1, n),
    })


def test_lower_is_better_stats_are_flipped(axes):
    """A player who rarely loses the ball should read as strong, not as a red
    bar at the 10th percentile."""
    df = _population()
    tidy = df.copy()
    tidy.loc[0] = {'Poss Lost/90': 0.1, 'Pas %': 90.0,
                   'Hdrs L/90': 0.1, 'Asts/90': 0.5}
    table = shortlist.percentile_table(
        tidy, tidy.loc[0], ['Poss Lost/90', 'Pas %']).set_index('stat')

    losses = table.loc['Poss Lost/90']
    assert bool(losses.inverted)
    assert losses.raw_percentile < 10, 'fixture should be near the bottom'
    assert losses.percentile > 90, 'flipped percentile should read as a strength'
    assert losses.percentile == pytest.approx(100 - losses.raw_percentile)


def test_ordinary_stats_are_untouched(axes):
    df = _population()
    row = df.loc[df['Pas %'].idxmax()]
    table = shortlist.percentile_table(df, row, ['Pas %', 'Asts/90']).set_index('stat')
    for stat in ('Pas %', 'Asts/90'):
        assert not bool(table.loc[stat].inverted)
        assert table.loc[stat].percentile == table.loc[stat].raw_percentile


def test_the_worst_offender_scores_zero_not_a_hundred():
    df = _population()
    row = df.loc[df['Poss Lost/90'].idxmax()]
    table = shortlist.percentile_table(df, row, ['Poss Lost/90']).set_index('stat')
    assert table.loc['Poss Lost/90'].percentile == pytest.approx(0.0)


def test_direction_is_declared_not_inferred():
    """Possession lost correlates POSITIVELY with CA (+0.166 across positions)
    because better players have the ball more, so inferring the direction from
    the data would rank giving the ball away as a virtue."""
    from config import LOWER_IS_BETTER
    assert 'Poss Lost/90' in LOWER_IS_BETTER


def test_missing_values_stay_missing(axes):
    df = _population()
    row = df.loc[0].copy()
    row['Poss Lost/90'] = np.nan
    table = shortlist.percentile_table(df, row, ['Poss Lost/90']).set_index('stat')
    assert table.loc['Poss Lost/90'].percentile is None


# --- budget slider ---------------------------------------------------------

def test_budget_steps_are_log_spaced_not_linear():
    """A linear slider over 0-298M puts every fee people actually filter on --
    a few hundred thousand to a few million -- in the first centimetre."""
    import app
    steps = app.budget_steps(298_500_000)
    assert steps[0] == 0
    assert steps[-1] == 298_500_000
    # Under 5M should get as many notches as everything above it.
    below = sum(1 for s in steps if 0 < s <= 5_000_000)
    above = sum(1 for s in steps if s > 5_000_000)
    assert below >= above


@pytest.mark.parametrize('ceiling', [298_500_000, 5_000_000, 250_000, 9_000])
def test_budget_steps_are_sorted_unique_and_reach_the_ceiling(ceiling):
    import app
    steps = app.budget_steps(ceiling)
    assert steps == sorted(steps)
    assert len(steps) == len(set(steps))
    assert steps[-1] == float(ceiling)
    assert steps[0] == 0.0


def test_budget_steps_survive_an_empty_position():
    import app
    assert app.budget_steps(0) == [0.0]


def test_budget_steps_do_not_end_with_two_near_identical_notches():
    """Adjacent stops meaning the same budget are a nuisance to land on."""
    import app
    steps = app.budget_steps(1_020_000)   # ceiling just above the 1.0M notch
    assert steps[-1] == 1_020_000
    assert steps[-2] < steps[-1] / 1.15


def _frame(values, seed=0):
    """Minimal frame with the columns app.scatter reads."""
    rng = np.random.default_rng(seed)
    n = len(values)
    return pd.DataFrame({
        'Name': [f'P{i}' for i in range(n)],
        'Nation': ['England'] * n,
        'Division': ['Premier League'] * n,
        'Transfer Value': values,
        'CA': rng.integers(80, 180, n).astype(float),
        'CA_pred': rng.uniform(90, 170, n),
    })


@pytest.mark.parametrize('values, expected', [
    ([0.0] + list(np.geomspace(1_000, 1_000_000, 60)), 'log'),      # capped at 1M
    ([0.0] + list(np.geomspace(1_000, 300_000_000, 60)), 'log'),    # uncapped
    ([0.0] + list(np.linspace(20_000, 25_000, 60)), 'linear'),      # tight budget
])
def test_axis_type_follows_the_spread_of_fees(values, expected):
    """A shortlist capped at 1M still runs free-to-a-million. Keying the switch
    on the largest fee flipped that view to a linear axis, which buried most of
    the players at the left edge."""
    import app
    figure = app.scatter(_frame(np.asarray(values, dtype=float)), None, 'Predicted CA')
    assert figure.layout.xaxis.type == expected


def test_free_players_keep_their_tick_on_a_log_axis():
    """A log axis silently drops zero, hiding every free transfer."""
    import app
    values = np.array([0.0] * 5 + list(np.geomspace(1_000, 1_000_000, 60)))
    figure = app.scatter(_frame(values), None, 'Predicted CA')
    assert figure.layout.xaxis.type == 'log'
    assert app.FREE_LABEL in list(figure.layout.xaxis.ticktext)
    # and the zero-priced players are actually drawn
    plotted = figure.data[0].x
    assert (plotted > 0).all(), 'a zero would vanish on a log axis'
    assert len(plotted) == len(values)

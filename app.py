"""Streamlit front end: browse a position's shortlist and drill into a player.

    python -m streamlit run app.py

Everything here is presentation. Scoring, league handling and the chart maths
live in analysis/ and visualisation/shortlist.py, shared with the CLI.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import ca_model
from config import NAME_COL, VALUE_COL, stats_dict
from utils import html_import
from visualisation.shortlist import money, points_to_plot, value_curve

CHART_STATS = 12
BACKGROUND = '#1c1c1c'
PANEL = '#262626'
FOREGROUND = '#e8e8e8'
MUTED = '#8a8a8a'
SCORE_SCALE = 'RdYlGn'
FREE_LABEL = 'Free'
# Cheapest budget notch. Below this the filter is 'free only'.
FIRST_BUDGET_STEP = 10_000.0
# Use a log value axis once the fees on screen span this many times over.
LOG_AXIS_SPREAD = 100

SORTS = {
    'Best players (predicted CA)': ca_model.PRED_COL,
    'Most underrated for their level': ca_model.ADJ_RESIDUAL_COL,
    'Most underrated (uncorrected)': ca_model.RESIDUAL_COL,
    'Best relative to their league': 'pred_above_league',
}

# Sorts that need a real CA to mean anything. A scouted player has none, so
# these are hidden when the data came from an upload.
NEEDS_ACTUAL_CA = {'Most underrated for their level',
                   'Most underrated (uncorrected)'}

st.set_page_config(page_title='FM Data Hub', page_icon='⚽', layout='wide')


# --- data ------------------------------------------------------------------
# score_players retrains nothing but does run 5-fold out-of-fold predictions, so
# it costs about a second. Cached on position, which is the only thing it varies
# on; the filters below all work on the result.
@st.cache_data(show_spinner='Scoring players...')
def load_scored(position: str) -> pd.DataFrame:
    return ca_model.score_players(position, verbose=False)


@st.cache_data(show_spinner=False)
def load_meta(position: str) -> dict:
    return ca_model.load_model(position)[1]


@st.cache_data(show_spinner=False)
def chart_stats_for(position: str) -> list[str]:
    """The model's most influential features, as player-readable stats."""
    pipeline, meta = ca_model.load_model(position)
    weights = ca_model.top_weights(pipeline, meta['features'],
                                   n=len(meta['features']))
    keep = [s for s in weights.index
            if s != ca_model.LEAGUE_FEATURE
            and ' x league' not in s
            and not s.startswith(ca_model.LEAGUE_EFFECT_PREFIX)]
    return keep[:CHART_STATS]


def budget_steps(ceiling: float) -> list[float]:
    """Log-spaced budget stops, 1-2.5-5 per decade, ending at `ceiling`.

    A linear slider over 0 to 298M is useless: the fees people actually filter
    on -- a few hundred thousand to a few million -- live in the first
    centimetre of it. Fees are log-distributed, so the stops should be too.
    """
    if ceiling <= 0:
        return [0.0]

    # 1, 2.5, 5 through each decade -- the usual stops on a log scale, and all
    # round numbers a manager would actually think in.
    steps, decade = [0.0], FIRST_BUDGET_STEP
    while decade < ceiling:
        for mantissa in (1, 2.5, 5):
            value = decade * mantissa
            if value < ceiling:
                steps.append(value)
        decade *= 10
    steps.append(float(ceiling))

    # Drop a stop sitting almost on top of the ceiling; two adjacent notches
    # that mean the same budget are just a nuisance to land on.
    if len(steps) > 2 and steps[-1] < steps[-2] * 1.15:
        steps.pop(-2)
    return steps


@st.cache_data(show_spinner='Reading and scoring your export...')
def score_upload(raw: bytes, filename: str, position: str,
                 min_minutes: int) -> pd.DataFrame:
    """Parse an uploaded FM export and score it with a trained model.

    Cached on the file's bytes, so re-running the script for a filter change
    does not re-parse a 15MB export.
    """
    buffer = io.BytesIO(raw)
    buffer.name = filename  # so any parse error names the file, not a BytesIO
    table, _ = html_import.read_export(buffer)
    table = html_import.clean_table(table, filename, verbose=False)
    return ca_model.score_frame(table, position, min_minutes=min_minutes)


def guess_position(filename: str, positions: list[str]) -> int:
    """Index of the position whose name best matches the uploaded file."""
    stem = Path(filename).stem.casefold().replace(' ', '').replace('_', '')
    for index, name in enumerate(positions):
        if name.casefold() in stem or stem in name.casefold():
            return index
    return 0


def trained_positions() -> list[str]:
    return [name for name in ca_model.available_positions()
            if (ca_model.MODEL_DIR / f'{name}.joblib').exists()]


# --- chart -----------------------------------------------------------------
def value_axis(values):
    """Map fees onto a log axis, parking free transfers on their own tick.

    Plotly has no symlog, and a log axis silently drops zero -- which would
    hide every free transfer, up to 6.6%% of a plotted shortlist and exactly
    the bargains worth seeing. They are drawn just below the cheapest real fee
    with a tick that says so, rather than pretending they cost something.

    Returns (plotted positions, tick values, tick labels).
    """
    values = np.asarray(values, dtype=float)
    positive = values[values > 0]
    if positive.size == 0:
        return values, None, None

    floor = max(float(positive.min()), 1.0)
    free_at = floor / 4  # a clear gap, so the cluster reads as separate
    plotted = np.where(values > 0, np.maximum(values, floor), free_at)

    top = float(np.nanmax(values)) or floor
    # Stop at the last decade the data reaches; an extra tick stretches the axis.
    decades = range(int(np.floor(np.log10(floor))),
                    int(np.floor(np.log10(top))) + 1)
    ticks = [10.0 ** power for power in decades]
    return plotted, [free_at] + ticks, [FREE_LABEL] + [money(t) for t in ticks]



def scatter(shown: pd.DataFrame, curve, score_label: str) -> go.Figure:
    """Predicted ability against transfer value, with the price-expectation curve."""
    figure = go.Figure()
    x, tickvals, ticktext = value_axis(shown[VALUE_COL].to_numpy())

    if curve is not None:
        centres, medians = curve
        # Same mapping as the points, or the curve would sit off the scale.
        centres = value_axis(np.concatenate([centres, shown[VALUE_COL]]))[0][:len(centres)]
        figure.add_trace(go.Scatter(
            x=centres, y=medians, mode='lines', name='typical for this price',
            line=dict(color=FOREGROUND, width=2, dash='dash'),
            hovertemplate='typical at %{x:,.0f}: %{y:.1f}<extra></extra>'))

    custom = np.stack([
        shown[NAME_COL].to_numpy(),
        shown['Nation'].to_numpy() if 'Nation' in shown else np.full(len(shown), ''),
        shown['Division'].to_numpy() if 'Division' in shown else np.full(len(shown), ''),
        shown[VALUE_COL].to_numpy(),
        shown[ca_model.TARGET].to_numpy() if ca_model.TARGET in shown
        else np.full(len(shown), np.nan),
    ], axis=-1)

    figure.add_trace(go.Scatter(
        x=x, y=shown[ca_model.PRED_COL], mode='markers',
        name='players', customdata=custom,
        marker=dict(size=7, color=shown[ca_model.PRED_COL], colorscale=SCORE_SCALE,
                    line=dict(width=0), showscale=False),
        hovertemplate=('<b>%{customdata[0]}</b><br>'
                       '%{customdata[2]} (%{customdata[1]})<br>'
                       'predicted CA %{y:.1f} &nbsp; actual %{customdata[4]}<br>'
                       'value %{customdata[3]:,.0f}<extra></extra>')))

    # Switch on the spread of the fees, not their size. A shortlist capped at
    # 1M still runs from free to a million -- three decades -- and a linear
    # axis buries most of it at the left edge just as badly as an uncapped one.
    priced = shown.loc[shown[VALUE_COL] > 0, VALUE_COL]
    spread = (priced.max() / priced.min()) if len(priced) > 1 else 1.0
    logarithmic = spread > LOG_AXIS_SPREAD and tickvals is not None
    figure.update_xaxes(
        title='Transfer value', gridcolor=PANEL, zeroline=False,
        type='log' if logarithmic else 'linear',
        tickvals=tickvals if logarithmic else None,
        ticktext=ticktext if logarithmic else None)
    figure.update_yaxes(title=score_label, gridcolor=PANEL, zeroline=False)
    figure.update_layout(
        height=560, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font=dict(color=FOREGROUND),
        legend=dict(orientation='h', yanchor='bottom', y=1.01, x=0),
        hovermode='closest',
        # Must be set explicitly, and must not be 'pan'. Streamlit applies
        # dragmode='pan' by default, under which mousedown starts a pan
        # gesture and a click on a point never becomes a selection -- the
        # drill-down then silently never fires, while hover keeps working,
        # which makes it look like a data bug rather than a mode one.
        # Pan is still available from the modebar.
        dragmode='zoom',
        # 'event' alone fires a click without marking the point selected,
        # and the selection is what Streamlit reads back.
        clickmode='event+select')
    return figure


def percentile_bars(df: pd.DataFrame, player: pd.Series, stats) -> go.Figure:
    """Where this player ranks against the whole position, stat by stat."""
    from visualisation.shortlist import percentile_table

    table = percentile_table(df, player, stats).dropna(subset=['percentile'])
    # Flipped stats are labelled, so a strong bar on 'possession lost' cannot
    # be misread as losing the ball often.
    labels = [stats_dict.get(row.stat, row.stat) + (' (fewer)' if row.inverted else '')
              for row in table.itertuples()]
    figure = go.Figure(go.Bar(
        x=table['percentile'], y=labels, orientation='h',
        marker=dict(color=table['percentile'], colorscale=SCORE_SCALE,
                    cmin=0, cmax=100),
        text=[f'{p:.0f}' for p in table['percentile']],
        textposition='outside',
        hovertemplate=('%{y}<br>%{customdata[0]:.2f} per 90<br>'
                       'rated %{x:.0f}/100 &nbsp; (raw rank %{customdata[1]:.0f})'
                       '<extra></extra>'),
        customdata=np.stack([table['value'], table['raw_percentile']], axis=-1)))
    figure.update_xaxes(range=[0, 108],
                        title='Percentile within position (higher is better)',
                        gridcolor=PANEL)
    figure.update_yaxes(autorange='reversed', gridcolor=PANEL)
    figure.update_layout(
        height=max(260, 30 * len(table) + 80),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor=BACKGROUND, plot_bgcolor=BACKGROUND,
        font=dict(color=FOREGROUND), showlegend=False)
    return figure


def _selected_player(event, shown: pd.DataFrame):
    """The player behind the most recent click, or None.

    Clicks accumulate in the selection, so the last entry is the one just
    made -- reading the first would keep showing whatever was clicked
    earliest. The curve is trace 0, so only the scatter is a player.
    """
    points = (event.get('selection') or {}).get('points', []) if event else []
    for point in reversed(points):
        if point.get('curve_number') == 0:
            continue  # the price curve, not a player
        index = point.get('point_index', point.get('point_number'))
        if index is not None and 0 <= index < len(shown):
            return shown.iloc[index]
    return None


# --- app -------------------------------------------------------------------
def main():
    st.title('FM Data Hub')

    positions = trained_positions()
    if not positions:
        st.error(f'No trained models in `{ca_model.MODEL_DIR}`.\n\n'
                 'Convert your exports and train first:\n\n'
                 '```\npython main.py convert "path/to/exports"\n'
                 'python main.py train --all\n```')
        return

    with st.sidebar:
        st.header('Data')
        upload = st.file_uploader(
            'Score your own export', type=['html', 'htm'],
            help='An FM squad view saved as HTML -- a scouting shortlist, say. '
                 'CA is hidden in the game, so an export of players you do not '
                 'own has no CA column; the model does not need one.')

        if upload is None:
            position = st.selectbox('Position', positions)
            scored = load_scored(position)
            meta = load_meta(position)
            uploaded = False
        else:
            position = st.selectbox(
                'Score with which position model?', positions,
                index=guess_position(upload.name, positions),
                help='Pick the model matching the players in the file. A '
                     'striker scored by the goalkeeper model is meaningless.')
            meta = load_meta(position)
            trained_floor = meta.get('min_minutes', ca_model.MIN_MINUTES)
            floor = st.number_input(
                'Minimum minutes', min_value=0, max_value=5000,
                value=int(trained_floor), step=100,
                help=f'The model was trained on {trained_floor}+ minutes. Lower '
                     f'it to see scouted players with less football behind '
                     f'them, but their per-90 figures are noisier.')
            try:
                scored = score_upload(upload.getvalue(), upload.name,
                                      position, int(floor))
            except html_import.ConversionError as exc:
                st.error(str(exc))
                return
            except (KeyError, ValueError) as exc:
                st.error(f'Could not score that file: {exc}')
                return
            uploaded = True
            st.success(f'{len(scored):,} players scored from {upload.name}')

        st.header('Filters')

        priced = scored[scored[VALUE_COL].notna()]
        ceiling = float(priced[VALUE_COL].max()) if not priced.empty else 0.0
        steps = budget_steps(ceiling)
        budget = st.select_slider(
            'Maximum transfer value', options=steps, value=steps[-1],
            format_func=lambda v: 'Free only' if v == 0 else money(v),
            help='Players at or below this fee')

        choices = [s for s in SORTS
                   if not (uploaded and s in NEEDS_ACTUAL_CA)]
        sort_label = st.radio('Rank by', choices, index=0)
        nations = sorted(scored['Nation'].dropna().unique()) if 'Nation' in scored else []
        chosen = st.multiselect('Leagues (all if empty)', nations)

        min_ca = 0
        has_actual_ca = (not uploaded and ca_model.TARGET in scored
                         and scored[ca_model.TARGET].notna().any())
        if has_actual_ca:
            low = int(np.nanmin(scored[ca_model.TARGET]))
            high = int(np.nanmax(scored[ca_model.TARGET]))
            min_ca = st.slider('Minimum actual CA', low, high, low)

        st.divider()
        metrics = meta['metrics']
        st.caption(f"**Model** · trained {meta['trained_at'][:10]}")
        st.caption(f"Error ±{metrics['test_mae_CA_points']} CA · "
                   f"R² {metrics['test_r2']} · {metrics['n_train']} players")
        st.caption(f"Predictions are unbiased read forwards, but do not filter "
                   f"on a `CA_pred` threshold — only ~63% of truly elite players "
                   f"clear their own mark.")
        if uploaded:
            known = int(scored[ca_model.TARGET].notna().sum())
            st.caption(
                (f'{known} of these players list a CA. '
                 if known else 'None of these players list a CA, which is '
                               'normal -- it is hidden in game. ')
                + 'The underrated sorts are unavailable either way: they '
                  'need out-of-fold predictions, and the model may have been '
                  'trained on some of these players. Percentiles compare '
                  f'them against every {position} in your data.')

    view = scored[scored[VALUE_COL].notna() & (scored[VALUE_COL] <= budget)]
    if chosen:
        view = view[view['Nation'].isin(chosen)]
    if min_ca and ca_model.TARGET in view:
        view = view[view[ca_model.TARGET].fillna(min_ca) >= min_ca]

    if view.empty:
        st.warning('No players match those filters.')
        return

    sort_col = SORTS[sort_label]
    ranked = view.sort_values(sort_col, ascending=False)
    shown = ranked.head(points_to_plot(len(ranked)))

    left, right = st.columns([3, 1])
    with right:
        st.metric('Matching', f'{len(view):,}')
        st.metric('Plotted', f'{len(shown):,}')
        st.metric('Best predicted CA', f'{view[ca_model.PRED_COL].max():.0f}')
    with left:
        st.caption('Above the dashed line beats its price. '
                   'Click a point for the breakdown.')

    curve = value_curve(shown[VALUE_COL].to_numpy(),
                        shown[ca_model.PRED_COL].to_numpy())
    event = st.plotly_chart(scatter(shown, curve, 'Predicted CA'),
                            use_container_width=True, on_select='rerun',
                            selection_mode='points', key=f'scatter-{position}')

    selected = _selected_player(event, shown)

    if selected is not None:
        st.subheader(f"{selected[NAME_COL]}")
        facts = st.columns(5)
        facts[0].metric('Predicted CA', f"{selected[ca_model.PRED_COL]:.1f}")
        actual = selected.get(ca_model.TARGET)
        facts[1].metric('Actual CA', '—' if pd.isna(actual) else f'{actual:.0f}')
        facts[2].metric('Value', money(selected[VALUE_COL]))
        facts[3].metric('Division', str(selected.get('Division', '—'))[:22])
        adjusted = selected.get(ca_model.ADJ_RESIDUAL_COL)
        facts[4].metric('Underrated by',
                        '—' if pd.isna(adjusted) else f'{adjusted:+.1f}')
        st.plotly_chart(percentile_bars(scored, selected, chart_stats_for(position)),
                        use_container_width=True)

    st.subheader(f'Ranked by {sort_label.lower()}')
    columns = [c for c in (NAME_COL, 'Nation', 'Division', ca_model.TARGET,
                           ca_model.PRED_COL, ca_model.ADJ_RESIDUAL_COL,
                           'pred_above_league', VALUE_COL)
               if c in ranked.columns]
    table = ranked[columns].head(300).reset_index(drop=True)
    st.dataframe(
        table, use_container_width=True, height=420,
        column_config={
            VALUE_COL: st.column_config.NumberColumn('Value', format='%.0f'),
            ca_model.PRED_COL: st.column_config.NumberColumn('Pred CA', format='%.1f'),
            ca_model.ADJ_RESIDUAL_COL: st.column_config.NumberColumn(
                'Underrated', format='%.1f'),
            'pred_above_league': st.column_config.NumberColumn(
                'vs league', format='%.1f'),
        })


if __name__ == '__main__':
    main()

"""CLI for the FM data hub: train CA models and rank players with them.

    python main.py positions
    python main.py train Strikers
    python main.py rank Strikers --max-value 5000000
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from analysis import ca_model
from config import NAME_COL, VALUE_COL
from utils import html_import

# Stats charted on click: the model's most influential features. The league
# feature and its interaction terms are excluded -- they describe the division
# a player is in, not something you can read off the player.
CHART_STATS = 10
INTERACTION_SUFFIX = ' x league'


def _is_player_stat(feature):
    return (feature != ca_model.LEAGUE_FEATURE
            and not feature.endswith(INTERACTION_SUFFIX)
            and not feature.startswith(ca_model.LEAGUE_EFFECT_PREFIX))


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (KeyError, ValueError, FileNotFoundError, OSError,
            ca_model.ModelLoadError) as exc:
        sys.stdout.flush()  # keep the message after whatever the handler printed
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    listing = sub.add_parser('positions', help='list positions found in data/')
    listing.set_defaults(handler=cmd_positions)

    converter = sub.add_parser(
        'convert', help="convert FM's HTML exports into data/*.csv")
    converter.add_argument('paths', nargs='+',
                           help='HTML export files, or directories of them')
    converter.add_argument('--encoding', default='utf-8',
                           help='FM exports declare no charset and are UTF-8; '
                                'override only if names come out mangled')
    converter.add_argument('--table', type=int, default=0,
                           help='which table to take when a file has several')
    converter.add_argument('--drop-last', type=int, default=0, metavar='N',
                           help='also drop N trailing rows. Current exports '
                                'keep the sigames credit outside the table, so '
                                'this defaults to 0 -- a blind slice would '
                                'delete real players')
    converter.add_argument('--overwrite', action='store_true',
                           help='replace existing CSVs')
    converter.set_defaults(handler=cmd_convert)

    auditor = sub.add_parser(
        'audit', help='show which divisions are being dropped, worst first')
    auditor.add_argument('position', nargs='?',
                         help='position name; omit to pool every export')
    auditor.add_argument('-n', '--top', type=int, default=30,
                         help='how many unmapped divisions to list')
    auditor.add_argument('--min-players', type=int, default=1,
                         help='ignore divisions with fewer players than this')
    auditor.set_defaults(handler=cmd_audit)

    trainer = sub.add_parser('train', help='train and save a position model')
    trainer.add_argument('position', nargs='?',
                         help='position name; omit with --all')
    trainer.add_argument('--all', action='store_true',
                         help='train every position found in data/')
    _add_model_options(trainer)
    trainer.set_defaults(handler=cmd_train)

    comparer = sub.add_parser(
        'compare', help='A/B the modelling choices on your own data')
    comparer.add_argument('position')
    comparer.set_defaults(handler=cmd_compare)

    ranker = sub.add_parser('rank', help='score players with a saved model')
    ranker.add_argument('position')
    ranker.add_argument('-m', '--max-value', type=float, default=float('inf'),
                        help='highest transfer value to consider')
    ranker.add_argument('-n', '--top', type=int, default=25,
                        help='how many rows to print (default 25)')
    ranker.add_argument('-s', '--sort', default='ca_pred',
                        choices=['ca_pred', 'residual', 'residual_raw', 'above_league'],
                        help='ca_pred: best players; residual: most '
                             'underrated for their ability level; '
                             'residual_raw: the same before the shrinkage '
                             'correction, which mostly surfaces weak '
                             'players; above_league: best relative to their '
                             'division')
    ranker.add_argument('--no-plot', action='store_true',
                        help='print the table without opening the chart')
    ranker.set_defaults(handler=cmd_rank)
    return parser


def _add_model_options(parser):
    parser.add_argument(
        '--target-transform', default=ca_model.DEFAULT_TARGET_TRANSFORM,
        choices=sorted(ca_model.TARGET_TRANSFORMS),
        help="'log' fits log(CA), matching the diminishing returns of the CA "
             "scale; 'identity' fits CA directly (default: log)")
    parser.add_argument(
        '--league-interactions', action='store_true',
        help='also learn how much each stat should be scaled by league '
             'strength; doubles the feature count')
    parser.add_argument(
        '--no-league-effects', dest='league_effects', action='store_false',
        help='do not fit a per-league effect; rely on the assigned '
             'coefficient alone. Fitted effects are on by default because '
             'they cut error by 10-15%% and remove the need to estimate a '
             'strength for leagues outside UEFA')


def cmd_positions(args):
    positions = ca_model.available_positions()
    if not positions:
        print(f"No CSVs found in {ca_model.DATA_DIR}.\n"
              "Add one file per position, e.g. data/Strikers.csv")
        return 1
    for name, path in positions.items():
        trained = (ca_model.MODEL_DIR / f'{name}.joblib').exists()
        print(f"  {name:<28} {path.name:<28} "
              f"{'trained' if trained else 'not trained'}")
    return 0


def cmd_convert(args):
    """Turn FM's HTML exports into the CSVs `train` and `rank` read."""
    files = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob('*.html')))
        else:
            files.append(path)
    if not files:
        raise ValueError(f"No HTML files found in {args.paths}")

    written, existing = html_import.convert_all(
        files, ca_model.DATA_DIR, encoding=args.encoding, table=args.table,
        drop_last=args.drop_last, overwrite=args.overwrite)
    if not written:
        if existing:
            print(f"\nNothing to do: {existing} file(s) already converted. "
                  "Pass --overwrite to rebuild them.")
            return 0
        raise ValueError('nothing was converted')
    print(f"\n{len(written)} file(s) written to {ca_model.DATA_DIR}")
    print(f"Next: python main.py compare {written[0].stem}")
    return 0


def cmd_audit(args):
    """List the divisions costing you the most players, so you can map them.

    Every unmapped division is dropped at load time. This ranks what is being
    lost by player count, with the average CA of each so you can judge whether
    it is worth adding to leagues.DIVISIONS -- a regional amateur league is a
    fair thing to drop, a national second tier is not.
    """
    import pandas as pd
    from leagues import lookup_division
    from utils.preprocessing import to_numeric

    positions = ca_model.available_positions()
    if args.position:
        if args.position not in positions:
            raise KeyError(f"No data for {args.position!r}. "
                           f"Available: {sorted(positions) or 'none'}")
        positions = {args.position: positions[args.position]}
    if not positions:
        raise ValueError(f"No CSVs found in {ca_model.DATA_DIR}")

    frames = []
    for name, path in positions.items():
        frame = pd.read_csv(path, usecols=lambda c: c in
                            ('Division', 'Mins', ca_model.TARGET),
                            low_memory=False)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df['Mins'] = to_numeric(df['Mins'])
    df[ca_model.TARGET] = to_numeric(df[ca_model.TARGET])
    eligible = df[df['Mins'].fillna(0) >= ca_model.MIN_MINUTES]

    mapped = eligible['Division'].map(lambda d: lookup_division(d) is not None)
    kept, lost = int(mapped.sum()), int((~mapped).sum())
    print(f"Pooled {', '.join(positions)} | {len(eligible)} players with "
          f">= {ca_model.MIN_MINUTES} minutes")
    print(f"  mapped: {kept} ({kept / max(len(eligible), 1):.1%})   "
          f"dropped: {lost}")

    unmapped = eligible[~mapped]
    if unmapped.empty:
        print('\nEvery division is mapped.')
        return 0
    grouped = (unmapped.groupby('Division')
               .agg(players=(ca_model.TARGET, 'size'),
                    avg_ca=(ca_model.TARGET, 'mean'))
               .query('players >= @args.min_players')
               .sort_values('players', ascending=False))

    print(f"\n{len(grouped)} unmapped divisions. Top {args.top} by players lost:")
    print(f"  {'division':<46}{'players':>8}{'avg CA':>8}")
    print('  ' + '-' * 62)
    for name, row in grouped.head(args.top).iterrows():
        print(f"  {str(name)[:45]:<46}{int(row.players):>8}{row.avg_ca:>8.1f}")
    print("\nTo recover one, add it to DIVISIONS in leagues.py as "
          "'<name>': ('<Nation>', <tier>).")
    return 0


def cmd_train(args):
    if args.all:
        targets = list(ca_model.available_positions())
        if not targets:
            raise ValueError(f"No CSVs found in {ca_model.DATA_DIR}")
    elif args.position:
        targets = [args.position]
    else:
        raise ValueError('give a position name, or --all')

    for position in targets:
        print(f"=== TRAIN: {position} ===")
        result = ca_model.train(
            position, target_transform=args.target_transform,
            league_interactions=args.league_interactions,
            league_effects=args.league_effects)
        for key, value in result['meta']['metrics'].items():
            print(f"  {key:>22}: {value}")
        print(f"  saved -> {result['path']}")
        print('\n  Top weights (standardised, so directly comparable):')
        weights = ca_model.top_weights(result['pipeline'], result['meta']['features'])
        for stat, coef in weights.items():
            print(f"    {stat:>24}: {coef:+.3f}")
        print()
    return 0


def cmd_compare(args):
    """Train every combination of the modelling choices and tabulate them.

    Synthetic data cannot settle which functional form fits your save; this
    runs the comparison on your real export. Lower MAE is better. The winning
    combination is the one to pass to `train`.
    """
    configs = [
        ('coefficient only (no fitted effects)', 'identity', False, False),
        ('identity target', 'identity', False, True),
        ('log target', 'log', False, True),
        ('identity + league interactions', 'identity', True, True),
        ('log target + league interactions', 'log', True, True),
    ]
    print(f"{'config':<38}{'cv_r2':>8}{'test_r2':>9}{'MAE(CA)':>9}"
          f"{'resid~league':>14}{'feats':>7}")
    print('-' * 85)
    rows = []
    for label, transform, interactions, effects in configs:
        try:
            meta = ca_model.train(args.position, target_transform=transform,
                                  league_interactions=interactions,
                                  league_effects=effects,
                                  verbose=False)['meta']
        except ValueError as exc:
            print(f"{label:<38}  skipped: {exc}")
            continue
        m = meta['metrics']
        rows.append((m['test_mae_CA_points'], label, transform, interactions,
                     effects))
        print(f"{label:<38}{m['cv_r2_mean']:>8.3f}{m['test_r2']:>9.3f}"
              f"{m['test_mae_CA_points']:>9.2f}"
              f"{str(m['resid_vs_league_corr']):>14}{len(meta['features']):>7}")
    if not rows:
        raise ValueError('no configuration could be trained')

    _, label, transform, interactions, effects = min(rows)
    flags = f'--target-transform {transform}'
    flags += ' --league-interactions' if interactions else ''
    flags += '' if effects else ' --no-league-effects'
    print(f"\nLowest error: {label}")
    print(f"  python main.py train {args.position} {flags}")
    print("\nNote: the saved model is now whichever ran last. Re-run train "
          "with the winning flags before ranking.")
    return 0


def cmd_rank(args):
    scored = ca_model.score_players(args.position)

    if np.isfinite(args.max_value) and VALUE_COL in scored.columns:
        affordable = scored[scored[VALUE_COL] <= args.max_value]
        print(f"[{args.position}] {len(affordable)}/{len(scored)} players at or "
              f"below {args.max_value:,.0f}")
        scored = affordable
    if scored.empty:
        raise ValueError('No players matched the value ceiling')

    sort_col = {'ca_pred': ca_model.PRED_COL,
                'residual': ca_model.ADJ_RESIDUAL_COL,
                'residual_raw': ca_model.RESIDUAL_COL,
                'above_league': 'pred_above_league'}[args.sort]
    ranked = scored.sort_values(sort_col, ascending=False)

    columns = [c for c in (NAME_COL, 'Nation', ca_model.TARGET, ca_model.PRED_COL,
                           ca_model.RESIDUAL_COL, ca_model.ADJ_RESIDUAL_COL,
                           'pred_above_league', VALUE_COL)
               if c in ranked.columns]
    print(f"\n=== {args.position}: top {args.top} by {args.sort} ===")
    print(ranked[columns].head(args.top).round(1).to_string(index=False))
    if 'Nation' in ranked.columns:
        print('\nLeagues represented:')
        print(ranked.head(args.top)['Nation'].value_counts().to_string())

    if not args.no_plot:
        _plot(args.position, scored, ranked)
    return 0


def _plot(position, scored, ranked):
    """Open the value-for-money chart: transfer value against predicted ability."""
    from visualisation.plotter import plot_shortlist

    pipeline, meta = ca_model.load_model(position)
    weights = ca_model.top_weights(pipeline, meta['features'],
                                   n=len(meta['features']))
    chart_stats = [s for s in weights.index if _is_player_stat(s)][:CHART_STATS]

    missing = [c for c in (NAME_COL, VALUE_COL) if c not in scored.columns]
    if missing:
        print(f"Skipping the chart: {', '.join(missing)} not in this export.")
        return
    plottable = scored[scored[VALUE_COL].notna()]
    if plottable.empty:
        print('Skipping the chart: no player has a readable transfer value.')
        return

    plot_shortlist(plottable, ranked,
                   title=f'{position}: predicted ability vs transfer value',
                   chart_stats=chart_stats,
                   score_label='Predicted CA',
                   reference_label='median of those shown')


if __name__ == '__main__':
    raise SystemExit(main())

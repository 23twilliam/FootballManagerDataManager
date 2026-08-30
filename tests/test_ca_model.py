import joblib
import numpy as np
import pandas as pd
import pytest

from analysis import ca_model
from leagues import LEAGUE, nation_for_division

DIVISIONS = ['Premier League', 'LALIGA EA Sports', 'Bundesliga', 'Eliteserien']


def _frame(n=180, scouted_frac=0.7, seed=0, divisions=None):
    """A miniature FM export: stats driven by a latent ability, plus league mix."""
    rng = np.random.default_rng(seed)
    divisions = divisions or DIVISIONS
    division = rng.choice(divisions, n)
    coef = np.array([LEAGUE[nation_for_division(d)].coefficient
                     if nation_for_division(d) else 50.0 for d in division])
    latent = 90 + coef * 0.4 + rng.normal(0, 8, n)

    df = pd.DataFrame({
        'Name': [f'Player {i}' for i in range(n)],
        'Division': division,
        'Mins': [f'{m:,}' for m in rng.integers(200, 3400, n)],
        'Apps': [f'{s} ({b})' for s, b in zip(rng.integers(0, 38, n),
                                              rng.integers(0, 9, n))],
    })
    for i, col in enumerate(['Gls/90', 'Asts/90', 'Pas %', 'Drb/90', 'ShT/90']):
        signal = (latent - latent.mean()) / latent.std() * (0.8 - i * 0.1)
        values = np.clip(signal + rng.normal(0, 0.4, n) + 3, 0.01, None)
        df[col] = [f'{v:.2f}%' if '%' in col else round(v, 2) for v in values]

    df['Pts/Gm'] = np.round(latent / 100, 2)
    df['PA'] = (latent + 20).round().astype(int)
    df['Transfer Value'] = [f'£{v:.1f}M' for v in rng.uniform(0.2, 40, n)]

    ca = np.clip(latent, 40, 200).round().astype(int)
    scouted = rng.random(n) < scouted_frac
    df['CA'] = [str(c) if k else '-' for c, k in zip(ca, scouted)]
    return df


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the model layer at a throwaway data/model directory."""
    data, models = tmp_path / 'data', tmp_path / 'models'
    data.mkdir()
    monkeypatch.setattr(ca_model, 'DATA_DIR', data)
    monkeypatch.setattr(ca_model, 'MODEL_DIR', models)
    return data


@pytest.fixture
def strikers(workspace):
    _frame().to_csv(workspace / 'Strikers.csv', index=False)
    return 'Strikers'


# --- discovery ------------------------------------------------------------

def test_positions_are_discovered_from_the_data_directory(workspace):
    _frame(n=40).to_csv(workspace / 'Wingers.csv', index=False)
    _frame(n=40).to_csv(workspace / 'Keepers.csv', index=False)
    assert sorted(ca_model.available_positions()) == ['Keepers', 'Wingers']


def test_missing_position_names_what_is_available(strikers):
    with pytest.raises(KeyError, match='Strikers'):
        ca_model.load_position('Nonexistent', verbose=False)


# --- loading --------------------------------------------------------------

def test_minutes_filter_excludes_low_sample_players(strikers):
    data = ca_model.load_position(strikers, min_minutes=2000, verbose=False)
    assert (data.df['Mins'] >= 2000).all()


def test_unmapped_divisions_are_dropped(workspace):
    df = _frame(n=120, divisions=['Premier League', 'Vanarama National League'])
    df.to_csv(workspace / 'Mixed.csv', index=False)
    data = ca_model.load_position('Mixed', min_minutes=0, verbose=False)
    assert set(data.df['Division']) == {'Premier League'}
    assert data.df['Nation'].notna().all()


def test_unscouted_players_are_kept_but_unlabelled(strikers):
    """The model's main job is scoring players with stats but no CA."""
    data = ca_model.load_position(strikers, min_minutes=0, verbose=False)
    assert (~data.labelled).sum() > 0, 'fixture should contain unscouted players'
    assert data.labelled.sum() > 0
    assert len(data.X) == len(data.df)


def test_unparseable_ca_does_not_become_a_label(strikers):
    """A CA of '-' is unknown, not zero -- it must not train the model."""
    data = ca_model.load_position(strikers, min_minutes=0, verbose=False)
    assert data.y.min() > 40, 'a placeholder CA leaked in as a numeric label'
    assert data.df['CA'].dtype.kind == 'f'


def test_transfer_value_is_parsed_for_output(strikers):
    data = ca_model.load_position(strikers, min_minutes=0, verbose=False)
    assert data.df['Transfer Value'].dtype.kind == 'f'
    assert data.df['Transfer Value'].max() > 100_000


@pytest.mark.parametrize('leaky', ['CA', 'PA', 'Pts/Gm', 'Transfer Value',
                                   'Name', 'Division', 'Nation', 'Mins', 'Apps'])
def test_leaky_columns_never_become_features(strikers, leaky):
    data = ca_model.load_position(strikers, min_minutes=0, verbose=False)
    assert leaky not in data.X.columns


def test_league_strength_enters_on_a_log_scale(strikers):
    """avg_CA fits log(coefficient) far better than the raw value."""
    data = ca_model.load_position(strikers, min_minutes=0, verbose=False)
    assert ca_model.LEAGUE_FEATURE in data.X.columns
    assert 'league_coef' not in data.X.columns, 'raw coefficient leaked in'
    logged = data.X[ca_model.LEAGUE_FEATURE]
    assert logged.notna().all()
    expected = np.log(data.df['league_coef'])
    assert np.allclose(logged, expected)


@pytest.mark.parametrize('raw, starts, subs', [
    ('41 (2)', 41, 2), ('45', 45, 0), ('0 (13)', 0, 13), ('', 0, 0),
])
def test_parse_apps(raw, starts, subs):
    got_starts, got_subs = ca_model.parse_apps(pd.Series([raw]))
    assert (got_starts.iloc[0], got_subs.iloc[0]) == (starts, subs)


# --- training -------------------------------------------------------------

def test_coefficients_align_with_saved_feature_names(strikers):
    """SimpleImputer drops all-NaN columns unless keep_empty_features is set,
    which silently desyncs coef_ from meta['features']."""
    result = ca_model.train(strikers, verbose=False)
    coefs = ca_model.ridge_step(result['pipeline']).coef_
    assert len(coefs) == len(result['meta']['features'])
    weights = ca_model.top_weights(result['pipeline'], result['meta']['features'])
    assert not weights.empty


def test_a_column_empty_within_the_training_split_does_not_break_training(workspace):
    df = _frame(n=180)
    df['Rare/90'] = np.nan
    df.loc[df.index[-3:], 'Rare/90'] = 1.0  # populated for 3 rows only
    df.to_csv(workspace / 'Sparse.csv', index=False)
    result = ca_model.train('Sparse', verbose=False)
    assert 'Rare/90' in result['meta']['features']
    coefs = ca_model.ridge_step(result['pipeline']).coef_
    assert len(coefs) == len(result['meta']['features'])


def test_training_persists_a_loadable_bundle(strikers):
    result = ca_model.train(strikers, verbose=False)
    assert result['path'].exists()
    pipeline, meta = ca_model.load_model(strikers)
    assert meta['features'] == result['meta']['features']
    assert meta['target'] == 'CA'
    assert pipeline.predict(result['data'].X[meta['features']]).shape[0] == len(result['data'].X)


def test_alpha_is_chosen_by_cross_validation(strikers):
    result = ca_model.train(strikers, verbose=False)
    chosen = result['meta']['metrics']['chosen_alpha']
    assert np.isclose(ca_model.ALPHAS, chosen, atol=1e-3).any(), (
        f'{chosen} is not one of the searched alphas')


def test_model_learns_something(strikers):
    result = ca_model.train(strikers, verbose=False)
    assert result['meta']['metrics']['test_r2'] > 0.5


def test_too_few_labels_raises_rather_than_training_garbage(workspace):
    _frame(n=60, scouted_frac=0.05).to_csv(workspace / 'Tiny.csv', index=False)
    with pytest.raises(ValueError, match='too few to train'):
        ca_model.train('Tiny', verbose=False)


def test_residual_league_correlation_is_none_for_a_single_league(workspace):
    _frame(n=180, divisions=['Premier League']).to_csv(
        workspace / 'OneLeague.csv', index=False)
    result = ca_model.train('OneLeague', verbose=False)
    assert result['meta']['metrics']['resid_vs_league_corr'] is None


def test_untrained_position_raises_a_useful_error(strikers):
    with pytest.raises(FileNotFoundError, match='train'):
        ca_model.load_model(strikers)


# --- scoring --------------------------------------------------------------

def test_every_player_is_scored_including_the_unscouted(strikers):
    ca_model.train(strikers, verbose=False)
    scored = ca_model.score_players(strikers, verbose=False)
    assert scored[ca_model.PRED_COL].notna().all()
    assert scored['CA'].isna().sum() > 0, 'fixture should contain unscouted players'
    assert scored.loc[scored['CA'].isna(), ca_model.PRED_COL].notna().all()


def test_score_column_is_present_for_the_plotter(strikers):
    ca_model.train(strikers, verbose=False)
    scored = ca_model.score_players(strikers, verbose=False)
    assert (scored[ca_model.SCORE_COL] == scored[ca_model.PRED_COL]).all()


def test_residual_is_out_of_fold_not_in_sample(strikers):
    """In-sample residuals hug zero and would flatter trained players."""
    ca_model.train(strikers, verbose=False)
    scored = ca_model.score_players(strikers, verbose=False)
    residual = scored[ca_model.RESIDUAL_COL].dropna()
    assert len(residual) > 0
    in_sample = (scored[ca_model.PRED_COL] - scored['CA']).dropna()
    assert residual.abs().mean() > in_sample.abs().mean()


def test_residual_is_nan_where_there_is_no_ca(strikers):
    ca_model.train(strikers, verbose=False)
    scored = ca_model.score_players(strikers, verbose=False)
    assert scored.loc[scored['CA'].isna(), ca_model.RESIDUAL_COL].isna().all()


def test_scoring_realigns_columns_to_the_trained_feature_order(strikers, workspace):
    ca_model.train(strikers, verbose=False)
    _, meta = ca_model.load_model(strikers)

    # Re-export with the stat columns shuffled and one of them absent.
    df = pd.read_csv(workspace / 'Strikers.csv')
    reordered = df[list(reversed(df.columns))].drop(columns=['Drb/90'])
    reordered.to_csv(workspace / 'Strikers.csv', index=False)

    scored = ca_model.score_players(strikers, verbose=False)
    assert scored[ca_model.PRED_COL].notna().all()
    assert 'Drb/90' in meta['features']  # absent column imputed, not crashed


# --- league quality and the log scale -------------------------------------

def test_log_is_the_default_target_transform(strikers):
    result = ca_model.train(strikers, verbose=False)
    assert result['meta']['target_transform'] == 'log'


def test_log_target_predictions_come_back_in_ca_points(strikers):
    """The transform is internal: callers and metrics always see CA."""
    result = ca_model.train(strikers, verbose=False)
    scored = ca_model.score_players(strikers, verbose=False)
    predicted = scored[ca_model.PRED_COL]
    assert 40 < predicted.min() and predicted.max() < 220, (
        'predictions look like log values, not CA points')
    assert result['meta']['metrics']['test_mae_CA_points'] < 25


def test_identity_transform_still_available_for_ab_testing(strikers):
    result = ca_model.train(strikers, target_transform='identity', verbose=False)
    assert result['meta']['target_transform'] == 'identity'
    assert result['meta']['metrics']['test_r2'] > 0.5


def test_unknown_transform_is_rejected(strikers):
    with pytest.raises(ValueError, match='target_transform must be one of'):
        ca_model.train(strikers, target_transform='sqrt', verbose=False)


def test_log_target_rejects_non_positive_ca(workspace):
    df = _frame(n=180)
    df.loc[df.index[0], 'CA'] = '0'
    df.to_csv(workspace / 'Zero.csv', index=False)
    with pytest.raises(ValueError, match='zero or less'):
        ca_model.train('Zero', verbose=False)
    # identity has no such restriction
    ca_model.train('Zero', target_transform='identity', verbose=False)


def test_league_interactions_add_one_term_per_stat(strikers):
    plain = ca_model.load_position(strikers, min_minutes=0, verbose=False)
    crossed = ca_model.load_position(strikers, min_minutes=0,
                                     league_interactions=True, verbose=False)
    # League-effect columns are deliberately not crossed with league strength.
    stats = [c for c in plain.X.columns
             if c != ca_model.LEAGUE_FEATURE
             and not c.startswith(ca_model.LEAGUE_EFFECT_PREFIX)]
    assert len(crossed.X.columns) == len(plain.X.columns) + len(stats)
    assert all(f'{s} x league' in crossed.X.columns for s in stats)


def test_interaction_terms_are_centred_on_a_fixed_reference(strikers):
    """Centring on the loaded dataset would shift meaning between train and serve."""
    from leagues import LOG_COEF_REFERENCE
    data = ca_model.load_position(strikers, min_minutes=0,
                                  league_interactions=True, verbose=False)
    stat = 'Gls/90'
    expected = data.X[stat] * (data.X[ca_model.LEAGUE_FEATURE] - LOG_COEF_REFERENCE)
    assert np.allclose(data.X[f'{stat} x league'], expected)


def test_interaction_model_trains_and_scores(strikers):
    result = ca_model.train(strikers, league_interactions=True, verbose=False)
    assert result['meta']['league_interactions'] is True
    assert len(ca_model.ridge_step(result['pipeline']).coef_) == len(
        result['meta']['features'])
    scored = ca_model.score_players(strikers, verbose=False)
    assert scored[ca_model.PRED_COL].notna().all()


def test_serving_rebuilds_the_features_the_model_was_trained_with(strikers):
    """A model trained with interactions must not be served without them."""
    ca_model.train(strikers, league_interactions=True, verbose=False)
    _, meta = ca_model.load_model(strikers)
    assert meta['league_interactions'] is True
    assert any(' x league' in f for f in meta['features'])
    scored = ca_model.score_players(strikers, verbose=False)
    assert scored[ca_model.PRED_COL].notna().all()


def test_predictions_are_clipped_to_the_valid_ca_scale(strikers):
    """FM's CA is bounded; a prediction outside it is out of range, not bold."""
    ca_model.train(strikers, verbose=False)
    scored = ca_model.score_players(strikers, verbose=False)
    low, high = ca_model.CA_RANGE
    assert scored[ca_model.PRED_COL].between(low, high).all()


def test_cv_folds_are_shuffled(strikers):
    """FM exports arrive CA-sorted, so contiguous folds are ability bands."""
    splitter = ca_model.cv_splitter()
    assert splitter.shuffle is True
    assert splitter.random_state is not None


def test_out_of_fold_residuals_survive_a_ca_sorted_export(workspace):
    """Unshuffled folds made each split extrapolate, inflating every residual."""
    df = _frame(n=300).assign(CA=lambda d: pd.to_numeric(d['CA'], errors='coerce'))
    df = df.sort_values('CA', ascending=False)          # as FM exports them
    df['CA'] = df['CA'].astype('Int64').astype(str)
    df.to_csv(workspace / 'Sorted.csv', index=False)

    ca_model.train('Sorted', verbose=False)
    scored = ca_model.score_players('Sorted', verbose=False)
    mae = scored[ca_model.RESIDUAL_COL].abs().mean()
    spread = pd.to_numeric(scored['CA'], errors='coerce').std()
    assert mae < spread, (
        f'out-of-fold MAE {mae:.1f} exceeds the spread of CA itself ({spread:.1f}); '
        'the folds are not shuffled')


# --- fitted per-league effects --------------------------------------------

def test_league_effects_are_fitted_for_well_represented_divisions(strikers):
    data = ca_model.load_position(strikers, min_minutes=0, verbose=False)
    effects = [c for c in data.X.columns
               if c.startswith(ca_model.LEAGUE_EFFECT_PREFIX)]
    assert effects, 'expected at least one fitted league effect'
    counts = data.df.loc[data.labelled, 'Division'].value_counts()
    for column in effects:
        division = column[len(ca_model.LEAGUE_EFFECT_PREFIX):]
        assert counts[division] >= ca_model.MIN_LEAGUE_PLAYERS


def test_sparse_leagues_fall_back_to_the_coefficient(workspace):
    """A handful of players cannot pin down an intercept worth trusting."""
    rows = _frame(n=200, divisions=['Premier League'])
    rare = _frame(n=4, divisions=['Eliteserien'], seed=9)
    pd.concat([rows, rare], ignore_index=True).to_csv(
        workspace / 'Mixed.csv', index=False)
    data = ca_model.load_position('Mixed', min_minutes=0, verbose=False)
    assert f'{ca_model.LEAGUE_EFFECT_PREFIX}Eliteserien' not in data.X.columns
    assert ca_model.LEAGUE_FEATURE in data.X.columns


def test_league_effects_can_be_disabled(strikers):
    data = ca_model.load_position(strikers, min_minutes=0,
                                  league_effects=False, verbose=False)
    assert not any(c.startswith(ca_model.LEAGUE_EFFECT_PREFIX)
                   for c in data.X.columns)


def test_known_leagues_are_frozen_into_the_metadata(strikers):
    result = ca_model.train(strikers, verbose=False)
    known = result['meta']['known_leagues']
    assert known
    for division in known:
        assert f'{ca_model.LEAGUE_EFFECT_PREFIX}{division}' in result['meta']['features']


def test_serving_reuses_the_trained_league_columns(strikers, workspace):
    """A division absent at serve time must not shift the feature matrix."""
    ca_model.train(strikers, verbose=False)
    _, meta = ca_model.load_model(strikers)

    df = pd.read_csv(workspace / 'Strikers.csv')
    dropped = meta['known_leagues'][0]
    df[df['Division'] != dropped].to_csv(workspace / 'Strikers.csv', index=False)

    scored = ca_model.score_players(strikers, verbose=False)
    assert scored[ca_model.PRED_COL].notna().all()


def test_an_unseen_league_falls_back_rather_than_failing(strikers, workspace):
    ca_model.train(strikers, verbose=False)
    df = pd.read_csv(workspace / 'Strikers.csv')
    df.loc[df.index[:5], 'Division'] = 'Premier League'   # keep it mapped
    df.loc[df.index[5:10], 'Division'] = 'Eredivisie'     # not seen in training
    df.to_csv(workspace / 'Strikers.csv', index=False)
    scored = ca_model.score_players(strikers, verbose=False)
    assert scored[ca_model.PRED_COL].notna().all()


def test_interactions_do_not_cross_with_league_effect_columns(strikers):
    data = ca_model.load_position(strikers, min_minutes=0,
                                  league_interactions=True, verbose=False)
    crossed = [c for c in data.X.columns if c.endswith(' x league')]
    assert crossed
    assert not any(c.startswith(f'{ca_model.LEAGUE_EFFECT_PREFIX}')
                   for c in crossed)


# --- shrinkage correction --------------------------------------------------

def test_adjusted_residual_removes_the_ca_dependent_bias(strikers):
    """Ridge pulls predictions toward the mean, so the raw residual ranks by
    weakness rather than by being underrated."""
    ca_model.train(strikers, verbose=False)
    scored = ca_model.score_players(strikers, verbose=False)
    lab = scored[scored['CA'].notna()]
    low = lab[lab['CA'] < lab['CA'].median()]
    high = lab[lab['CA'] >= lab['CA'].median()]

    raw_gap = abs(low[ca_model.RESIDUAL_COL].mean() - high[ca_model.RESIDUAL_COL].mean())
    adj_gap = abs(low[ca_model.ADJ_RESIDUAL_COL].mean()
                  - high[ca_model.ADJ_RESIDUAL_COL].mean())
    assert adj_gap < raw_gap, 'correction did not reduce the CA-dependent bias'


def test_adjusted_residual_is_nan_without_a_ca(strikers):
    ca_model.train(strikers, verbose=False)
    scored = ca_model.score_players(strikers, verbose=False)
    assert scored.loc[scored['CA'].isna(), ca_model.ADJ_RESIDUAL_COL].isna().all()


def test_shrinkage_curve_is_monotonically_decreasing():
    ca = pd.Series([40, 60, 80, 100, 120, 140, 160, 180] * 4, dtype=float)
    residual = pd.Series([12, 9, 5, 1, -1, -4, -7, -10] * 4, dtype=float)
    curve = ca_model.shrinkage_curve(ca, residual)
    fitted = curve.predict(np.array([40., 80., 120., 180.]))
    assert list(fitted) == sorted(fitted, reverse=True)


# --- environment mismatch --------------------------------------------------

def test_a_model_trained_by_another_sklearn_warns(strikers, capsys, monkeypatch):
    """Unpickling across sklearn versions fails only at predict time, with an
    error that says nothing about versions."""
    ca_model.train(strikers, verbose=False)
    path = ca_model.MODEL_DIR / f'{strikers}.joblib'
    bundle = joblib.load(path)
    bundle['meta']['sklearn_version'] = '0.0.1-not-your-version'
    joblib.dump(bundle, path)

    ca_model.load_model(strikers)
    out = capsys.readouterr().out
    assert 'WARNING' in out and '0.0.1-not-your-version' in out
    assert f'train {strikers}' in out


def test_a_corrupt_artefact_says_how_to_fix_it(strikers):
    ca_model.train(strikers, verbose=False)
    path = ca_model.MODEL_DIR / f'{strikers}.joblib'
    path.write_bytes(b'not a joblib file')
    with pytest.raises(ca_model.ModelLoadError, match='Retrain it'):
        ca_model.load_model(strikers)


def test_matching_versions_are_silent(strikers, capsys):
    ca_model.train(strikers, verbose=False)
    capsys.readouterr()
    ca_model.load_model(strikers)
    assert 'WARNING' not in capsys.readouterr().out

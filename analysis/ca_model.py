"""Predict a player's FM Current Ability from their match statistics.

This replaces the correlation-weighted scorer. The old approach weighted each
stat by its correlation with `Pts/Gm`, which is a team outcome -- it measured
squad quality as much as player quality. Here the target is CA, an attribute of
the player, and a Ridge regression learns the weights jointly rather than
assuming each stat contributes independently.

The trained pipeline and its metadata are saved as one artefact so they can
never desync.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (KFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import VALUE_COL
from leagues import (LOG_COEF_REFERENCE, nation_for_division,
                     strength_for_division, tier_for_division)
from utils.preprocessing import to_numeric
from utils.value_conversion import value_to_float

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Overridable so several save games can be kept side by side.
DATA_DIR = Path(os.environ.get('FM_DATA_DIR', PROJECT_ROOT / 'data'))
MODEL_DIR = Path(os.environ.get('FM_MODEL_DIR', PROJECT_ROOT / 'models'))

TARGET = 'CA'
PRED_COL = 'CA_pred'
RESIDUAL_COL = 'residual_oof'
ADJ_RESIDUAL_COL = 'residual_adj'
LEAGUE_FEATURE = 'log_league_coef'

# Ability is logarithmic in league reputation and in CA itself: the step from
# 20 to 100 CA is a far bigger jump in real quality than 100 to 180. Fitting
# log(CA) rather than CA makes equal model errors mean equal *proportional*
# errors, which is the scale those quantities actually live on. Predictions
# are exponentiated back, so every metric below is still in CA points.
TARGET_TRANSFORMS = {
    'log': (np.log, np.exp),
    'identity': (None, None),
}
DEFAULT_TARGET_TRANSFORM = 'log'

# FM's CA scale is bounded. A prediction outside it is not a bold call, it is
# out of range, so served predictions are clipped back onto the scale. Training
# metrics are computed before clipping, so they still reflect raw model error.
CA_RANGE = (1.0, 200.0)
SCORE_COL = 'score'  # alias for PRED_COL so the plotter needs no special case

# Players below this many minutes have noisy per-90 stats, so they make poor
# training labels. Tune and watch the metrics.
MIN_MINUTES = 900

# Ridge penalties searched by cross-validation, rather than a fixed guess.
ALPHAS = np.logspace(-2, 3, 30)

CV_FOLDS = 5
CV_SEED = 42

# Prefix for per-league effect columns.
LEAGUE_EFFECT_PREFIX = 'lg='

# A league needs at least this many trainable players before it gets its own
# fitted effect. Below it, the coefficient prior carries the league instead --
# a handful of players cannot pin down an intercept worth trusting.
MIN_LEAGUE_PLAYERS = 20


def cv_splitter(seed: int = CV_SEED) -> KFold:
    """Shuffled folds. Never use the unshuffled default on this data.

    FM exports arrive sorted by CA, so contiguous folds are narrow ability
    bands -- on a real keeper export the five folds averaged CA 136, 114,
    103, 91 and 69. Each split then has to extrapolate outside the range it
    trained on, which took cross-validated R2 to -4.38 where shuffling gives
    +0.79, and inflated every out-of-fold residual by roughly 45%.
    """
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)

# Columns that must never become features.
#   Identifiers:  Name, Division, Nation, Club, Inf, Rec
#   The target:   CA, and PA (Potential Ability leaks CA directly)
#   Reputation:   CR (Current Reputation) and WR (World Reputation) are ability
#                 restated, not performance. On a real 9,260-keeper export they
#                 correlate +0.90 and +0.81 with CA -- against +0.51 for the
#                 best genuine stat -- and CR alone dominated the fitted model.
#                 A model that reads reputation is not scouting, it is looking
#                 up the answer.
#   Confounds:    Pts/Gm is a team outcome -- the reason we moved off it
#   Market price: Transfer Value and AP (Asking Price) are what we score
#                 *against*, so using them as features would make the
#                 value-for-money plot circular
#   Sample size:  Apps/Mins/Starts/Subs gate the rows, they are not skill
DROP_ALWAYS = frozenset({
    'Inf', 'Rec', 'Name', 'Division', 'Nation', 'Club', 'Position',
    'CA', 'PA', 'CR', 'WR', 'Pts/Gm', VALUE_COL, 'AP',
    'league_coef', 'league_avg_ca', 'tier',
    'Apps', 'Mins', 'Starts', 'Subs',
})


@dataclass
class PositionData:
    """Everything loading a position yields.

    df: every usable row, including players whose CA is unknown
    X:  the feature matrix, aligned to df
    y:  CA, NaN where the player has not been scouted
    """
    position: str
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    min_minutes: int
    league_interactions: bool
    league_effects: bool = True

    @property
    def labelled(self) -> pd.Series:
        """Mask selecting the rows that can be trained on."""
        return self.y.notna()

    @property
    def features(self) -> list[str]:
        return list(self.X.columns)


def available_positions() -> dict[str, Path]:
    """Discover positions from the CSVs present in data/.

    Data-driven rather than a hardcoded list, so adding a position means
    dropping in `data/<Position>.csv` and nothing else.
    """
    if not DATA_DIR.is_dir():
        return {}
    return {p.stem: p for p in sorted(DATA_DIR.glob('*.csv'))}


def parse_apps(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'41 (2)' -> (starts=41, subs=2). A bare '45' -> (45, 0)."""
    text = series.astype(str)
    starts = pd.to_numeric(text.str.extract(r'^\s*(\d+)')[0], errors='coerce').fillna(0)
    subs = pd.to_numeric(text.str.extract(r'\((\d+)\)')[0], errors='coerce').fillna(0)
    return starts, subs


def load_position(position: str, min_minutes: int = MIN_MINUTES,
                  league_interactions: bool = False,
                  league_effects: bool = True,
                  known_leagues: list[str] | None = None,
                  verbose: bool = True) -> PositionData:
    """Load one position's CSV, filter it, and build the feature matrix.

    Rows whose CA is unknown are kept. They cannot be trained on, but they are
    exactly the players worth predicting -- an unscouted player has stats but no
    rating, which is the case the model earns its keep in.
    """
    positions = available_positions()
    if position not in positions:
        raise KeyError(
            f"No data for {position!r}. Expected {DATA_DIR / (position + '.csv')}. "
            f"Available: {sorted(positions) or 'none -- data/ is empty or missing'}")

    df = pd.read_csv(positions[position])
    total = len(df)

    for required in ('Division', 'Mins'):
        if required not in df.columns:
            raise KeyError(f"{position}: required column {required!r} is missing")

    # --- sample-size filter -------------------------------------------------
    df['Mins'] = to_numeric(df['Mins'])
    if 'Apps' in df.columns:
        df['Starts'], df['Subs'] = parse_apps(df['Apps'])
    df = df[df['Mins'].fillna(0) >= min_minutes].copy()
    if verbose:
        print(f"[{position}] {len(df)}/{total} players with >= {min_minutes} minutes")

    # --- league strength ----------------------------------------------------
    df['Nation'] = df['Division'].map(nation_for_division)
    unmapped = sorted(df.loc[df['Nation'].isna(), 'Division'].dropna().unique())
    if unmapped and verbose:
        shown = ', '.join(unmapped[:5])
        print(f"[{position}] dropping {len(unmapped)} unmapped division(s): "
              f"{shown}{' ...' if len(unmapped) > 5 else ''}")
    df = df[df['Nation'].notna()].copy()
    if df.empty:
        raise ValueError(
            f"{position}: no players left after the minutes and league filters")

    # Strength is looked up per division, not per nation, so a second tier
    # carries its own reduced coefficient rather than its country's top-flight one.
    strength = df['Division'].map(strength_for_division)
    df['tier'] = df['Division'].map(tier_for_division)
    df['league_coef'] = strength.map(lambda s: s.coefficient)
    df['league_avg_ca'] = strength.map(lambda s: s.avg_ca)

    if VALUE_COL in df.columns:
        df[VALUE_COL] = df[VALUE_COL].map(value_to_float)

    df = df.reset_index(drop=True)

    # --- target -------------------------------------------------------------
    # Coerce before testing for missing: a CA that is present but unparseable
    # ("-", "") is unknown, not zero, and must not become a training label.
    if TARGET in df.columns:
        y = to_numeric(df[TARGET])
        df[TARGET] = y  # write back, so '-' does not linger in output
    else:
        y = pd.Series(np.nan, index=df.index, dtype='float64')

    # --- features -----------------------------------------------------------
    feature_cols = [c for c in df.columns if c not in DROP_ALWAYS]
    X = df[feature_cols].apply(to_numeric)
    X = X.dropna(axis=1, how='all')  # columns this position never records
    stat_cols = list(X.columns)

    # League strength enters on a log scale -- see LeagueStrength.log_coefficient.
    X[LEAGUE_FEATURE] = np.log(df['league_coef'].to_numpy())

    if league_effects:
        # One column per well-represented division, letting the model fit that
        # league's own effect instead of trusting an assigned coefficient.
        # This is not the kind of leakage reputation was: which league a player
        # plays in is visible to a scout, so it is a legitimate input. It also
        # removes the need to estimate a strength for leagues outside UEFA,
        # which have no coefficient to borrow.
        divisions = df['Division'].astype(str)
        if known_leagues is None:
            counts = divisions[y.notna().to_numpy()].value_counts()
            chosen = sorted(counts[counts >= MIN_LEAGUE_PLAYERS].index)
        else:
            chosen = list(known_leagues)
        effects = {f'{LEAGUE_EFFECT_PREFIX}{d}': (divisions == d).astype(float)
                   for d in chosen}
        if effects:
            X = pd.concat([X, pd.DataFrame(effects, index=X.index)], axis=1)
        if verbose:
            print(f"[{position}] fitted effects for {len(chosen)} league(s) "
                  f"with >= {MIN_LEAGUE_PLAYERS} players; the rest fall back "
                  f"to their coefficient")

    if league_interactions:
        # A goal in Gibraltar is not a goal in England. The log feature alone
        # only shifts a player's prediction by division; these terms let the
        # model scale each stat by league strength as well. Centred on a fixed
        # reference from the league table, never on the loaded dataset, so the
        # feature means the same thing at training and scoring time.
        centred = X[LEAGUE_FEATURE] - LOG_COEF_REFERENCE
        interactions = {f'{c} x league': X[c] * centred for c in stat_cols
                        if not c.startswith(LEAGUE_EFFECT_PREFIX)}
        X = pd.concat([X, pd.DataFrame(interactions, index=X.index)], axis=1)

    if verbose:
        known = int(y.notna().sum())
        print(f"[{position}] {len(X.columns)} features | {known} players with "
              f"known CA, {len(df) - known} without")
    return PositionData(position=position, df=df, X=X, y=y,
                        min_minutes=min_minutes,
                        league_interactions=league_interactions,
                        league_effects=league_effects)


def build_estimator(target_transform: str = DEFAULT_TARGET_TRANSFORM):
    """The transform + model chain. This whole object is what gets saved.

    With target_transform='log' the regression is fitted against log(CA) and
    its predictions exponentiated back, so callers and metrics still see CA
    points. Retransformation bias is left uncorrected: at the residual spread
    this model achieves it is worth well under a tenth of a CA point.
    """
    if target_transform not in TARGET_TRANSFORMS:
        raise ValueError(
            f"target_transform must be one of {sorted(TARGET_TRANSFORMS)}, "
            f"got {target_transform!r}")
    pipeline = Pipeline([
        # keep_empty_features keeps the column count stable. Without it the
        # imputer drops columns that are all-NaN within the training split, and
        # coef_ silently stops lining up with the saved feature list.
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()),
        # RidgeCV picks the penalty by cross-validation instead of guessing.
        ('model', RidgeCV(alphas=ALPHAS)),
    ])
    func, inverse = TARGET_TRANSFORMS[target_transform]
    if func is None:
        return pipeline
    return TransformedTargetRegressor(regressor=pipeline, func=func,
                                      inverse_func=inverse)


def train(position: str, test_size: float = 0.2, random_state: int = 42,
          target_transform: str = DEFAULT_TARGET_TRANSFORM,
          league_interactions: bool = False, league_effects: bool = True,
          verbose: bool = True) -> dict:
    """Train, evaluate honestly, and persist one position's model.

    Every metric is reported in CA points regardless of target_transform, so
    'log' and 'identity' runs can be compared directly.
    """
    data = load_position(position, league_interactions=league_interactions,
                         league_effects=league_effects, verbose=verbose)

    X = data.X.loc[data.labelled]
    y = data.y.loc[data.labelled]
    if len(X) < 30:
        raise ValueError(
            f"{position}: only {len(X)} players have a known CA -- too few to "
            "train on. Scout more players or lower MIN_MINUTES.")
    if target_transform == 'log' and (y <= 0).any():
        raise ValueError(
            f"{position}: {(y <= 0).sum()} player(s) have a CA of zero or less, "
            "which log cannot take. Fix the export or pass "
            "target_transform='identity'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    pipe = build_estimator(target_transform)
    cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv_splitter(random_state),
                            scoring='r2')
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    metrics = {
        'cv_r2_mean': round(float(cv_r2.mean()), 3),
        'cv_r2_std': round(float(cv_r2.std()), 3),
        'test_r2': round(float(r2_score(y_test, pred)), 3),
        'test_mae_CA_points': round(float(mean_absolute_error(y_test, pred)), 2),
        'resid_vs_league_corr': _residual_league_correlation(y_test, pred, X_test),
        'chosen_alpha': round(float(ridge_step(pipe).alpha_), 3),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'n_unlabelled': int((~data.labelled).sum()),
    }

    meta = {
        'position': position,
        'target': TARGET,
        'features': data.features,  # ORDER MATTERS at predict time
        'min_minutes': data.min_minutes,
        'target_transform': target_transform,
        'league_interactions': data.league_interactions,
        'league_effects': data.league_effects,
        # Frozen so serving rebuilds exactly the same columns. A division not
        # in this list falls back to its coefficient, which is what makes an
        # unseen league degrade gracefully rather than fail.
        'known_leagues': [c[len(LEAGUE_EFFECT_PREFIX):] for c in data.features
                          if c.startswith(LEAGUE_EFFECT_PREFIX)],
        'sklearn_version': sklearn.__version__,
        'trained_at': dt.datetime.now().isoformat(timespec='seconds'),
        'metrics': metrics,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODEL_DIR / f'{position}.joblib'
    joblib.dump({'pipeline': pipe, 'meta': meta}, artifact_path)
    (MODEL_DIR / f'{position}.json').write_text(json.dumps(meta, indent=2))

    return {'path': artifact_path, 'meta': meta, 'pipeline': pipe, 'data': data}


def _residual_league_correlation(y_test, pred, X_test):
    """Does the model's error still depend on league strength? Want ~0.

    Returns None when the test split spans a single league, where the
    correlation is undefined rather than zero.
    """
    league = X_test[LEAGUE_FEATURE].to_numpy()
    if np.ptp(league) == 0:
        return None
    resid = np.asarray(y_test) - pred
    return round(float(np.corrcoef(resid, league)[0, 1]), 3)


class ModelLoadError(Exception):
    """A saved model exists but cannot be used by the current environment."""


def _stale_model_message(position, path, exc) -> str:
    return (f"Could not load {path.name}: {type(exc).__name__}: {exc}\n"
            f"  This usually means the model was trained by a different "
            f"scikit-learn (you are on {sklearn.__version__}).\n"
            f"  Retrain it: python main.py train {position}")


def load_model(position: str):
    """The serve-side path: load a persisted model and its metadata."""
    path = MODEL_DIR / f'{position}.joblib'
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model for {position!r}. Run train({position!r}) first.")
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(_stale_model_message(position, path, exc)) from exc

    meta = bundle['meta']
    trained_with = meta.get('sklearn_version')
    if trained_with and trained_with != sklearn.__version__:
        # A pipeline pickled by one scikit-learn often cannot be used by
        # another: unpickling 1.5.1 artefacts under 1.9.0 yields a
        # SimpleImputer with no _fill_dtype, which fails only at predict
        # time with an error that says nothing about versions.
        print(f"[{position}] WARNING: model was trained with scikit-learn "
              f"{trained_with}, you are running {sklearn.__version__}. "
              f"Retrain if anything looks wrong: "
              f"python main.py train {position}")
    return bundle['pipeline'], meta


def ridge_step(estimator):
    """The fitted Ridge, whether or not a target transform wraps the pipeline."""
    inner = getattr(estimator, 'regressor_', estimator)
    return inner.named_steps['model']


def top_weights(pipeline, features: list[str], n: int = 12) -> pd.Series:
    """Standardised Ridge coefficients: directly comparable stat importances.

    Under a log target these are multiplicative: a coefficient of 0.05 means a
    one-standard-deviation rise in that stat lifts predicted CA by about 5%,
    not by 0.05 CA points.
    """
    coefs = ridge_step(pipeline).coef_
    if len(coefs) != len(features):
        raise ValueError(
            f"model has {len(coefs)} coefficients but {len(features)} feature "
            'names -- the pipeline dropped columns; check keep_empty_features')
    ordered = pd.Series(coefs, index=features).sort_values(key=np.abs, ascending=False)
    return ordered.head(n)


def score_players(position: str, verbose: bool = True) -> pd.DataFrame:
    """Load the saved model and score every player in a position.

    Adds:
      CA_pred            model's predicted ability, for every player
      score              alias of CA_pred, so the plotter needs no special case
      residual_oof       out-of-fold prediction minus actual CA
      pred_above_league  CA_pred minus the league's average CA

    residual_oof will NOT equal CA_pred - CA, and is named to say so. CA_pred
    comes from the model fitted on everything; the residual comes from folds
    that never saw the player. On a real keeper export the two agree to about
    2 CA points for most players and diverge sharply for a handful, which is
    itself worth knowing.

    `residual_oof` uses out-of-fold predictions for players the model was
    trained on. In-sample predictions sit optimistically close to their own
    labels, which would make trained players look falsely well-rated next to
    unseen ones.
    """
    pipe, meta = load_model(position)
    data = load_position(
        position,
        min_minutes=meta.get('min_minutes', MIN_MINUTES),
        league_interactions=meta.get('league_interactions', False),
        league_effects=meta.get('league_effects', False),
        known_leagues=meta.get('known_leagues'),
        verbose=verbose)

    missing = [c for c in meta['features'] if c not in data.X.columns]
    if missing and verbose:
        shown = ', '.join(missing[:5])
        print(f"[{position}] {len(missing)} trained feature(s) absent from this "
              f"export, imputed from training medians: {shown}"
              f"{' ...' if len(missing) > 5 else ''}")
    X = data.X.reindex(columns=meta['features'])

    out = data.df.copy()
    out[PRED_COL] = np.clip(pipe.predict(X), *CA_RANGE)
    out[SCORE_COL] = out[PRED_COL]
    out[RESIDUAL_COL] = _out_of_fold_residual(
        data, X, meta.get('target_transform', DEFAULT_TARGET_TRANSFORM))
    out[ADJ_RESIDUAL_COL] = _debiased_residual(out)
    out['pred_above_league'] = out[PRED_COL] - out['league_avg_ca']
    return out


def shrinkage_curve(ca: pd.Series, residual: pd.Series):
    """Fit how much the model's residual depends on the player's own CA.

    A regularised regression pulls predictions toward the mean, so weak
    players are over-predicted and elite ones under-predicted -- across nine
    positions the bias ran about +10 CA below 80 and -7 above 140. Left in,
    that dominates the residual ranking: every low-CA player looks underrated
    and no strong player ever does.

    The bias falls monotonically with CA, so an isotonic fit captures it
    without assuming a shape.
    """
    model = IsotonicRegression(increasing=False, out_of_bounds='clip')
    model.fit(ca.to_numpy(), residual.to_numpy())
    return model


def _debiased_residual(out: pd.DataFrame) -> pd.Series:
    """residual_oof with the shrinkage bias for that CA level removed.

    This is the column to rank on when hunting for players the game rates
    below their output: it answers "unusually underrated *for a player of
    this ability*", where the raw residual mostly answers "low CA".
    """
    adjusted = pd.Series(np.nan, index=out.index, dtype='float64')
    usable = out[TARGET].notna() & out[RESIDUAL_COL].notna()
    if usable.sum() < CV_FOLDS * 6:
        return adjusted
    curve = shrinkage_curve(out.loc[usable, TARGET], out.loc[usable, RESIDUAL_COL])
    expected = curve.predict(out.loc[usable, TARGET].to_numpy())
    adjusted.loc[usable] = out.loc[usable, RESIDUAL_COL].to_numpy() - expected
    return adjusted


def _out_of_fold_residual(data: PositionData, X: pd.DataFrame,
                          target_transform: str) -> pd.Series:
    """CA_pred - CA, using out-of-fold predictions where a label exists."""
    residual = pd.Series(np.nan, index=data.df.index, dtype='float64')
    labelled = data.labelled
    if labelled.sum() < CV_FOLDS * 6:
        return residual
    oof = cross_val_predict(build_estimator(target_transform), X.loc[labelled],
                            data.y.loc[labelled], cv=cv_splitter())
    # Clipped like CA_pred is, so both live on the same bounded scale.
    oof = np.clip(oof, *CA_RANGE)
    residual.loc[labelled] = oof - data.y.loc[labelled].to_numpy()
    return residual

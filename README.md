# FM Data Hub

Predicts a player's Current Ability from their match statistics, then plots
predicted ability against transfer value so bargains stand out.

## Install

```bash
python -m pip install -r requirements.txt
```

Verified on:

| | Version |
|---|---|
| Python | 3.14.7 |
| pandas | 3.0.5 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.0 |
| joblib | 1.5.3 |
| matplotlib | 3.11.1 |
| mplcursors | 0.7.1 |
| lxml | 6.1.2 |

**Use one interpreter throughout.** A model is a pickled scikit-learn
pipeline, and a pipeline saved by one minor release often cannot be used by
another — a 1.5.1 artefact loaded under 1.9.0 unpickles, then fails at
predict time with `'SimpleImputer' object has no attribute '_fill_dtype'`,
an error mentioning nothing about versions. `scikit-learn` is therefore
pinned to its minor version in `requirements.txt`; changing that line means
running `python main.py train --all` again. `load_model` warns when the
saved version differs from the one you are running.

The stale `.venv/` in this directory (Python 3.12 / scikit-learn 1.5.1) is
**not** what `python` resolves to here, and models trained in it will not
load. Either delete it or make sure you activate it consistently.

## Importing from the game

FM exports a squad view as HTML. Convert it first:

```bash
python main.py convert "C:/Users/you/Documents/Goalkeepers.html"
```

Pass several files or a directory. Each becomes `data/<filename>.csv`.

Two things the converter handles that a bare `pd.read_html(path)` does not:

- **Encoding.** FM's export declares no charset, so the parser guesses and on
  Windows decodes UTF-8 as cp1252. On a real 9,260-row export that mangled
  1,853 names (`Patryk Stępień` → `Patryk StÄpieÅ`) and broke five
  leagues out of the coefficient table, silently dropping 298 players.
  `--encoding` overrides it if your export differs.
- **Footer rows.** The sigames.com credit sits *outside* `</table>` in current
  exports, so `read_html` never sees it and slicing off the last two rows
  would delete real players. Junk rows are detected by content instead. Use
  `--drop-last N` if an export of yours does put them in the table.

## Data layout

Only squad exports are accepted — a file without `Name`, `Division` and
`Mins` is skipped, so pointing `convert` at a whole folder will not turn a
coefficient table into a bogus position.

One CSV per position in `data/`. The file name is the position name:

```
data/
  Strikers.csv
  AttackingMidfielders.csv
  Midfielders.csv
  Defenders.csv
```

Positions are discovered from the directory, so adding one means dropping in a
file — no code change. Set `FM_DATA_DIR` and `FM_MODEL_DIR` to keep several save
games side by side.

Required columns: `Division`, `Mins`, and the stat columns. `CA` is the training
target — players without it are still scored, just not trained on. `Name` and
`Transfer Value` are needed for the chart.

## Use

### The app

```bash
python -m streamlit run app.py
```

Use `python -m streamlit`, not bare `streamlit` — pip installs
`streamlit.exe` into a Scripts directory that is not on PATH by default
on this machine, so the bare command fails with `CommandNotFoundException`.

Position, budget, league and minimum-CA filters in the sidebar; the
value-for-money scatter as the main panel; click a point for that player's
percentile breakdown; the full ranking as a sortable table below.

### The CLI

Still the way to convert exports and train models.

```bash
python main.py positions
```

```bash
python main.py audit
```

`audit` ranks the divisions being dropped by how many players they cost, with each one's average CA so you can judge whether it is worth mapping.

```bash
python main.py compare Strikers
```

`compare` A/Bs the two modelling choices on your own data and prints the
flags for whichever wins. Then train with them:

```bash
python main.py train --all --target-transform log
```

```bash
python main.py rank Strikers --max-value 5000000
```

Sort modes:

| `--sort` | Answers |
|---|---|
| `ca_pred` | Who is best? |
| `residual` | Who does FM rate below what their stats say, **for their ability level**? |
| `residual_raw` | The same before the shrinkage correction — mostly surfaces weak players. |
| `above_league` | Who stands out most relative to their division? |

Grouped by *actual* CA, weak players carry a positive residual (+7 to +12
below CA 80) and elite players a negative one (−6 to −10 above 140).
`residual` removes that with an isotonic fit of residual against CA, cutting
the bias to under ±2 in every band. `residual_raw` keeps the uncorrected
figure.

This is **not** caused by the Ridge penalty — dropping alpha to 0.001 leaves
the high-CA bias unchanged at −6.7 — nor by stats saturating at the top;
`Gls/90` and `ShT/90` keep climbing across every band. It is the ordinary
attenuation of any predictor with irreducible error, and it only appears
when you group by the answer. See *Reading a prediction* below.

Hover a point for a name; click for a percentile breakdown of the stats that
most influenced the model.

Every bar reads the same way round: **higher is better**. Stats where a lower
number is the better one — possession lost, headers lost, minutes per goal —
are flipped and labelled *(fewer)*, so a player who rarely gives the ball away
shows as a strength rather than a red bar at the 10th percentile. The set is
declared in `config.LOWER_IS_BETTER` and cannot be inferred: possession lost
correlates *positively* with CA (+0.166 across the nine positions) because
better players have the ball more, so a data-derived direction would rank
giving it away as a virtue.

The dashed line is **what that fee normally buys** — a rolling median of
predicted ability across players sorted by price. Players above it beat their
price; that is the whole point of the chart. A flat population-average line
was useless here, because the plot only draws the strongest half of the
shortlist, so a whole-population reference sat at or below the bottom of the
cloud every time.

Two details make it span the whole axis. The window **shrinks** at the ends
rather than sliding inward, so the curve keeps moving over the cheapest and
dearest players instead of flatlining. And the point at a price of zero is
**extrapolated** from the gradient between £50K and £500K rather than measured:
free transfers are released players and expiring contracts, a different
population whose median ability says nothing about what a fee buys. The
reference window shrinks to fit when a budget cap means prices never reach it.

Transfer value is on a symlog axis. Values span five orders of magnitude and
pile up at zero, so a linear axis crushed almost every player into the left
edge; symlog keeps a linear stretch near zero, since free transfers are real
and need to stay visible.

## How it works

A Ridge regression predicts `CA` from per-90 statistics, one model per position.

- **Target is CA, not `Pts/Gm`.** Points per game is a team outcome, so weighting
  stats by their correlation with it measured squad quality as much as player
  quality.
- **Weights are learned jointly.** Correlation weighting treats each stat as
  independent; regression accounts for stats that carry the same information.
- **Each league's effect is fitted, not assumed.** Any division with at least
  20 trainable players gets its own column, so the model learns that league's
  effect from its players rather than trusting an assigned coefficient. This
  cut error 10-15% and — more usefully — removes the need to *estimate* a
  strength for leagues outside UEFA, which have no coefficient to borrow.
  Sparser leagues fall back to their coefficient, so unseen divisions degrade
  rather than fail. Disable with `--no-league-effects`.
  
  This is not the leakage that reputation was: which league a player plays in
  is visible to a scout, so it is a legitimate input. Be aware, though, that a
  fitted per-league effect absorbs league-level signal, which is exactly what
  `--sort above_league` is looking for.
- **League strength is per division, not per nation.** A second tier is a
  fraction of its parent's coefficient (tier 2 ≈ 0.245, tier 3 ≈ 0.142),
  which covers hundreds of divisions without inventing a number for each.
  Derived averages land close to observed: the Championship predicts 115.9
  against 114.2 measured, League One 104.0 against 100.0.
- **Cross-validation folds are shuffled.** FM exports arrive sorted by CA, so
  the unshuffled default makes each fold a narrow ability band that the other
  folds must extrapolate into — worth −4.38 R² against +0.79 shuffled.
- **League strength enters on a log scale.** Reputation relates to ability
  logarithmically, not linearly. Fitting the league table's own `avg_ca`
  against each form gives R² 0.834 for the raw UEFA coefficient and **0.912**
  for its log (RMSE 6.89 → 5.02 CA points). `resid_vs_league_corr` should sit
  near zero — if it drifts, league bias is still leaking into the errors.
- **The target is log(CA) by default.** The CA scale has diminishing returns:
  20 → 100 is a far bigger jump in real quality than 100 → 180. Fitting
  log(CA) makes equal model errors mean equal *proportional* errors, which is
  the scale ability actually lives on. Predictions are exponentiated back and
  clipped to 1–200, so every number you see is still in CA points.
- **The penalty is cross-validated.** `RidgeCV` picks alpha rather than guessing.

### What is excluded from the features, and why

| Column | Reason |
|---|---|
| `CA` | The target. |
| `PA` | Potential Ability; correlates +0.82 with CA. |
| `CR`, `WR` | Current / World Reputation. **+0.90 and +0.81** with CA, against +0.51 for the best real stat. A model that reads reputation is looking up the answer, not scouting. |
| `Pts/Gm` | Team outcome, not player quality. |
| `Transfer Value`, `AP` | Market price — what we score *against*. Using it would make the value-for-money plot circular. |
| `Apps`, `Mins` | Sample size; they gate the rows, they are not skill. |

Excluding reputation costs real accuracy — on a goalkeeper export MAE rose
from 4.65 to 6.51 CA points. The 4.65 was not scouting.

### Reading the metrics

| Metric | What to watch for |
|---|---|
| `cv_r2_mean` / `test_r2` | Close together. A big gap means overfitting. |
| `test_mae_CA_points` | Average error in CA points. Under ~5 is usable. |
| `resid_vs_league_corr` | Near zero. Non-zero means league bias remains. |
| `n_unlabelled` | Players scored but not trained on — the model's real job. |
| `chosen_alpha` | At the top of the searched range means the data is noisy. |

### Reading a prediction

Grouped by **predicted** CA — the direction you actually read the tool — the
model is unbiased: players predicted 140-200 average an actual CA of 147.5,
and every band sits within ±1 CA point. So when it says 145, take 145.

The bias only appears grouped by **actual** CA, which conditions on the answer
and selects players whose ability comes partly from things per-90 stats do not
measure. Two consequences worth knowing:

- **Do not filter on an absolute `CA_pred` threshold.** Only 63% of players
  with a real CA of 140+ are predicted at 140+, falling to 30% at CA 160.
- **Rank, and cast a wide net.** The top 50 by `CA_pred` contains 27 of the
  true top 50; the top 200 contains 48 of them.

Among elite players the model's error (±7.5 CA) is close to the entire spread
of the group (sd 9.8), so it separates them only weakly — rank correlation
falls from +0.92 overall to +0.59 within CA 140+.

`residual_oof` uses out-of-fold predictions for trained players, so it is
comparable with unseen ones. In-sample residuals would hug zero and make trained
players look falsely well-rated. It will not equal `CA_pred - CA`; the
name says so.

## Layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit front end |
| `main.py` | CLI: `convert`, `positions`, `audit`, `compare`, `train`, `rank` |
| `analysis/ca_model.py` | Loading, training, persistence, scoring |
| `leagues.py` | League strengths, and division → (nation, tier) mapping |
| `config.py` | Column names and display labels |
| `utils/preprocessing.py` | Unit stripping and numeric coercion |
| `utils/value_conversion.py` | Transfer-value string parsing |
| `utils/html_import.py` | FM HTML export → CSV |
| `visualisation/shortlist.py` | Chart maths, shared by both front ends |
| `visualisation/plotter.py` | matplotlib scatter, used by `main.py rank` |
| `visualisation/percentile_chart.py` | Per-player percentile breakdown |

### Modelling options

| Flag | Effect |
|---|---|
| `--target-transform log` | Fit log(CA). Default — matches the diminishing returns of the CA scale. |
| `--target-transform identity` | Fit CA directly. Use if `compare` says it wins. |
| `--league-interactions` | Also learn how much each stat should be scaled by league strength. Doubles the feature count; a goal in Gibraltar is not a goal in England. |


## Tests

```bash
python -m pytest
```

## Known limitations

- Minutes below `MIN_MINUTES` (900) are excluded — per-90 stats are too noisy
  below that. Promising youngsters with few minutes are invisible.
- `leagues.py` maps all 54 UEFA nations, 10 non-UEFA leagues and the major
  second-to-fourth tiers — about 32% of a real export. Everything else is
  dropped rather than scored against a coefficient it has no basis for. Run
  `audit` to see what is being lost and extend `DIVISIONS`.
- Non-UEFA strengths are **estimates** — no UEFA coefficient exists for MLS or
  the Saudi Pro League. They now matter much less, because any such league
  with 20+ players gets a fitted effect instead. They still carry the sparse
  ones. Two cross-checks on a real export: median transfer value predicts the
  UEFA coefficient at R² 0.911 (±25%), and agreed with the hand estimates
  within 20% for Brazil, Russia, USA, Mexico and Argentina — but implied 86
  for Saudi Arabia against an estimate of 55, because state money inflates
  fees above playing quality. Market value tracks a league's economy, not its
  strength.
- Tier factors (a second tier carries 0.245 of its parent's coefficient) are
  calibrated from 15 divisions in one export. Tier 4 rests on a single
  division.
- Ridge is linear in its features. `--league-interactions` adds the one
  interaction that matters most, but stat-by-stat interactions ("good at X
  *and* Y") would need a gradient-boosted model, at the cost of interpretable
  weights.
- Under a log target, coefficients are multiplicative: 0.05 means a
  one-standard-deviation rise in that stat lifts predicted CA by about 5%,
  not by 0.05 CA points.
- CA is FM's own rating, so the model learns to reproduce FM's opinion. The
  `residual` sort is where it disagrees, which is the interesting part.

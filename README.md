# FM Data Hub

Predicts a player's Current Ability from their match statistics, then plots
predicted ability against transfer value so bargains stand out.

## Install

```bash
pip install -r requirements.txt
```

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

```bash
python main.py positions
```

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
| `residual` | Who does FM rate below what their stats say? |
| `above_league` | Who stands out most relative to their division? |

Hover a point for a name; click for a percentile breakdown of the stats that
most influenced the model.

## How it works

A Ridge regression predicts `CA` from per-90 statistics, one model per position.

- **Target is CA, not `Pts/Gm`.** Points per game is a team outcome, so weighting
  stats by their correlation with it measured squad quality as much as player
  quality.
- **Weights are learned jointly.** Correlation weighting treats each stat as
  independent; regression accounts for stats that carry the same information.
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

`residual_oof` uses out-of-fold predictions for trained players, so it is
comparable with unseen ones. In-sample residuals would hug zero and make trained
players look falsely well-rated. It will not equal `CA_pred - CA`; the
name says so.

## Layout

| Path | Purpose |
|---|---|
| `main.py` | CLI: `convert`, `positions`, `compare`, `train`, `rank` |
| `analysis/ca_model.py` | Loading, training, persistence, scoring |
| `leagues.py` | UEFA coefficients and division → nation mapping |
| `config.py` | Column names and display labels |
| `utils/preprocessing.py` | Unit stripping and numeric coercion |
| `utils/value_conversion.py` | Transfer-value string parsing |
| `utils/html_import.py` | FM HTML export → CSV |
| `visualisation/plotter.py` | Value-for-money scatter plot |
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
- Only the 34 top flights in the coefficient table are mapped. Every other
  division is dropped at load time rather than scored against a coefficient it
  has no basis for.
- Ridge is linear in its features. `--league-interactions` adds the one
  interaction that matters most, but stat-by-stat interactions ("good at X
  *and* Y") would need a gradient-boosted model, at the cost of interpretable
  weights.
- Under a log target, coefficients are multiplicative: 0.05 means a
  one-standard-deviation rise in that stat lifts predicted CA by about 5%,
  not by 0.05 CA points.
- CA is FM's own rating, so the model learns to reproduce FM's opinion. The
  `residual` sort is where it disagrees, which is the interesting part.

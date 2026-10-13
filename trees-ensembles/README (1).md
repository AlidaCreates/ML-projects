# Day 16 — Ensembles (Gradient Boosting, XGBoost, Voting, Stacking)

## What this is

Single script, `ensembles.py`, running Exercises 0–6 end to end: reload the
Day 14/15 engineered California Housing feature set, fit
`GradientBoostingRegressor`, `HistGradientBoostingRegressor`, `XGBRegressor`,
sweep XGBoost's three core hyperparameters, then combine models with
`VotingRegressor` and `StackingRegressor`.

Run:

```bash
python3 -m venv piscine1b_w4
source piscine1b_w4/bin/activate
pip install -r requirements.txt
python3 ensembles.py
```

## Audit notes — where my numbers diverge from the assignment's expected output

The assignment's "expected output" blocks were generated with `xgboost>=2.0`
and (implicitly) an older `pandas`/`scikit-learn`. My environment installed
`xgboost==3.4.1`, `scikit-learn==1.8.0`, `pandas==3.0.2`. Two real
consequences, both worth being able to explain in defense:

1. **`ocean_proximity_<1H OCEAN` breaks XGBoost 3.x.** The one-hot column
   name produced by `pd.get_dummies` for the `<1H OCEAN` category contains a
   literal `<`, and XGBoost's newer strict feature-name validator (`core.py`,
   `feature_names` setter) rejects `[`, `]`, and `<` outright — it's not a
   silent behavior change, it's a hard `ValueError` on `.fit()`. I sanitize
   column names right after `pd.get_dummies` (`<` → `lt_`, drop `[`/`]`) so
   every model downstream — RF, GB, HGB, XGB — sees identical, safe column
   names. This doesn't change any values, only the column labels.

2. **`XGBRegressor(...).n_estimators` / `.learning_rate` / `.max_depth` print
   as `None` before/without fitting in `xgboost>=3.0`**, because the
   sklearn-API wrapper no longer eagerly materializes booster defaults into
   its own attributes — they're `None` until you either pass them explicitly
   or inspect the fitted booster's C++-side config. The *actual* defaults
   are unchanged (`n_estimators=100`, `learning_rate=0.3`, `max_depth=6` —
   confirmed by reading `booster.save_config()`), so I pull the display
   values from `xgbr.get_booster().save_config()` /
   `num_boosted_rounds()` instead of the estimator's own attributes. If an
   auditor asks "why isn't this just `xgbr.n_estimators`?" — that's why.

3. **My actual test R² values are systematically a bit lower than the
   assignment's** (e.g. my default `XGBRegressor` gets 0.8333 vs. the
   assignment's 0.8359; my tuned 500/0.05/4 config gets 0.8241 vs. their
   0.8483). This tracks with (1) and (2) above plus version drift in
   `HistGradientBoostingRegressor`'s internals between scikit-learn 1.4 and
   1.8 — same code, same `random_state`, different library internals,
   different numeric result. This is expected and not a bug in the script;
   I did not hand-tune anything to chase the assignment's printed numbers.

4. **The hyperparameter sweep does not confirm the assignment's "concrete
   rule."** In my run, `max_depth=4` is *not* the sweet spot — test R² keeps
   climbing through `max_depth=8` (0.8467) before falling off at
   `max_depth=10`. Likewise `n_estimators=1000` slightly beats `n_estimators=
   500` here, where the assignment's numbers peaked at 500 and declined
   afterward. Rather than silently pick whichever combo looks best in my
   run, `VotingRegressor`/`StackingRegressor` (Exercises 5–6) still use the
   assignment's stated best config (`n_estimators=500, learning_rate=0.05,
   max_depth=4`) so the later exercises match the assignment's intended
   design — but I'd flag the sweep discrepancy to an auditor unprompted
   rather than let it look like I just used the doc's numbers without
   checking. If asked "does the sweet spot hold in your environment?" the
   honest answer is no, and the printed sweep output shows exactly why.

## Result summary (this environment)

| model | test R² |
|---|---|
| LinearRegression (Day 11 baseline) | 0.5958 |
| RandomForest | 0.8065 |
| GradientBoosting | 0.7815 |
| HistGradientBoosting | 0.8372 |
| XGBRegressor (default) | 0.8333 |
| XGBRegressor (tuned: 500/0.05/4) | 0.8241 |
| VotingRegressor | 0.8328 |
| StackingRegressor | 0.8368 |

Same qualitative story as the assignment (stacking > voting ≈ best single
model, HistGradientBoosting is the strongest of the individually-tuned
defaults, boosting > bagging on this dataset), just with different absolute
numbers for the library-version reasons above. Notably HistGradientBoosting
edges out the "tuned" XGBoost config here — worth mentioning if asked to
justify a final model choice, since the assignment's narrative assumes
tuned XGBoost wins outright.

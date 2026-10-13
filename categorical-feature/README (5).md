# Day 15 — Categorical Feature Engineering (`ocean_proximity`)

California Housing dataset. Continues directly from Day 14 (numeric feature
engineering): same dataset, same `random_state=42` split, but `ocean_proximity`
is kept this time and encoded four different ways.

## Setup

```bash
source piscine1b_w3/bin/activate      # or: python -m venv piscine1b_w3 && source ...
pip install -r requirements.txt
python categorical_fe.py
```

## What the script does

One script, `categorical_fe.py`, run top to bottom, exercise by exercise:

| Ex | What | Result (test R², LinearRegression) |
|----|------|--------------------------------------|
| 0 | Baselines: raw (Day 11) and numeric-engineered (Day 14), both drop `ocean_proximity` | 0.6139 / 0.5942 |
| 1 | One-hot via `pd.get_dummies` | 0.6023 |
| 2 | One-hot via `sklearn.OneHotEncoder` — confirms it's numerically identical to Ex1 | 0.6023 |
| 3 | Ordinal, explicit order `INLAND < <1H OCEAN < NEAR OCEAN < NEAR BAY < ISLAND` | 0.5972 |
| 4 | Target (mean) encoding, train-only means + a deliberate leaky version for comparison | 0.6026 / 0.6026 (leaky) |
| 5 | `engineer_full()` — the combined numeric + one-hot pipeline, reindexed onto train columns | 0.6023 |
| 6 | Full grid: 5 feature sets × {LinearRegression, RandomForestRegressor} | see table below |

Final grid (Ex6):

| feature_set | linreg | random_forest |
|---|---|---|
| raw (no ocean) | 0.6139 | 0.8101 |
| numeric-engineered (no ocean) | 0.5942 | 0.8021 |
| + one-hot ocean | 0.6023 | 0.8065 |
| + ordinal ocean | 0.5972 | 0.8061 |
| + target-encoded ocean | 0.6026 | 0.8061 |

## Decisions (and why)

- **Kept: one-hot encoding.** Cleanest, no leakage risk, no arbitrary ordering
  assumption. +0.0082 R² over numeric-only for LinearRegression, and the best
  categorical encoding for RandomForest too (0.8065).
- **Rejected: ordinal encoding.** The "distance from ocean" order is a guess,
  and it doesn't match the actual price relationship — `ISLAND` is the most
  expensive category but sits at the *top* of the ordinal scale, which a
  single linear coefficient can't represent without distorting the other
  categories. -0.0052 R² vs one-hot.
- **Rejected (but noted): target encoding.** Essentially tied with one-hot
  (+0.0002 R²) and uses one column instead of five, but it's more code, has a
  real leakage failure mode, and needs an explicit fallback for
  unseen-category NaNs — not worth it for 5 well-populated categories. Would
  reconsider for a high-cardinality categorical where one-hot blows up the
  column count.

## Audit notes (non-obvious design decisions)

1. **`engineer_full(X, train_columns=None)` reindexes test onto train's
   columns.** `pd.get_dummies` only creates columns for categories actually
   present in the DataFrame it's called on. If a rare category (e.g.
   `ISLAND`, n=5) doesn't land in the test split, `get_dummies(X_test)` would
   silently produce one fewer column than train and misalign the model
   input. Reindexing with `fill_value=0` guarantees identical column sets and
   order regardless of which categories the split happened to separate.

2. **Target encoding leakage is deliberately demonstrated, not just
   described.** `cat_means` (clean, Ex4) is fit on `X_train` only;
   `leaky_means` is fit on the full `X`/`y` (train+test) as a worked
   counter-example. In this dataset the two give nearly identical scores
   (0.6026 vs 0.6026) because the per-category means happen to be stable
   across an 80/20 random split of 20k+ rows — that's a property of *this*
   dataset, not a guarantee. On a smaller dataset or a high-cardinality
   categorical, the leaky version would score noticeably higher on the test
   set while performing worse in production, since production data was never
   part of the "training" mean.

3. **Unseen-category fallback for target encoding uses `y_train.mean()`, not
   `y.mean()`.** Consistent with "test-set information must never touch the
   encoding" — even the fallback value for an unmapped category comes from
   train only.

4. **`OneHotEncoder(handle_unknown="ignore")`.** Not exercised by this
   dataset (all 5 categories appear in both splits), but it's the reason
   `OneHotEncoder` is preferred over `pd.get_dummies` in a real pipeline: an
   unseen category at inference time gets encoded as all-zeros instead of
   raising or silently misaligning columns.

5. **Numeric features are re-derived from `expm1(log1p(x))`, not from the
   raw column, inside `engineer_numeric`.** Same as Day 14 — the log
   transform is applied to `total_rooms` / `total_bedrooms` / `population` /
   `households` first (for the linear model's benefit), then the ratio
   features (`rooms_per_household`, etc.) are computed by undoing that
   transform via `expm1`, so the ratios stay on the original scale rather
   than being ratios-of-logs.

6. **RandomForest scores drift run-to-run despite `random_state=42`.** The
   forest's own seed is fixed, but `n_jobs=-1` parallelism can produce
   ±0.005-ish floating point differences depending on core count/scheduling.
   This is expected and noted in the exercise spec — don't chase exact
   digit-for-digit reproduction on the RF column.

## Anticipated audit / defense questions

- **"Why does one-hot *reduce* R² vs the raw (unencoded) baseline in Ex5/Ex6?"**
  Because the raw baseline in Ex0/Ex6 still contains `ocean_proximity` as a
  raw string column dropped entirely — no, wait, it's dropped in both. The
  real answer: the raw baseline (0.6139) has no numeric feature engineering
  at all, while "numeric-engineered" (0.5942) and "+ one-hot" (0.6023) do.
  The Day 14 numeric engineering itself cost R² for `LinearRegression`
  (multicollinearity from `rooms_per_household` etc. correlating with
  existing columns); one-hot ocean recovers *some* but not all of that loss.
  It's a controlled comparison, not evidence that categorical encoding hurts
  — see the ordinal/target rows for the categorical-only effect
  (numeric-engineered → +one-hot is +0.0082).
- **"Why is ordinal encoding fine for RandomForest but not LinearRegression?"**
  A tree can split on the encoded column at any threshold and isolate
  `ISLAND` on its own branch regardless of where it sits on the number line.
  A linear model assigns one coefficient to the whole column, so it must
  treat the encoding as evenly spaced and monotonic in its effect on price —
  which `INLAND < ... < ISLAND` isn't.
- **"What happens if a new category appears in production that wasn't in
  training?"** `pd.get_dummies` path: reindex to train columns handles it
  (new category's dummy simply doesn't exist, row gets all-zeros across the
  known categories). `OneHotEncoder` path: `handle_unknown="ignore"` does
  the same automatically. Target encoding: falls back to `y_train.mean()`.

## Files

```
day15_categorical_fe/
├── categorical_fe.py   # runnable script, all 7 exercises (0-6)
├── requirements.txt    # pinned: pandas, numpy, scikit-learn
└── README.md
```

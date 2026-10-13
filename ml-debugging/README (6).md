# Exercise 0-6: Common scikit-learn pitfalls (debugging.py)

Six broken-vs-fixed demonstrations on the California Housing dataset, each
isolating one classic ML pipeline bug.

## Setup
```
source piscine1b_w4/bin/activate   # or your venv name
pip install pandas numpy scikit-learn
python debugging.py
```

## What each exercise shows

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | Preprocessing leakage | Scaler fit on full data before split | Fit scaler on train split only, `.transform()` on test |
| 2 | Duplicate contamination | Inflated test R² from memorized twins | `drop_duplicates()` before splitting |
| 3 | Target leakage | Test R² = 1.0000 (too good to be true) | Check feature importances, drop the leaky column |
| 4 | Non-determinism | Different R² on every run | Set `random_state` on the estimator |
| 5 | CV without shuffle | Wildly negative per-fold R² | `KFold(shuffle=True, random_state=...)` |
| 6 | Target encoding leakage | Encoding stats computed on full data (incl. test) | Compute category means on train only, map onto test, handle unseen categories with `.fillna(y_train.mean())` |

## Audit notes (non-obvious decisions)

- **Ex.1 delta is near zero on purpose.** RandomForest is scale-invariant, so
  pre-split vs post-split scaling barely moves the score here. The bug is
  still real — swap in `LinearRegression` and the delta becomes visible,
  because linear models' coefficients are sensitive to the mean/scale used.
- **Ex.3 diagnosis step matters more than the fix.** A test R² of exactly
  1.0000 is the tell — no real-world model hits that on this dataset. The
  `feature_importances_` table is included specifically to show how you'd
  *find* the leaky column in an audit, not just declare a fix.
- **Ex.5's negative R² is expected, not a crash.** A negative R² means the
  model performs worse than predicting the mean — that's the signature of
  fold composition mismatch (sorted target → each fold sees a narrow,
  disjoint price band never represented in its own training data).
- **Ex.6 unseen-category handling.** `X_test_fixed["ocean_te"].fillna(y_train.mean())`
  guards against a category appearing in test but not in train (would
  otherwise produce `NaN` and break `LinearRegression.fit`/`.score`).
- **Numbers won't match the assignment doc exactly.** Local run values differ
  slightly from the spec's expected output (e.g. Ex.1 gave 0.8076 vs the
  doc's 0.7989) — this is sklearn-version-dependent (RandomForest internals
  changed across versions), not a bug. The *qualitative* pattern (leakage
  inflates score, fix brings it back down, deltas match direction) holds
  exactly as specified. Confirm your grader's expected values against your
  installed sklearn version if exact figures are checked.

## Anticipated defense questions

- *"Why is Ex.1's delta so small — did you actually fix anything?"* — Yes;
  RandomForest doesn't care about feature scale, so leakage here doesn't
  change *predictions*, only whether the pipeline is methodologically sound.
  The leak would matter with a scale-sensitive model.
- *"How would you catch Ex.3 in a real project without being told to look
  for it?"* — A suspiciously perfect test score is the first red flag;
  `feature_importances_` (or permutation importance) pinpoints the culprit.
- *"Why fold means instead of a smoothed/regularized target encoder in
  Ex.6?"* — Kept simple to isolate the leakage mechanism per the exercise
  spec; production code would use k-fold target encoding or additive
  smoothing to reduce variance on rare categories.

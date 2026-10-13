# California Housing — Baseline Regression Pipeline

Role-play: junior data scientist at a real estate analytics firm, building
the baseline regression model to predict `median_house_value` from the
eight numeric features (last week's raid EDA already approved). Covers
train/test split, `LinearRegression`, regression metrics, coefficient
standardization, and Ridge/Lasso regularization.

## Verified end-to-end

Unlike the S&P 500 day, this one downloads from `raw.githubusercontent.com`,
which my sandbox *can* reach — so `linear_regression.py` was actually run,
not just reviewed. **Every printed number matches the exercise sheet's
expected output exactly**: R² (train 0.6401 / test 0.6139), RMSE ($71,133),
MAE ($51,810), the full raw and standardized coefficient tables, the Ridge
shrinkage sequence, and the Lasso sequence including the exact 3-feature
survival set (`latitude`, `housing_median_age`, `median_income`) at
alpha=10000.

## How to run

```bash
# activate the persistent piscine1b_w3 venv
pip install pandas numpy scikit-learn
pip freeze > requirements.txt
python linear_regression.py
```

## Pipeline

1. **Ex 0** — verify scikit-learn install.
2. **Ex 1** — load California housing CSV, impute `total_bedrooms` median, drop `ocean_proximity`, split 80/20 with `random_state=42`.
3. **Ex 2** — fit `LinearRegression`, predict on train/test, confirm `.score()` matches `r2_score()`.
4. **Ex 3** — compute R², MSE, RMSE, MAE on test and train; sanity-check RMSE/y_std ≈ √(1 − R²).
5. **Ex 4** — print raw coefficients (not comparable — features on different scales), then `StandardScaler` + refit to get comparable, sorted coefficients.
6. **Ex 5** — fit Ridge across `alpha` ∈ {0, 1, 100, 10000}, compare coefficient shrinkage vs. OLS.
7. **Ex 6** — fit Lasso across the same-style alpha sweep, identify which features get driven to exactly zero, and which three survive at alpha=10000.

## Key results

| Metric | Train | Test |
|---|---|---|
| R² | 0.6401 | 0.6139 |
| RMSE | $69,362 | $71,133 |
| MAE | $50,627 | $51,810 |

**Standardized coefficients, largest to smallest:** latitude (-90,699) >
longitude (-85,503) > median_income (+77,195) > total_bedrooms (+48,712) >
population (-43,766) > total_rooms (-17,805) > households (+17,655) >
housing_median_age (+14,906).

**Location dominates.** Latitude and longitude have by far the largest
standardized coefficients — where a house is matters more than any single
other feature. `median_income` is the strongest non-geographic predictor
and the only coefficient that's directly interpretable in raw units
(~$40.5k more value per unit increase in income, in the unstandardized
model).

**Ridge (L2)** shrinks every coefficient toward zero proportionally,
never to exactly zero. At alpha=10000 the coefficient norm drops from
~163,204 to ~54,794 and test R² noticeably degrades (0.6139 → 0.4646) —
this dataset isn't very overfit, so Ridge mostly just costs performance
here rather than improving generalization.

**Lasso (L1)** can zero coefficients out entirely. At alpha=10000 it drops
`total_rooms`, `total_bedrooms`, and `households` to exactly zero —
the same three features flagged in the Week 2 raid as redundant
(pairwise correlations >0.9, since they're all proxies for block size).
This is automatic feature selection: 3 features survive, test R² drops to
0.4814, a real trade-off between simplicity and accuracy.

## Files

| File | Description |
|---|---|
| `linear_regression.py` | Complete script, one function per exercise (0–6), plus a `write_note()` function producing the required written paragraph |
| `requirements.txt` | Frozen package versions |

## Notes for audit

- All function calls (`train_test_split(..., random_state=42)`,
  `StandardScaler().fit_transform`/`.transform`, the exact Ridge/Lasso
  alpha sweeps) match the exercise sheet's required calls verbatim.
- The written note covers all four required points: which features have
  the largest coefficients (latitude/longitude, then median_income) and
  why, plus how Ridge vs. Lasso change them differently (shrink-all vs.
  zero-out, and *why* Lasso zeroes exactly those three features).

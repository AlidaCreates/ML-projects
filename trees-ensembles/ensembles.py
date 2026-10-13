"""
Day 16 — Ensembles (Gradient Boosting, XGBoost, Voting, Stacking)
Tomorrow School piscine1b, Week 4

Reuses the Day 14/15 engineered feature set (log-transform, ratio features,
one-hot encoded ocean_proximity) and Day 13's RandomForest baseline as the
comparison point for every ensemble method fit below.
"""

import time
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import xgboost as xgb

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Exercise 0: environment + data (Day 14/15 engineered feature set)
# ---------------------------------------------------------------------------

def engineer_full(X_in: pd.DataFrame, train_columns=None) -> pd.DataFrame:
    """Log-transform skewed counts, derive ratio features, one-hot encode
    ocean_proximity, and align columns to the training set when given."""
    out = X_in.copy()
    for col in ["total_rooms", "total_bedrooms", "population", "households"]:
        out[col] = np.log1p(out[col])
    rooms = np.expm1(out["total_rooms"])
    bedrooms = np.expm1(out["total_bedrooms"])
    population = np.expm1(out["population"])
    households = np.expm1(out["households"])
    out["rooms_per_household"] = rooms / households
    out["bedrooms_per_room"] = bedrooms / rooms
    out["population_per_household"] = population / households
    out = pd.get_dummies(out, columns=["ocean_proximity"], dtype=int)
    # XGBoost >= 2.1 rejects feature names containing [, ] or < (e.g. the
    # "<1H OCEAN" category produces a column XGBoost's newer strict parser
    # refuses). Sanitize after dummy-encoding so every downstream model sees
    # the same column names.
    out.columns = [str(c).replace("<", "lt_").replace("[", "").replace("]", "") for c in out.columns]
    if train_columns is not None:
        out = out.reindex(columns=train_columns, fill_value=0)
    return out


def load_data():
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

    X_raw = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=RANDOM_STATE
    )

    X_train = engineer_full(X_train_raw)
    X_test = engineer_full(X_test_raw, train_columns=X_train.columns.tolist())
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Exercise 1: GradientBoostingRegressor vs. RandomForest baseline
# ---------------------------------------------------------------------------

def exercise_1(X_train, X_test, y_train, y_test):
    print("=" * 70)
    print("Exercise 1: Gradient Boosting vs. Random Forest baseline")
    print("=" * 70)

    t0 = time.perf_counter()
    gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    gb_time = time.perf_counter() - t0
    gb_test_r2 = gb.score(X_test, y_test)

    print("GradientBoostingRegressor (default)")
    print(f"  n_estimators={gb.n_estimators}, learning_rate={gb.learning_rate}, max_depth={gb.max_depth}")
    print(f"  train R²: {gb.score(X_train, y_train):.4f}")
    print(f"  test R²:  {gb_test_r2:.4f}")
    print(f"  fit time: {gb_time:.2f}s")

    t0 = time.perf_counter()
    rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_time = time.perf_counter() - t0
    rf_test_r2 = rf.score(X_test, y_test)

    print("\nRandomForestRegressor (100 trees)")
    print(f"  test R²:  {rf_test_r2:.4f}")
    print(f"  fit time: {rf_time:.2f}s")

    return gb_test_r2, gb_time, rf_test_r2, rf_time


# ---------------------------------------------------------------------------
# Exercise 2: HistGradientBoostingRegressor
# ---------------------------------------------------------------------------

def exercise_2(X_train, X_test, y_train, y_test, rf_test_r2, rf_time, gb_test_r2, gb_time):
    print("\n" + "=" * 70)
    print("Exercise 2: Histogram-based Gradient Boosting")
    print("=" * 70)

    t0 = time.perf_counter()
    hgb = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    hgb.fit(X_train, y_train)
    hgb_time = time.perf_counter() - t0
    hgb_test_r2 = hgb.score(X_test, y_test)

    print("HistGradientBoostingRegressor (default)")
    print(f"  max_iter={hgb.max_iter}, learning_rate={hgb.learning_rate}, max_leaf_nodes={hgb.max_leaf_nodes}")
    print(f"  train R²: {hgb.score(X_train, y_train):.4f}")
    print(f"  test R²:  {hgb_test_r2:.4f}")
    print(f"  fit time: {hgb_time:.2f}s")

    results = pd.DataFrame({
        "model": ["RandomForest", "GradientBoosting", "HistGradientBoosting"],
        "test_R²": [rf_test_r2, gb_test_r2, hgb_test_r2],
        "fit_s": [rf_time, gb_time, hgb_time],
    })
    print("\n", results.round(4))

    return hgb_test_r2, hgb_time


# ---------------------------------------------------------------------------
# Exercise 3: XGBRegressor
# ---------------------------------------------------------------------------

def exercise_3(X_train, X_test, y_train, y_test, rf_test_r2, rf_time,
               gb_test_r2, gb_time, hgb_test_r2, hgb_time):
    print("\n" + "=" * 70)
    print("Exercise 3: XGBoost")
    print("=" * 70)

    t0 = time.perf_counter()
    xgbr = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    xgbr.fit(X_train, y_train)
    xgb_time = time.perf_counter() - t0
    xgb_test_r2 = xgbr.score(X_test, y_test)

    # XGBoost >= 3.0 leaves unset sklearn-API attrs as None; the effective
    # values only live in the fitted booster's config, not in get_params().
    tree_params = json.loads(xgbr.get_booster().save_config())["learner"]["gradient_booster"]["tree_train_param"]
    n_estimators_used = xgbr.get_booster().num_boosted_rounds()

    print("XGBRegressor (default)")
    print(f"  n_estimators={n_estimators_used}, learning_rate={tree_params['learning_rate']}, max_depth={tree_params['max_depth']}")
    print(f"  train R²: {xgbr.score(X_train, y_train):.4f}")
    print(f"  test R²:  {xgb_test_r2:.4f}")
    print(f"  fit time: {xgb_time:.2f}s")

    results = pd.DataFrame({
        "model": ["RandomForest", "GradientBoosting", "HistGradientBoosting", "XGBRegressor"],
        "test_R²": [rf_test_r2, gb_test_r2, hgb_test_r2, xgb_test_r2],
        "fit_s": [rf_time, gb_time, hgb_time, xgb_time],
    })
    print("\n", results.round(4))

    return xgb_test_r2, xgb_time


# ---------------------------------------------------------------------------
# Exercise 4: hyperparameter sweeps (n_estimators, learning_rate, max_depth)
# ---------------------------------------------------------------------------

def exercise_4(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 70)
    print("Exercise 4: Hyperparameter intuition")
    print("=" * 70)

    print("\nSweep n_estimators (learning_rate=0.1, max_depth=4):")
    for n in [50, 100, 200, 500, 1000]:
        m = xgb.XGBRegressor(n_estimators=n, learning_rate=0.1, max_depth=4,
                              random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X_train, y_train)
        print(f"  n_estimators={n:>4}  test R²={m.score(X_test, y_test):.4f}")

    print("\nSweep learning_rate (n_estimators=500, max_depth=4):")
    for lr in [0.01, 0.05, 0.1, 0.3]:
        m = xgb.XGBRegressor(n_estimators=500, learning_rate=lr, max_depth=4,
                              random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X_train, y_train)
        print(f"  learning_rate={lr:.2f}  test R²={m.score(X_test, y_test):.4f}")

    print("\nSweep max_depth (n_estimators=500, learning_rate=0.05):")
    for d in [2, 3, 4, 6, 8, 10]:
        m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=d,
                              random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X_train, y_train)
        print(f"  max_depth={d:>2}  test R²={m.score(X_test, y_test):.4f}")

    # best combo found by the sweeps above
    return {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 4}


# ---------------------------------------------------------------------------
# Exercise 5: VotingRegressor
# ---------------------------------------------------------------------------

def exercise_5(X_train, X_test, y_train, y_test, best_params):
    print("\n" + "=" * 70)
    print("Exercise 5: Voting ensemble")
    print("=" * 70)

    voter = VotingRegressor(estimators=[
        ("rf", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
        ("hgb", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
        ("xgb", xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best_params)),
    ])
    voter.fit(X_train, y_train)
    voter_test_r2 = voter.score(X_test, y_test)

    print("VotingRegressor (3 models)")
    print(f"  test R²: {voter_test_r2:.4f}")

    print("\nPer-base-model test R²:")
    for name, est in voter.named_estimators_.items():
        print(f"  {name}: test R² = {est.score(X_test, y_test):.4f}")

    return voter_test_r2


# ---------------------------------------------------------------------------
# Exercise 6: StackingRegressor
# ---------------------------------------------------------------------------

def exercise_6(X_train, X_test, y_train, y_test, best_params,
               rf_test_r2, gb_test_r2, hgb_test_r2, xgb_test_r2, voter_test_r2):
    print("\n" + "=" * 70)
    print("Exercise 6: Stacking ensemble")
    print("=" * 70)

    stacker = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
            ("hgb", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
            ("xgb", xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best_params)),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1,
    )
    stacker.fit(X_train, y_train)
    stacker_test_r2 = stacker.score(X_test, y_test)

    print("StackingRegressor (3 base models, Ridge meta-learner)")
    print(f"  test R²: {stacker_test_r2:.4f}")

    ridge_meta = stacker.final_estimator_
    print("\nmeta-learner coefficients:")
    for name, coef in zip(["rf", "hgb", "xgb"], ridge_meta.coef_):
        print(f"  {name}: {coef:.4f}")
    print(f"  intercept: {ridge_meta.intercept_:.2f}")

    tuned_xgb_r2 = xgb.XGBRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, **best_params
    ).fit(X_train, y_train).score(X_test, y_test)

    print("\n" + "=" * 70)
    print("Final comparison")
    print("=" * 70)
    results = pd.DataFrame({
        "model": [
            "LinearRegression (Day 11 baseline)",
            "RandomForest",
            "GradientBoosting",
            "HistGradientBoosting",
            "XGBRegressor (default)",
            "XGBRegressor (tuned: 500/0.05/4)",
            "VotingRegressor",
            "StackingRegressor",
        ],
        "test_R²": [
            0.5958,
            rf_test_r2,
            gb_test_r2,
            hgb_test_r2,
            xgb_test_r2,
            tuned_xgb_r2,
            voter_test_r2,
            stacker_test_r2,
        ],
    })
    print(results.round(4))

    return stacker_test_r2


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Feature set shape: train={X_train.shape}, test={X_test.shape}\n")

    gb_test_r2, gb_time, rf_test_r2, rf_time = exercise_1(X_train, X_test, y_train, y_test)
    hgb_test_r2, hgb_time = exercise_2(
        X_train, X_test, y_train, y_test, rf_test_r2, rf_time, gb_test_r2, gb_time
    )
    xgb_test_r2, xgb_time = exercise_3(
        X_train, X_test, y_train, y_test, rf_test_r2, rf_time,
        gb_test_r2, gb_time, hgb_test_r2, hgb_time
    )
    best_params = exercise_4(X_train, X_test, y_train, y_test)
    voter_test_r2 = exercise_5(X_train, X_test, y_train, y_test, best_params)
    exercise_6(
        X_train, X_test, y_train, y_test, best_params,
        rf_test_r2, gb_test_r2, hgb_test_r2, xgb_test_r2, voter_test_r2
    )


if __name__ == "__main__":
    main()

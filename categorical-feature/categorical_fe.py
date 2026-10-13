"""
Day 15 — Categorical feature engineering (ocean_proximity)
California Housing dataset.

Pipeline:
  Ex0  environment + baselines (Day 11 raw, Day 14 numeric-engineered)
  Ex1  one-hot encoding (pd.get_dummies)
  Ex2  one-hot encoding (sklearn OneHotEncoder) — sanity check vs Ex1
  Ex3  ordinal encoding (explicit distance-from-ocean order)
  Ex4  target encoding (train-only means, + leakage demo)
  Ex5  combined pipeline: numeric engineering + one-hot -> engineer_full()
  Ex6  final comparison grid: 5 feature sets x {LinearRegression, RandomForest}

Run:
    python categorical_fe.py
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score  # noqa: F401  (imported per spec; .score() used instead)

pd.set_option("display.width", 120)

URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"


# --------------------------------------------------------------------------- #
# Ex0 — data + numeric feature engineering (reused from Day 14)
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    df = pd.read_csv(URL)
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
    return df


def engineer_numeric(X_in: pd.DataFrame) -> pd.DataFrame:
    """Log-transform skewed counts, then derive ratio features from the
    *un*-logged values (expm1 undoes the log1p above) — same as Day 14."""
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
    return out


def engineer_full(X_in: pd.DataFrame, train_columns: list[str] | None = None) -> pd.DataFrame:
    """Ex5: numeric engineering + one-hot encoding of ocean_proximity.

    train_columns, when given, reindexes the result onto the training
    columns (fill_value=0) so the test set can never end up with a
    different / missing dummy column than train.
    """
    out = engineer_numeric(X_in)
    out = pd.get_dummies(out, columns=["ocean_proximity"], dtype=int)
    if train_columns is not None:
        out = out.reindex(columns=train_columns, fill_value=0)
    return out


def fit_score(X_tr, X_te, y_train, y_test, model_cls=LinearRegression):
    m = model_cls()
    m.fit(X_tr, y_train)
    return m.score(X_te, y_test)


def rf_score(X_tr, X_te, y_train, y_test):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_train)
    return rf.score(X_te, y_test)


def main() -> None:
    # ----------------------------------------------------------------- #
    # Ex0
    # ----------------------------------------------------------------- #
    print("=" * 70)
    print("Exercise 0: environment and baseline")
    print("=" * 70)

    df = load_data()
    print(df["ocean_proximity"].value_counts())

    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline_r2 = fit_score(
        X_train.drop(columns=["ocean_proximity"]),
        X_test.drop(columns=["ocean_proximity"]),
        y_train, y_test,
    )

    X_train_num = engineer_numeric(X_train.drop(columns=["ocean_proximity"]))
    X_test_num = engineer_numeric(X_test.drop(columns=["ocean_proximity"]))
    numeric_r2 = fit_score(X_train_num, X_test_num, y_train, y_test)

    print(f"\nDay 11 baseline (raw, no ocean):           {baseline_r2:.4f}")
    print(f"Day 14 numeric-engineered (no ocean):      {numeric_r2:.4f}")

    # ----------------------------------------------------------------- #
    # Ex1 — pd.get_dummies
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Exercise 1: one-hot encoding with pandas")
    print("=" * 70)

    X_train_eng = engineer_numeric(X_train)
    X_test_eng = engineer_numeric(X_test)

    X_train_oh = pd.get_dummies(X_train_eng, columns=["ocean_proximity"], dtype=int)
    X_test_oh = pd.get_dummies(X_test_eng, columns=["ocean_proximity"], dtype=int)
    # Guards against a category present in train but absent from test
    # (or vice versa) silently producing mismatched columns.
    X_test_oh = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0)

    print(f"engineered + one-hot column count: {X_train_oh.shape[1]}")
    print(
        "new columns: "
        f"{[c for c in X_train_oh.columns if c.startswith('ocean_proximity_')]}"
    )

    oh_r2 = fit_score(X_train_oh, X_test_oh, y_train, y_test)
    print(f"\nnumeric-only:        {numeric_r2:.4f}")
    print(f"+ one-hot ocean:     {oh_r2:.4f}")
    print(f"delta:               {oh_r2 - numeric_r2:+.4f}")
    print("\nDecision: KEEP one-hot encoding as the new working baseline.")

    # ----------------------------------------------------------------- #
    # Ex2 — sklearn OneHotEncoder
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Exercise 2: one-hot encoding with sklearn")
    print("=" * 70)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    ocean_train = X_train_eng[["ocean_proximity"]]
    ocean_test = X_test_eng[["ocean_proximity"]]

    ocean_train_oh = ohe.fit_transform(ocean_train)
    ocean_test_oh = ohe.transform(ocean_test)

    print(f"OHE categories: {ohe.categories_[0].tolist()}")
    print(f"shape: {ocean_train_oh.shape}")

    X_train_ohe = np.hstack(
        [X_train_eng.drop(columns=["ocean_proximity"]).values, ocean_train_oh]
    )
    X_test_ohe = np.hstack(
        [X_test_eng.drop(columns=["ocean_proximity"]).values, ocean_test_oh]
    )
    print(f"shape: {X_train_ohe.shape}")

    ohe_r2 = fit_score(X_train_ohe, X_test_ohe, y_train, y_test)
    print(f"\npd.get_dummies test R²:      {oh_r2:.4f}")
    print(f"OneHotEncoder test R²:       {ohe_r2:.4f}")
    print(f"agree: {np.isclose(oh_r2, ohe_r2)}")

    # ----------------------------------------------------------------- #
    # Ex3 — OrdinalEncoder
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Exercise 3: ordinal encoding")
    print("=" * 70)

    order = [["INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BAY", "ISLAND"]]
    ord_enc = OrdinalEncoder(categories=order)

    ocean_train_ord = ord_enc.fit_transform(ocean_train).astype(int)
    ocean_test_ord = ord_enc.transform(ocean_test).astype(int)

    print(f"ord_enc.categories_: {ord_enc.categories_}")
    print(f"first 10 train values: {ocean_train_ord[:10].ravel().tolist()}")

    X_train_ord_arr = np.hstack(
        [X_train_eng.drop(columns=["ocean_proximity"]).values, ocean_train_ord]
    )
    X_test_ord_arr = np.hstack(
        [X_test_eng.drop(columns=["ocean_proximity"]).values, ocean_test_ord]
    )

    ord_r2 = fit_score(X_train_ord_arr, X_test_ord_arr, y_train, y_test)
    print(f"\n+ one-hot:           {oh_r2:.4f}")
    print(f"+ ordinal:           {ord_r2:.4f}")
    print(f"delta vs one-hot:    {ord_r2 - oh_r2:+.4f}")
    print(
        "\nDecision: DO NOT KEEP ordinal encoding — the assumed linear order "
        "doesn't match the true price relationship (ISLAND is priciest but "
        "sits at the top of the scale, which one coefficient can't capture "
        "without distorting the rest)."
    )

    # ----------------------------------------------------------------- #
    # Ex4 — target encoding (+ leakage demo)
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Exercise 4: target encoding")
    print("=" * 70)

    train_with_y = X_train_eng.copy()
    train_with_y["__target__"] = y_train.values

    cat_means = train_with_y.groupby("ocean_proximity", observed=True)["__target__"].mean()
    print(cat_means.round(2).sort_values())

    X_train_te = X_train_eng.copy()
    X_test_te = X_test_eng.copy()

    X_train_te["ocean_te"] = X_train_te["ocean_proximity"].map(cat_means).astype(float)
    X_test_te["ocean_te"] = X_test_te["ocean_proximity"].map(cat_means).astype(float)
    # A category unseen in training would map to NaN — fall back to the
    # train's global mean rather than leaking test-set information.
    X_test_te["ocean_te"] = X_test_te["ocean_te"].fillna(y_train.mean())

    X_train_te = X_train_te.drop(columns=["ocean_proximity"])
    X_test_te = X_test_te.drop(columns=["ocean_proximity"])

    print(f"\n{X_train_te.shape} {X_test_te.shape}")

    te_r2 = fit_score(X_train_te, X_test_te, y_train, y_test)
    print(f"\n+ one-hot:           {oh_r2:.4f}")
    print(f"+ target encoding:   {te_r2:.4f}")
    print(f"delta vs one-hot:    {te_r2 - oh_r2:+.4f}")

    # Leakage demo: means computed on train+test combined
    df_all = X.copy()
    df_all["__target__"] = y.values
    leaky_means = df_all.groupby("ocean_proximity", observed=True)["__target__"].mean()

    X_train_leak = X_train_eng.copy()
    X_test_leak = X_test_eng.copy()
    X_train_leak["ocean_te_leaky"] = (
        X_train_leak["ocean_proximity"].map(leaky_means).astype(float)
    )
    X_test_leak["ocean_te_leaky"] = (
        X_test_leak["ocean_proximity"].map(leaky_means).astype(float)
    )
    X_train_leak = X_train_leak.drop(columns=["ocean_proximity"])
    X_test_leak = X_test_leak.drop(columns=["ocean_proximity"])

    leak_r2 = fit_score(X_train_leak, X_test_leak, y_train, y_test)
    print(f"\nclean target encoding:   {te_r2:.4f}")
    print(f"leaky target encoding:   {leak_r2:.4f}")
    print(
        "\nNote: the leak is tiny here because per-category means are stable "
        "across the split. On high-cardinality categoricals or small samples "
        "it can inflate test R² by 0.02+ — always fit means on train only."
    )

    # ----------------------------------------------------------------- #
    # Ex5 — combined pipeline
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Exercise 5: combined pipeline")
    print("=" * 70)

    X_train_full = engineer_full(X_train)
    X_test_full = engineer_full(X_test, train_columns=X_train_full.columns.tolist())

    print(f"{X_train_full.shape} {X_test_full.shape}")

    final_r2 = fit_score(X_train_full, X_test_full, y_train, y_test)
    print(f"\nDay 11 baseline (raw, no ocean):                  {baseline_r2:.4f}")
    print(f"Day 14 numeric-engineered (no ocean):             {numeric_r2:.4f}")
    print(f"Day 15 numeric + categorical (one-hot):           {final_r2:.4f}")
    print(f"total improvement vs Day 11 baseline:             {final_r2 - baseline_r2:+.4f}")

    # ----------------------------------------------------------------- #
    # Ex6 — final comparison grid
    # ----------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("Exercise 6: final comparison")
    print("=" * 70)

    X_train_raw = X_train.drop(columns=["ocean_proximity"])
    X_test_raw = X_test.drop(columns=["ocean_proximity"])

    X_train_oh_full = engineer_full(X_train)
    X_test_oh_full = engineer_full(X_test, train_columns=X_train_oh_full.columns.tolist())

    X_train_ord_df = X_train_eng.drop(columns=["ocean_proximity"]).copy()
    X_train_ord_df["ocean_ord"] = ord_enc.fit_transform(
        X_train_eng[["ocean_proximity"]]
    ).astype(int)
    X_test_ord_df = X_test_eng.drop(columns=["ocean_proximity"]).copy()
    X_test_ord_df["ocean_ord"] = ord_enc.transform(
        X_test_eng[["ocean_proximity"]]
    ).astype(int)

    X_train_te_only = X_train_te.copy()
    X_test_te_only = X_test_te.copy()

    results = pd.DataFrame(
        {
            "feature_set": [
                "raw (no ocean)",
                "numeric-engineered (no ocean)",
                "+ one-hot ocean",
                "+ ordinal ocean",
                "+ target-encoded ocean",
            ],
            "linreg": [
                fit_score(X_train_raw, X_test_raw, y_train, y_test),
                numeric_r2,
                fit_score(X_train_oh_full, X_test_oh_full, y_train, y_test),
                fit_score(X_train_ord_df, X_test_ord_df, y_train, y_test),
                fit_score(X_train_te_only, X_test_te_only, y_train, y_test),
            ],
            "random_forest": [
                rf_score(X_train_raw, X_test_raw, y_train, y_test),
                rf_score(X_train_num, X_test_num, y_train, y_test),
                rf_score(X_train_oh_full, X_test_oh_full, y_train, y_test),
                rf_score(X_train_ord_df, X_test_ord_df, y_train, y_test),
                rf_score(X_train_te_only, X_test_te_only, y_train, y_test),
            ],
        }
    )
    print(results.round(4))

    print()
    log = pd.DataFrame(
        {
            "step": ["Day 14 numeric", "+ one-hot", "+ ordinal", "+ target encoding"],
            "linreg_test_R2": [numeric_r2, oh_r2, ord_r2, te_r2],
            "kept": [
                "yes",
                "yes (chosen)",
                "no (worse than one-hot)",
                "no (essentially tied with one-hot, more code)",
            ],
        }
    )
    print(log.round(4))


if __name__ == "__main__":
    main()


"""
Day XX - Numeric Feature Engineering (California Housing)
Reuses the piscine1b_w3 virtual environment from Day 11.

Run:
    python numeric_fe.py
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, KBinsDiscretizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------------------------------
# Exercise 0: Environment and baseline
# --------------------------------------------------------------------------

URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(URL)
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
df = df.drop(columns=["ocean_proximity"])

X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def fit_score(X_tr, X_te, y_tr=y_train, y_te=y_test, model_cls=LinearRegression):
    """Fit a fresh model and return test R2."""
    m = model_cls()
    m.fit(X_tr, y_tr)
    return m.score(X_te, y_te)


def fit_score_model(X_tr, X_te, model, y_tr=y_train, y_te=y_test):
    """Fit a given (already-instantiated) model and return test R2."""
    model.fit(X_tr, y_tr)
    return model.score(X_te, y_te)


baseline_r2 = fit_score(X_train, X_test)
print("=== Exercise 0: baseline ===")
print(f"baseline test R2 (LinearRegression, raw features): {baseline_r2:.4f}\n")


# --------------------------------------------------------------------------
# Exercise 1: Log transformations
# --------------------------------------------------------------------------

log_cols = ["total_rooms", "total_bedrooms", "population", "households"]

X_train_log = X_train.copy()
X_test_log = X_test.copy()
for col in log_cols:
    X_train_log[col] = np.log1p(X_train_log[col])
    X_test_log[col] = np.log1p(X_test_log[col])

log_r2 = fit_score(X_train_log, X_test_log)

print("=== Exercise 1: log transformations ===")
print(f"baseline:     {baseline_r2:.4f}")
print(f"log-features: {log_r2:.4f}")
print(f"delta:        {log_r2 - baseline_r2:+.4f}\n")

print("skewness before log:")
print(X_train[log_cols].skew().round(4))
print("\nskewness after log:")
print(X_train_log[log_cols].skew().round(4))
print()

log_decision = "KEEP" if log_r2 > baseline_r2 else "DO NOT KEEP"
print(f"Decision: {log_decision} the log transformations. "
      f"Test R2 changed by {log_r2 - baseline_r2:+.4f}.\n")


# --------------------------------------------------------------------------
# Exercise 2: Ratio features
# --------------------------------------------------------------------------

def add_ratios(X_in: pd.DataFrame) -> pd.DataFrame:
    out = X_in.copy()
    # Reconstruct the original (non-logged) columns via expm1 for the ratios.
    rooms = np.expm1(X_in["total_rooms"])
    bedrooms = np.expm1(X_in["total_bedrooms"])
    population = np.expm1(X_in["population"])
    households = np.expm1(X_in["households"])
    out["rooms_per_household"] = rooms / households
    out["bedrooms_per_room"] = bedrooms / rooms
    out["population_per_household"] = population / households
    return out


X_train_ratio = add_ratios(X_train_log)
X_test_ratio = add_ratios(X_test_log)

print("=== Exercise 2: ratio features ===")
print(X_train_ratio.shape)
print(X_train_ratio.columns.tolist())

ratio_r2 = fit_score(X_train_ratio, X_test_ratio)
print(f"baseline:      {baseline_r2:.4f}")
print(f"log-features:  {log_r2:.4f}")
print(f"+ ratios:      {ratio_r2:.4f}")
print(f"delta vs base: {ratio_r2 - baseline_r2:+.4f}")
print(f"delta vs log:  {ratio_r2 - log_r2:+.4f}\n")

ratio_decision = "KEEP" if ratio_r2 > log_r2 else "DO NOT KEEP"
print(f"Decision: {ratio_decision} the ratio features. "
      f"Cumulative change vs baseline is {ratio_r2 - baseline_r2:+.4f}.\n")


# --------------------------------------------------------------------------
# Exercise 3: Polynomial features
# --------------------------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_ratio)
X_test_scaled = scaler.transform(X_test_ratio)

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print("=== Exercise 3: polynomial features ===")
print(f"input features:  {X_train_scaled.shape[1]}")
print(f"output features: {X_train_poly.shape[1]}")

poly_r2 = fit_score(X_train_poly, X_test_poly)
print(f"+ ratios:  {ratio_r2:.4f}")
print(f"+ poly(2): {poly_r2:.4f}")
print(f"delta:     {poly_r2 - ratio_r2:+.4f}\n")

ridge_r2 = fit_score_model(X_train_poly, X_test_poly, Ridge(alpha=1.0))
print(f"+ poly(2) [LinReg]: {poly_r2:.4f}")
print(f"+ poly(2) [Ridge]:  {ridge_r2:.4f}\n")

poly_decision = "KEEP (with Ridge)" if ridge_r2 > ratio_r2 else "DO NOT KEEP"
print(f"Decision: {poly_decision}. Plain LinearRegression score is {poly_r2:.4f}, "
      f"Ridge score is {ridge_r2:.4f}, vs ratio baseline {ratio_r2:.4f}.\n")


# --------------------------------------------------------------------------
# Exercise 4: Binning
# --------------------------------------------------------------------------

ages_train = X_train_ratio[["housing_median_age"]].values
ages_test = X_test_ratio[["housing_median_age"]].values

binner = KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile")
ages_train_bin = binner.fit_transform(ages_train)
ages_test_bin = binner.transform(ages_test)

print("=== Exercise 4: binning ===")
print(f"binned ages shape: {ages_train_bin.shape}")
print(f"bin edges: {binner.bin_edges_[0].round(2)}")

X_train_binned = np.hstack([
    X_train_ratio.drop(columns=["housing_median_age"]).values,
    ages_train_bin,
])
X_test_binned = np.hstack([
    X_test_ratio.drop(columns=["housing_median_age"]).values,
    ages_test_bin,
])
print(f"binned feature matrix shape: {X_train_binned.shape}")

binned_r2 = fit_score(X_train_binned, X_test_binned)
print(f"+ ratios (no bin):  {ratio_r2:.4f}")
print(f"+ ratios + age bin: {binned_r2:.4f}")
print(f"delta:              {binned_r2 - ratio_r2:+.4f}\n")

bin_decision = "KEEP" if (binned_r2 - ratio_r2) > 0.001 else "DO NOT KEEP"
print(f"Decision: {bin_decision} the binning. Marginal gain is {binned_r2 - ratio_r2:+.4f}.\n")


# --------------------------------------------------------------------------
# Exercise 5: Linear vs forest comparison
# --------------------------------------------------------------------------

def rf_score(X_tr, X_te):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_train)
    return rf.score(X_te, y_test)


rf_baseline = rf_score(X_train, X_test)
rf_log = rf_score(X_train_log, X_test_log)
rf_ratio = rf_score(X_train_ratio, X_test_ratio)
rf_poly = rf_score(X_train_poly, X_test_poly)

results = pd.DataFrame({
    "feature_set": ["raw", "+ log", "+ ratios", "+ poly(2)"],
    "linreg_test_R2": [baseline_r2, log_r2, ratio_r2, poly_r2],
    "rf_test_R2": [rf_baseline, rf_log, rf_ratio, rf_poly],
})
print("=== Exercise 5: linear vs forest ===")
print(results.round(4))
print()
print(f"LinReg gain (raw -> ratios): {ratio_r2 - baseline_r2:+.4f}")
print(f"RF gain (raw -> ratios):     {rf_ratio - rf_baseline:+.4f}\n")


# --------------------------------------------------------------------------
# Exercise 6: Combined engineered baseline
# --------------------------------------------------------------------------

def engineer_features(X_in: pd.DataFrame) -> pd.DataFrame:
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


X_train_eng = engineer_features(X_train)
X_test_eng = engineer_features(X_test)

linreg_eng_r2 = fit_score(X_train_eng, X_test_eng)
rf_eng_r2 = rf_score(X_train_eng, X_test_eng)

print("=== Exercise 6: combined engineered baseline ===")
print(f"LinearRegression baseline:   {baseline_r2:.4f}")
print(f"LinearRegression engineered: {linreg_eng_r2:.4f}  (delta {linreg_eng_r2 - baseline_r2:+.4f})")
print(f"RandomForest baseline:       {rf_baseline:.4f}")
print(f"RandomForest engineered:     {rf_eng_r2:.4f}  (delta {rf_eng_r2 - rf_baseline:+.4f})\n")

log_table = pd.DataFrame({
    "step": ["baseline", "+ log", "+ ratios", "+ poly(2)", "+ age bin"],
    "test_R2": [baseline_r2, log_r2, ratio_r2, poly_r2, binned_r2],
    "kept": [
        "-",
        log_decision.lower(),
        ratio_decision.lower(),
        poly_decision.lower(),
        bin_decision.lower(),
    ],
})
print("Experiment log:")
print(log_table.round(4))

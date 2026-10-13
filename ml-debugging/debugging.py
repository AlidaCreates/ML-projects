"""
debugging.py — Exercise 0-6: Common scikit-learn pitfalls (broken vs fixed)

Each exercise demonstrates a realistic bug (data leakage, non-determinism,
bad CV splitting) by running a BROKEN version, showing why it's wrong, then
a FIXED version, and reporting the delta.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42

URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(URL)
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
df = df.drop(columns=["ocean_proximity"])

X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

summary_rows = {}  # exercise label -> (broken, fixed)


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Exercise 1: Preprocessing leakage (scale before split vs scale after split)
# ---------------------------------------------------------------------------
header("Exercise 1: Preprocessing leakage")

# BROKEN — scaler fit on the full dataset before splitting
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X)

X_train_b, X_test_b, y_train, y_test = train_test_split(
    X_full_scaled, y, test_size=0.2, random_state=RANDOM_STATE
)

model = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_train_b, y_train)
broken_r2 = model.score(X_test_b, y_test)
print(f"BROKEN test R²:  {broken_r2:.4f}")

# FIXED — fit the scaler on the train split only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler_fixed = StandardScaler()
X_train_std = scaler_fixed.fit_transform(X_train)
X_test_std = scaler_fixed.transform(X_test)

model_fixed = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
model_fixed.fit(X_train_std, y_train)
fixed_r2 = model_fixed.score(X_test_std, y_test)
print(f"FIXED test R²:   {fixed_r2:.4f}")
print(f"delta:           {broken_r2 - fixed_r2:+.4f}")
summary_rows["1: pre-split scaling"] = (broken_r2, fixed_r2)


# ---------------------------------------------------------------------------
# Exercise 2: Train/test contamination via duplicates
# ---------------------------------------------------------------------------
header("Exercise 2: Train/test contamination via duplicates")

# BROKEN — duplicates survive the split
df_dup = pd.concat([df, df.head(1000)], ignore_index=True)
print(f"original rows: {len(df)}")
print(f"with duplicates: {len(df_dup)}")
print(f"actual duplicates: {df_dup.duplicated().sum()}")

X_dup = df_dup.drop(columns=["median_house_value"])
y_dup = df_dup["median_house_value"]

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_dup, y_dup, test_size=0.2, random_state=RANDOM_STATE
)

model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_train_b, y_train_b)
broken_r2 = model.score(X_test_b, y_test_b)
print(f"BROKEN test R²:  {broken_r2:.4f}")

# FIXED — deduplicate before splitting
df_clean = df_dup.drop_duplicates().reset_index(drop=True)
print(f"after dedup: {len(df_clean)}")
print(f"remaining duplicates: {df_clean.duplicated().sum()}")

X_clean = df_clean.drop(columns=["median_house_value"])
y_clean = df_clean["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=RANDOM_STATE
)

model_fixed = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
model_fixed.fit(X_train, y_train)
fixed_r2 = model_fixed.score(X_test, y_test)
print(f"FIXED test R²:   {fixed_r2:.4f}")
print(f"delta:           {broken_r2 - fixed_r2:+.4f}")
summary_rows["2: duplicate contamination"] = (broken_r2, fixed_r2)


# ---------------------------------------------------------------------------
# Exercise 3: Target leakage
# ---------------------------------------------------------------------------
header("Exercise 3: Target leakage")

# BROKEN — engineered feature derived directly from the target
df_leaky = df.copy()
df_leaky["value_indicator"] = df_leaky["median_house_value"] / 1000.0  # leaks the target!

X_leaky = df_leaky.drop(columns=["median_house_value"])
y_leaky = df_leaky["median_house_value"]

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_leaky, y_leaky, test_size=0.2, random_state=RANDOM_STATE
)

model = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_train_b, y_train_b)
broken_r2 = model.score(X_test_b, y_test_b)
print(f"BROKEN test R²:  {broken_r2:.4f}")

# Diagnose: feature importances expose the leaky column
imp = pd.DataFrame({
    "feature": X_leaky.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
print(imp.round(4).head())

# FIXED — drop the leaky feature and refit
df_fixed = df_leaky.drop(columns=["value_indicator"])

X_fixed = df_fixed.drop(columns=["median_house_value"])
y_fixed = df_fixed["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X_fixed, y_fixed, test_size=0.2, random_state=RANDOM_STATE
)

model_fixed = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
model_fixed.fit(X_train, y_train)
fixed_r2 = model_fixed.score(X_test, y_test)
print(f"FIXED test R²:   {fixed_r2:.4f}")
print(f"delta:           {broken_r2 - fixed_r2:+.4f}")
summary_rows["3: target leakage"] = (broken_r2, fixed_r2)


# ---------------------------------------------------------------------------
# Exercise 4: Non-deterministic results
# ---------------------------------------------------------------------------
header("Exercise 4: Non-deterministic results")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# BROKEN — no random_state on the estimator
scores = []
for i in range(3):
    m = RandomForestRegressor(n_estimators=50, n_jobs=-1)  # no random_state
    m.fit(X_train, y_train)
    scores.append(m.score(X_test, y_test))

print(f"BROKEN run 1 test R²: {scores[0]:.4f}")
print(f"BROKEN run 2 test R²: {scores[1]:.4f}")
print(f"BROKEN run 3 test R²: {scores[2]:.4f}")
print(f"std across runs:      {np.std(scores):.4f}")

# FIXED — set random_state
scores_fixed = []
for i in range(3):
    m = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(X_train, y_train)
    scores_fixed.append(m.score(X_test, y_test))

print(f"FIXED run 1 test R²: {scores_fixed[0]:.4f}")
print(f"FIXED run 2 test R²: {scores_fixed[1]:.4f}")
print(f"FIXED run 3 test R²: {scores_fixed[2]:.4f}")
print(f"std across runs:     {np.std(scores_fixed):.4f}")
summary_rows["4: missing random_state"] = ("varies", scores_fixed[0])


# ---------------------------------------------------------------------------
# Exercise 5: Cross-validation without shuffle
# ---------------------------------------------------------------------------
header("Exercise 5: Cross-validation without shuffle")

# BROKEN — sorted data + KFold without shuffle
df_sorted = df.sort_values("median_house_value").reset_index(drop=True)
X_sorted = df_sorted.drop(columns=["median_house_value"])
y_sorted = df_sorted["median_house_value"]

cv_no_shuffle = KFold(n_splits=5, shuffle=False)

model = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
scores_b = cross_val_score(model, X_sorted, y_sorted, cv=cv_no_shuffle, scoring="r2", n_jobs=-1)
print(f"BROKEN per-fold R²: {scores_b.round(4)}")
print(f"BROKEN mean R²:     {scores_b.mean():.4f}")

# FIXED — KFold with shuffle and a random_state
cv_shuffled = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

model_fixed = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
scores_f = cross_val_score(model_fixed, X_sorted, y_sorted, cv=cv_shuffled, scoring="r2", n_jobs=-1)
print(f"FIXED per-fold R²:  {scores_f.round(4)}")
print(f"FIXED mean R²:      {scores_f.mean():.4f}")
print(f"delta on mean:      {scores_f.mean() - scores_b.mean():+.4f}")
summary_rows["5: CV without shuffle"] = (scores_b.mean(), scores_f.mean())


# ---------------------------------------------------------------------------
# Exercise 6: Target encoding leakage
# ---------------------------------------------------------------------------
header("Exercise 6: Target encoding leakage")

# BROKEN — target encoding computed on the FULL dataset (incl. test rows)
df_cat = pd.read_csv(URL)
df_cat["total_bedrooms"] = df_cat["total_bedrooms"].fillna(df_cat["total_bedrooms"].median())

full_means = df_cat.groupby("ocean_proximity")["median_house_value"].mean()
df_cat["ocean_te"] = df_cat["ocean_proximity"].map(full_means)
df_cat = df_cat.drop(columns=["ocean_proximity"])

X_cat = df_cat.drop(columns=["median_house_value"])
y_cat = df_cat["median_house_value"]

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_cat, y_cat, test_size=0.2, random_state=RANDOM_STATE
)

model = LinearRegression()
model.fit(X_train_b, y_train_b)
broken_r2 = model.score(X_test_b, y_test_b)
print(f"BROKEN test R²:  {broken_r2:.4f}")

# FIXED — compute means on TRAIN ONLY, map onto both splits
df_cat = pd.read_csv(URL)
df_cat["total_bedrooms"] = df_cat["total_bedrooms"].fillna(df_cat["total_bedrooms"].median())

X_raw = df_cat.drop(columns=["median_house_value"])
y_raw = df_cat["median_house_value"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=RANDOM_STATE
)

train_with_y = X_train_raw.copy()
train_with_y["__target__"] = y_train.values
train_means = train_with_y.groupby("ocean_proximity")["__target__"].mean()

X_train_fixed = X_train_raw.copy()
X_train_fixed["ocean_te"] = X_train_fixed["ocean_proximity"].map(train_means).astype(float)
X_train_fixed = X_train_fixed.drop(columns=["ocean_proximity"])

X_test_fixed = X_test_raw.copy()
X_test_fixed["ocean_te"] = X_test_fixed["ocean_proximity"].map(train_means).astype(float)
X_test_fixed["ocean_te"] = X_test_fixed["ocean_te"].fillna(y_train.mean())  # unseen-category safety
X_test_fixed = X_test_fixed.drop(columns=["ocean_proximity"])

model_fixed = LinearRegression()
model_fixed.fit(X_train_fixed, y_train)
fixed_r2 = model_fixed.score(X_test_fixed, y_test)
print(f"FIXED test R²:   {fixed_r2:.4f}")
print(f"delta:           {broken_r2 - fixed_r2:+.4f}")
summary_rows["6: target encoding leakage"] = (broken_r2, fixed_r2)


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
header("Final summary")

diagnoses = {
    "1: pre-split scaling": "small in this case (RF is scale-invariant)",
    "2: duplicate contamination": "test set contains memorized training rows",
    "3: target leakage": "feature contains the target",
    "4: missing random_state": "stochastic model with no seed",
    "5: CV without shuffle": "fold composition mismatched between train and test",
    "6: target encoding leakage": "encoding statistics leaked test labels",
}

summary = pd.DataFrame({
    "exercise": list(summary_rows.keys()),
    "broken_score": [v[0] for v in summary_rows.values()],
    "fixed_score": [v[1] for v in summary_rows.values()],
    "diagnosis": [diagnoses[k] for k in summary_rows.keys()],
})
print(summary.to_string(index=False))

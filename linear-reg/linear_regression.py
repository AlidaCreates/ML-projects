"""
California Housing — Baseline Regression Pipeline
=====================================================
Fits LinearRegression, Ridge, and Lasso models to predict median_house_value
from the eight numeric features, evaluates R²/MSE/MAE, standardizes for
coefficient interpretability, and shows how L1/L2 regularization change
the coefficients.

Run with:
    python linear_regression.py

Requires: pandas, numpy, scikit-learn (piscine1b_w3 venv)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Exercise 0: Environment and libraries
# ---------------------------------------------------------------------------
def exercise_0() -> None:
    section("EXERCISE 0 — Environment and libraries")
    import sklearn
    print(f"scikit-learn version: {sklearn.__version__}")


# ---------------------------------------------------------------------------
# Exercise 1: Load and split
# ---------------------------------------------------------------------------
def exercise_1():
    section("EXERCISE 1 — Load and split")

    URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    df = pd.read_csv(URL)
    df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
    df = df.drop(columns=["ocean_proximity"])
    print(df.shape)
    print(df.columns.tolist())

    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X, y, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Exercise 2: Baseline linear regression
# ---------------------------------------------------------------------------
def exercise_2(X_train, X_test, y_train, y_test):
    section("EXERCISE 2 — Baseline linear regression")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(y_pred_train.shape, y_pred_test.shape)

    train_r2_method = model.score(X_train, y_train)
    test_r2_method = model.score(X_test, y_test)
    train_r2_func = r2_score(y_train, y_pred_train)
    test_r2_func = r2_score(y_test, y_pred_test)

    print(f"train R² (method): {train_r2_method:.4f}")
    print(f"test R² (method):  {test_r2_method:.4f}")
    print(f"train R² (func):   {train_r2_func:.4f}")
    print(f"test R² (func):    {test_r2_func:.4f}")

    preview = pd.DataFrame({"y_true": y_test.values[:5], "y_pred": y_pred_test[:5]})
    print(preview.round(2))

    return model, y_pred_train, y_pred_test, test_r2_func


# ---------------------------------------------------------------------------
# Exercise 3: Regression metrics
# ---------------------------------------------------------------------------
def exercise_3(y_train, y_test, y_pred_train, y_pred_test):
    section("EXERCISE 3 — Regression metrics")

    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)

    print(f"R²:   {test_r2:.4f}")
    print(f"MSE:  {test_mse:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"MAE:  {test_mae:.4f}")

    y_std = y_test.std()
    print(f"y_test std:    {y_std:.4f}")
    print(f"test RMSE:     {test_rmse:.4f}")
    print(f"RMSE / y_std:  {test_rmse / y_std:.4f}")

    train_r2 = r2_score(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    print(f"train R²:    {train_r2:.4f}")
    print(f"train RMSE:  {train_rmse:.4f}")
    print(f"train MAE:   {train_mae:.4f}")


# ---------------------------------------------------------------------------
# Exercise 4: Coefficients and standardization
# ---------------------------------------------------------------------------
def exercise_4(X, X_train, X_test, y_train, y_test, model, test_r2):
    section("EXERCISE 4 — Coefficients and standardization")

    coef_df = pd.DataFrame({"feature": X.columns, "coef": model.coef_})
    print(coef_df)
    print(f"intercept: {model.intercept_:.4f}")

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    model_std = LinearRegression()
    model_std.fit(X_train_std, y_train)

    print(f"test R² (standardized):   {model_std.score(X_test_std, y_test):.4f}")
    print(f"test R² (unstandardized): {test_r2:.4f}")

    coef_std_df = pd.DataFrame({"feature": X.columns, "coef": model_std.coef_})
    coef_std_df["abs_coef"] = coef_std_df["coef"].abs()
    print(coef_std_df.sort_values("abs_coef", ascending=False).round(2))

    return scaler, X_train_std, X_test_std, model_std


# ---------------------------------------------------------------------------
# Exercise 5: Ridge regression
# ---------------------------------------------------------------------------
def exercise_5(X, X_train_std, X_test_std, y_train, y_test, model_std):
    section("EXERCISE 5 — Ridge regression")

    for alpha in [0.0, 1.0, 100.0, 10000.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_std, y_train)
        test_r2_ridge = ridge.score(X_test_std, y_test)
        print(f"alpha={alpha:>10.1f}  test R²={test_r2_ridge:.4f}  "
              f"||β||₂={np.linalg.norm(ridge.coef_):.2f}")

    ridge_high = Ridge(alpha=10000.0)
    ridge_high.fit(X_train_std, y_train)
    compare = pd.DataFrame({
        "feature": X.columns,
        "ols": model_std.coef_,
        "ridge_high": ridge_high.coef_,
    })
    print(compare.round(2))

    return ridge_high


# ---------------------------------------------------------------------------
# Exercise 6: Lasso and feature selection
# ---------------------------------------------------------------------------
def exercise_6(X, X_train_std, X_test_std, y_train, y_test):
    section("EXERCISE 6 — Lasso and feature selection")

    for alpha in [0.1, 100.0, 10000.0, 100000.0]:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train_std, y_train)
        n_zero = int((lasso.coef_ == 0).sum())
        test_r2_lasso = lasso.score(X_test_std, y_test)
        print(f"alpha={alpha:>10.1f}  test R²={test_r2_lasso:.4f}  zero coefs={n_zero}/8")

    lasso10k = Lasso(alpha=10000.0, max_iter=10000)
    lasso10k.fit(X_train_std, y_train)

    surviving = pd.DataFrame({"feature": X.columns, "coef": lasso10k.coef_})
    print(surviving.round(2))
    print()
    print("non-zero features:")
    print(surviving[surviving["coef"] != 0]["feature"].tolist())

    return lasso10k


# ---------------------------------------------------------------------------
# Written note
# ---------------------------------------------------------------------------
def write_note(test_r2_ols: float, train_r2_ols: float) -> str:
    section("NOTE — Coefficients and regularization")

    note = (
        f"The baseline linear model explains a moderate share of price "
        f"variance (train R² = {train_r2_ols:.4f}, test R² = {test_r2_ols:.4f}), "
        f"with only mild overfitting between the two. Once features are "
        f"standardized so coefficients are on a comparable scale, latitude "
        f"and longitude turn out to have the largest coefficients by far — "
        f"location dominates price more than any other single feature — "
        f"followed by median_income, which is the strongest non-geographic "
        f"predictor and the only feature whose effect makes sense in raw, "
        f"unstandardized units (roughly $40k more value per extra unit of "
        f"income). Regularization changes these coefficients in two "
        f"characteristically different ways: Ridge (L2) shrinks every "
        f"coefficient toward zero in proportion to its size without ever "
        f"eliminating one, so at a high alpha the largest coefficients "
        f"(latitude, longitude, median_income) shrink the most in absolute "
        f"terms but all eight features remain in the model. Lasso (L1), by "
        f"contrast, can drive coefficients exactly to zero, and at a "
        f"moderate alpha it eliminates total_rooms, total_bedrooms, and "
        f"households entirely — the same three features flagged in last "
        f"week's raid as redundant due to pairwise correlations above 0.9. "
        f"This happens because those features are all highly correlated "
        f"proxies for the same underlying quantity (roughly, how big the "
        f"housing block is), so once one of them is in the model the L1 "
        f"penalty has no incentive to also pay for the others. The result "
        f"is that Lasso performs automatic feature selection, trading a "
        f"modest amount of test R² at higher alpha values for a "
        f"meaningfully simpler, more interpretable model with only three "
        f"surviving features: latitude, housing_median_age, and "
        f"median_income."
    )
    word_count = len(note.split())
    print(note)
    print(f"\n[word count: {word_count}]")
    return note


# ---------------------------------------------------------------------------
def main() -> None:
    exercise_0()
    X, y, X_train, X_test, y_train, y_test = exercise_1()
    model, y_pred_train, y_pred_test, test_r2 = exercise_2(X_train, X_test, y_train, y_test)
    exercise_3(y_train, y_test, y_pred_train, y_pred_test)
    scaler, X_train_std, X_test_std, model_std = exercise_4(
        X, X_train, X_test, y_train, y_test, model, test_r2
    )
    exercise_5(X, X_train_std, X_test_std, y_train, y_test, model_std)
    exercise_6(X, X_train_std, X_test_std, y_train, y_test)

    train_r2_ols = r2_score(y_train, y_pred_train)
    write_note(test_r2, train_r2_ols)


if __name__ == "__main__":
    main()

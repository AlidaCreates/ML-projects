import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
    plot_tree,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score, classification_report


HOUSING_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    "datasets/housing/housing.csv"
)


def load_housing():
    df = pd.read_csv(HOUSING_URL)
    df["total_bedrooms"] = df["total_bedrooms"].fillna(
        df["total_bedrooms"].median()
    )
    df = df.drop(columns=["ocean_proximity"])

    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )


def exercise1(X_train, X_test, y_train, y_test):
    print("\n=== Exercise 1: Decision Tree regressor ===")

    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X_train, y_train)

    train_r2 = tree.score(X_train, y_train)
    test_r2 = tree.score(X_test, y_test)

    print(f"train R²: {train_r2:.4f}")
    print(f"test R²:  {test_r2:.4f}")
    print(f"tree depth: {tree.get_depth()}")
    print(f"leaf count: {tree.get_n_leaves()}")

    return tree


def exercise2(X_train, X_test, y_train, y_test):
    print("\n=== Exercise 2: Effect of max_depth ===")

    depths = list(range(2, 21))
    train_scores = []
    test_scores = []

    for depth in depths:
        t = DecisionTreeRegressor(max_depth=depth, random_state=42)
        t.fit(X_train, y_train)
        train_scores.append(t.score(X_train, y_train))
        test_scores.append(t.score(X_test, y_test))

    results = pd.DataFrame(
        {
            "max_depth": depths,
            "train_R²": train_scores,
            "test_R²": test_scores,
        }
    )
    print(results.round(4).to_string(index=False))

    best_depth = depths[int(np.argmax(test_scores))]
    best_test = max(test_scores)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        depths,
        train_scores,
        marker="o",
        label="Train R²",
        color="tab:blue",
    )
    ax.plot(
        depths,
        test_scores,
        marker="s",
        label="Test R²",
        color="tab:orange",
    )
    ax.axvline(
        best_depth,
        color="gray",
        linestyle="--",
        label=f"Best depth = {best_depth} (R² = {best_test:.4f})",
    )
    ax.set_title("Decision Tree — Train vs Test R² by max_depth")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("R²")
    ax.legend()

    fig.savefig("ex02_depth.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\nBest depth: {best_depth}, test R²: {best_test:.4f}")


def exercise3(X_train, X_test, y_train, y_test):
    print("\n=== Exercise 3: Random Forest regressor ===")

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    print(f"train R²: {rf.score(X_train, y_train):.4f}")
    print(f"test R²:  {rf.score(X_test, y_test):.4f}")

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)

    default_tree = DecisionTreeRegressor(random_state=42)
    default_tree.fit(X_train, y_train)

    results = pd.DataFrame(
        {
            "model": [
                "LinearRegression",
                "DecisionTree (default)",
                "RandomForest (100)",
            ],
            "test_R²": [
                linreg.score(X_test, y_test),
                default_tree.score(X_test, y_test),
                rf.score(X_test, y_test),
            ],
        }
    )
    print("\nModel comparison:")
    print(results.round(4).to_string(index=False))

    print("\nn_estimators effect:")
    for n in [1, 10, 50, 100, 200]:
        rf_n = RandomForestRegressor(
            n_estimators=n,
            random_state=42,
            n_jobs=-1,
        )
        rf_n.fit(X_train, y_train)
        print(
            f"n_estimators={n:>3}  "
            f"test R²={rf_n.score(X_test, y_test):.4f}"
        )

    return rf


def exercise4(rf, X):
    print("\n=== Exercise 4: Feature importance ===")

    imp = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print(imp.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    imp_sorted = imp.sort_values("importance")
    ax.barh(
        imp_sorted["feature"],
        imp_sorted["importance"],
        color="tab:blue",
        edgecolor="black",
    )
    ax.set_title("Feature Importance — Random Forest")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    fig.savefig("ex04_importance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    top_5_rf = imp.head(5)["feature"].tolist()
    lasso_surviving = [
        "longitude",
        "latitude",
        "housing_median_age",
        "population",
        "median_income",
    ]
    agreement = set(top_5_rf) & set(lasso_surviving)

    print(f"\ntop-5 random forest features: {sorted(top_5_rf)}")
    print(f"Lasso surviving features:     {sorted(lasso_surviving)}")
    print(f"agreement: {len(agreement)} / 5 features")


def exercise5():
    print("\n=== Exercise 5: Decision Tree classifier ===")

    data = load_breast_cancer(as_frame=True)
    X_bc = data.data
    y_bc = data.target

    X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(
        X_bc,
        y_bc,
        test_size=0.2,
        random_state=42,
        stratify=y_bc,
    )

    print(X_bc_train.shape, X_bc_test.shape)

    dtc = DecisionTreeClassifier(max_depth=4, random_state=42)
    dtc.fit(X_bc_train, y_bc_train)

    print(f"train accuracy: {dtc.score(X_bc_train, y_bc_train):.4f}")
    print(f"test accuracy:  {dtc.score(X_bc_test, y_bc_test):.4f}")

    fig, ax = plt.subplots(figsize=(16, 10))
    plot_tree(
        dtc,
        feature_names=X_bc.columns.tolist(),
        class_names=["malignant", "benign"],
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=8,
    )
    ax.set_title("Decision Tree (max_depth=4) — Breast Cancer")
    fig.savefig("ex05_tree.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(
        f"root split feature: "
        f"{X_bc.columns[dtc.tree_.feature[0]]}"
    )
    print(f"root split threshold: {dtc.tree_.threshold[0]:.4f}")

    return X_bc_train, X_bc_test, y_bc_train, y_bc_test


def exercise6(X_bc_train, X_bc_test, y_bc_train, y_bc_test, X_bc):
    print("\n=== Exercise 6: Random Forest classifier ===")

    rfc = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    rfc.fit(X_bc_train, y_bc_train)

    print(f"train accuracy: {rfc.score(X_bc_train, y_bc_train):.4f}")
    print(f"test accuracy:  {rfc.score(X_bc_test, y_bc_test):.4f}")

    results = pd.DataFrame(
        {
            "model": [
                "LogisticRegression (Day 12)",
                "DecisionTree (max_depth=4)",
                "RandomForest (100)",
            ],
            "test_accuracy": [
                0.9737,
                accuracy_score(
                    y_bc_test,
                    DecisionTreeClassifier(
                        max_depth=4,
                        random_state=42,
                    )
                    .fit(X_bc_train, y_bc_train)
                    .predict(X_bc_test),
                ),
                rfc.score(X_bc_test, y_bc_test),
            ],
        }
    )

    print("\nClassification model comparison:")
    print(results.round(4).to_string(index=False))

    y_pred_rfc = rfc.predict(X_bc_test)
    print("\nClassification report:")
    print(
        classification_report(
            y_bc_test,
            y_pred_rfc,
            target_names=["malignant", "benign"],
        )
    )

    imp_clf = (
        pd.DataFrame(
            {
                "feature": X_bc.columns,
                "importance": rfc.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(5)
    )

    print("Top 5 classifier features:")
    print(imp_clf.round(4).to_string(index=False))


def main():
    print("Tree Models — Decision Trees and Random Forests")

    X_train, X_test, y_train, y_test = load_housing()
    print(f"Housing train/test shapes: {X_train.shape} {X_test.shape}")

    exercise1(X_train, X_test, y_train, y_test)
    exercise2(X_train, X_test, y_train, y_test)
    rf = exercise3(X_train, X_test, y_train, y_test)
    exercise4(rf, X_train)

    X_bc_train, X_bc_test, y_bc_train, y_bc_test = exercise5()
    data = load_breast_cancer(as_frame=True)
    exercise6(
        X_bc_train,
        X_bc_test,
        y_bc_train,
        y_bc_test,
        data.data,
    )

    print("\nDone. Generated:")
    print("- ex02_depth.png")
    print("- ex04_importance.png")
    print("- ex05_tree.png")


if __name__ == "__main__":
    main()

"""
Diamond EDA Piscine — Day: Exploratory Data Analysis Pipeline
================================================================
Runs the four-stage EDA pipeline (inspection, univariate, bivariate,
multivariate) on the seaborn `diamonds` dataset, plus outlier detection
and a final written summary.

Run with:
    python eda.py

Requires: pandas, numpy, matplotlib, seaborn (piscine1b_w2 venv)
All figures are saved as PNG files in the current working directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Exercise 0: Environment and libraries
# ---------------------------------------------------------------------------
def exercise_0() -> pd.DataFrame:
    section("EXERCISE 0 — Environment and libraries")
    print(f"seaborn version: {sns.__version__}")

    df = sns.load_dataset("diamonds")
    print(df.shape)
    print(df.head(3))
    return df


# ---------------------------------------------------------------------------
# Exercise 1: Initial inspection
# ---------------------------------------------------------------------------
def exercise_1(df: pd.DataFrame) -> None:
    section("EXERCISE 1 — Initial inspection")

    print(df.shape)
    print(df.dtypes)
    print(df.isna().sum())
    print(df.duplicated().sum())

    print(df.describe().round(2))

    zero_dim = ((df["x"] == 0) | (df["y"] == 0) | (df["z"] == 0)).sum()
    print(zero_dim)

    for col in ["cut", "color", "clarity"]:
        print(f"--- {col} ---")
        print(df[col].value_counts())


# ---------------------------------------------------------------------------
# Exercise 2: Univariate analysis
# ---------------------------------------------------------------------------
def exercise_2(df: pd.DataFrame) -> None:
    section("EXERCISE 2 — Univariate analysis")

    # --- numeric histograms ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, col in zip(axes.flat, ["carat", "depth", "table", "price"]):
        sns.histplot(data=df, x=col, bins=50, ax=ax, color="tab:blue")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    fig.suptitle("Univariate — Numeric Features", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("ex02_univariate_numeric.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    skew = df[["carat", "depth", "table", "price"]].skew().round(4)
    print(skew)

    print(f"price mean:   {df['price'].mean():.2f}")
    print(f"price median: {df['price'].median():.2f}")
    print(f"mean > median: {df['price'].mean() > df['price'].median()}")

    # --- categorical countplots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, col, order in zip(
        axes,
        ["cut", "color", "clarity"],
        [["Fair", "Good", "Very Good", "Premium", "Ideal"],
         ["D", "E", "F", "G", "H", "I", "J"],
         ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]],
    ):
        sns.countplot(data=df, x=col, order=order, ax=ax, hue=col,
                       palette="viridis", legend=False)
        ax.set_title(f"Counts of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    fig.suptitle("Univariate — Categorical Features", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("ex02_univariate_cat.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exercise 3: Bivariate analysis
# ---------------------------------------------------------------------------
def exercise_3(df: pd.DataFrame) -> None:
    section("EXERCISE 3 — Bivariate analysis")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df.sample(2000, random_state=0),
                     x="carat", y="price",
                     hue="cut", alpha=0.6, ax=ax)
    ax.set_title("Carat vs Price (2000 sampled diamonds)")
    ax.set_xlabel("Carat")
    ax.set_ylabel("Price (USD)")
    fig.savefig("ex03_carat_price.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    mean_price_by_cut = (
        df.groupby("cut", observed=True)["price"]
          .mean().round(2)
    )
    print(mean_price_by_cut)

    fig, ax = plt.subplots(figsize=(9, 5))
    order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    sns.boxplot(data=df, x="cut", y="price", order=order, ax=ax,
                hue="cut", palette="Blues", legend=False)
    ax.set_title("Price by Cut")
    ax.set_xlabel("Cut")
    ax.set_ylabel("Price (USD)")
    fig.savefig("ex03_price_by_cut.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(df.groupby("clarity", observed=True)["carat"].mean().round(4))


# ---------------------------------------------------------------------------
# Exercise 4: Multivariate synthesis
# ---------------------------------------------------------------------------
def exercise_4(df: pd.DataFrame) -> pd.DataFrame:
    section("EXERCISE 4 — Multivariate synthesis")

    numeric_cols = ["carat", "depth", "table", "price", "x", "y", "z"]
    corr = df[numeric_cols].corr()
    print(corr.round(4))

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix — Diamond Numeric Features")
    fig.savefig("ex04_corr.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    sample = df.sample(1000, random_state=0)
    g = sns.pairplot(sample[["carat", "depth", "table", "price", "cut"]],
                      hue="cut", diag_kind="kde", palette="Set1",
                      height=2.0, plot_kws={"alpha": 0.5})
    g.figure.suptitle("Pairplot — Diamond Features by Cut", y=1.02)
    g.savefig("ex04_pairplot.png", dpi=120, bbox_inches="tight")
    plt.close(g.figure)

    return corr


# ---------------------------------------------------------------------------
# Exercise 5: Outlier detection
# ---------------------------------------------------------------------------
def iqr_outliers_count(s: pd.Series) -> int:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((s < lower) | (s > upper)).sum())


def exercise_5(df: pd.DataFrame) -> pd.DataFrame:
    section("EXERCISE 5 — Outlier detection")

    for col in ["carat", "depth", "table", "price", "x", "y", "z"]:
        print(f"{col}: {iqr_outliers_count(df[col])}")

    print(df[df["y"] > 30][["carat", "x", "y", "z", "price"]])
    print(df[df["z"] > 30][["carat", "x", "y", "z", "price"]])

    df_clean = df[
        (df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0) &
        (df["y"] < 30) & (df["z"] < 30)
    ].copy()

    print(f"original rows: {len(df)}")
    print(f"cleaned rows:  {len(df_clean)}")
    print(f"dropped:       {len(df) - len(df_clean)}")

    print(f"before: {df[['carat', 'price']].corr().iloc[0, 1]:.4f}")
    print(f"after:  {df_clean[['carat', 'price']].corr().iloc[0, 1]:.4f}")

    return df_clean


# ---------------------------------------------------------------------------
# Exercise 6: EDA summary
# ---------------------------------------------------------------------------
def exercise_6(df: pd.DataFrame, df_clean: pd.DataFrame, corr: pd.DataFrame) -> str:
    section("EXERCISE 6 — EDA summary")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(data=df_clean.sample(2000, random_state=0),
                     x="carat", y="price", hue="cut", alpha=0.6, ax=axes[0])
    axes[0].set_title("Carat vs Price (cleaned)")
    axes[0].set_xlabel("Carat")
    axes[0].set_ylabel("Price (USD)")

    order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    sns.boxplot(data=df_clean, x="cut", y="price", order=order, ax=axes[1],
                hue="cut", palette="Blues", legend=False)
    axes[1].set_title("Price by Cut")
    axes[1].set_xlabel("Cut")
    axes[1].set_ylabel("Price (USD)")

    fig.suptitle("Diamond EDA — Summary", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("ex06_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    n_rows, n_cols = df.shape
    n_dupes = df.duplicated().sum()
    carat_price_corr = corr.loc["carat", "price"]
    price_mean = df["price"].mean()
    price_median = df["price"].median()
    mean_by_cut = df.groupby("cut", observed=True)["price"].mean()
    n_dropped = len(df) - len(df_clean)

    summary = (
        f"The diamonds dataset contains {n_rows:,} rows and {n_cols} columns "
        f"describing physical characteristics and price for each stone, "
        f"including {n_dupes} exact duplicate rows that were identified during "
        f"initial inspection. Among all physical features, carat weight is by "
        f"far the strongest single predictor of price, correlating at "
        f"{carat_price_corr:.4f}, with x, y, and z (the physical dimensions) "
        f"close behind since they are all effectively proxies for size. Price "
        f"itself is strongly right-skewed: the mean price is ${price_mean:,.2f} "
        f"while the median is only ${price_median:,.2f}, meaning a relatively "
        f"small number of large, expensive stones pull the average well above "
        f"the typical transaction. A counter-intuitive pattern emerged in the "
        f"bivariate analysis: Fair-cut diamonds have a higher mean price "
        f"(${mean_by_cut['Fair']:,.2f}) than Ideal-cut diamonds "
        f"(${mean_by_cut['Ideal']:,.2f}), which is not because poor cutting adds "
        f"value but because cut is confounded with carat — Fair diamonds in this "
        f"dataset tend to be larger stones, and size dominates price far more "
        f"than cut quality does. This is a clear illustration of why bivariate "
        f"analysis alone can mislead and why multivariate synthesis is needed "
        f"before drawing conclusions. During outlier detection, {n_dropped} rows "
        f"were dropped for having physically impossible dimensions, such as a "
        f"width of 0mm or a y value of 58.9mm on a 2-carat stone, without "
        f"materially changing the carat-price correlation. For downstream "
        f"modeling, carat should be treated as the primary feature, and either "
        f"price should be log-transformed to address its skew or a tree-based "
        f"model should be used, since tree-based models handle skewed targets "
        f"and non-linear feature interactions naturally."
    )

    word_count = len(summary.split())
    print("=== EDA SUMMARY ===")
    print(summary)
    print(f"\n[word count: {word_count}]")

    return summary


# ---------------------------------------------------------------------------
def main() -> None:
    df = exercise_0()
    exercise_1(df)
    exercise_2(df)
    exercise_3(df)
    corr = exercise_4(df)
    df_clean = exercise_5(df)
    exercise_6(df, df_clean, corr)


if __name__ == "__main__":
    main()

"""California Housing EDA for the raid/california-eda project."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
)
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

NUMERIC_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "median_house_value",
]

PROXIMITY_ORDER = ["INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BAY", "ISLAND"]


def load_housing(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df


def clean_housing(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    if cleaned["total_bedrooms"].isna().any():
        median_value = cleaned["total_bedrooms"].median()
        cleaned["total_bedrooms"] = cleaned["total_bedrooms"].fillna(median_value)
    return cleaned


def inspect_data(df: pd.DataFrame) -> None:
    print("Shape:", df.shape)
    print("Dtypes:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isna().sum())
    print("\nocean_proximity value counts:\n", df["ocean_proximity"].value_counts())
    print("\nDescriptive statistics:\n", df.describe().round(2))
    max_value = df["median_house_value"].max()
    print("\nKnown price cap: median_house_value.max() =", f"{max_value:.2f}")


def univariate_analysis(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for ax, column in zip(axes.flatten(), NUMERIC_COLUMNS):
        ax.hist(df[column], bins=50, color="#2c7fb8", edgecolor="black")
        ax.set_title(column)
        ax.set_xlabel("")
    fig.suptitle("Univariate Distributions", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(PLOTS_DIR / "ex02_univariate.png", dpi=150)
    plt.close(fig)

    skewness = df[NUMERIC_COLUMNS].skew()
    print("\nSkewness per numeric feature:\n", skewness.round(2))
    cap_count = int((df["median_house_value"] == 500001.0).sum())
    print("\nRows at artificial cap:", cap_count)


def bivariate_analysis(df: pd.DataFrame) -> None:
    # Exclude the target itself so it doesn't show up as its own
    # top "predictor" with a trivial correlation of 1.0.
    feature_columns = [c for c in NUMERIC_COLUMNS if c != "median_house_value"]
    corr_series = (
        df[feature_columns]
        .corrwith(df["median_house_value"])
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )
    print("\nCorrelation with median_house_value:\n", corr_series.round(4))

    sample = df.sample(n=2000, random_state=42)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        x="median_income",
        y="median_house_value",
        hue="ocean_proximity",
        palette="tab10",
        data=sample,
        alpha=0.8,
        ax=ax,
        edgecolor="none",
    )
    ax.set_title("Median income vs. median house value")
    fig.savefig(PLOTS_DIR / "ex03_income_value.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(
        x="ocean_proximity",
        y="median_house_value",
        order=PROXIMITY_ORDER,
        data=df,
        palette="pastel",
        ax=ax,
    )
    ax.set_title("Median house value by ocean proximity")
    fig.savefig(PLOTS_DIR / "ex03_value_by_proximity.png", dpi=150)
    plt.close(fig)

    medians = df.groupby("ocean_proximity")["median_house_value"].median().reindex(PROXIMITY_ORDER)
    print("\nPer-proximity median house values:\n", medians.astype(int))


def multivariate_analysis(df: pd.DataFrame) -> None:
    corr_matrix = df[NUMERIC_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title("Correlation matrix for numeric features")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ex04_corr.png", dpi=150)
    plt.close(fig)

    top_features = ["median_income", "total_rooms", "housing_median_age", "latitude", "median_house_value"]
    sample = df.sample(n=1000, random_state=42)
    pairplot = sns.pairplot(
        sample[top_features + ["ocean_proximity"]],
        hue="ocean_proximity",
        palette="tab10",
        corner=False,
        plot_kws={"alpha": 0.6, "s": 20},
    )
    pairplot.fig.suptitle("Pairplot for top predictors", y=1.02)
    pairplot.savefig(PLOTS_DIR / "ex04_pairplot.png", dpi=150)
    plt.close(pairplot.fig)

    high_corr = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .abs()
        .sort_values(ascending=False)
    )
    high_corr_pairs = high_corr[high_corr > 0.9]
    print("\nHighly correlated feature pairs (|corr| > 0.9):\n", high_corr_pairs)


def geographic_charts(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        df["longitude"],
        df["latitude"],
        c=df["median_house_value"],
        cmap="plasma",
        s=np.clip(df["population"] / 100, 5, 80),
        alpha=0.8,
        linewidths=0,
    )
    ax.set_title("California housing median values by location")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(scatter, ax=ax, label="median_house_value")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ex05_geo.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplot_mosaic(
        "AAB\nAAC\nDEF",
        figsize=(18, 14),
    )

    # Single scatter call for panel A; its return value is reused for the colorbar.
    scatter_a = axes["A"].scatter(
        df["longitude"],
        df["latitude"],
        c=df["median_house_value"],
        cmap="plasma",
        s=np.clip(df["population"] / 100, 5, 80),
        alpha=0.8,
        linewidths=0,
    )
    axes["A"].set_title("Geographic price pattern")
    axes["A"].set_xlabel("Longitude")
    axes["A"].set_ylabel("Latitude")
    fig.colorbar(
        scatter_a,
        ax=axes["A"],
        label="median_house_value",
        orientation="vertical",
        fraction=0.046,
        pad=0.04,
    )

    axes["B"].hist(df["median_house_value"], bins=50, color="#2c7fb8", edgecolor="black")
    axes["B"].set_title("Median house value distribution")

    axes["C"].hist(df["median_income"], bins=50, color="#feb24c", edgecolor="black")
    axes["C"].set_title("Median income distribution")

    sns.scatterplot(
        x="median_income",
        y="median_house_value",
        hue="ocean_proximity",
        palette="tab10",
        data=df.sample(n=2000, random_state=42),
        alpha=0.7,
        ax=axes["D"],
        edgecolor="none",
    )
    axes["D"].set_title("Income vs. price by proximity")

    sns.boxplot(
        x="ocean_proximity",
        y="median_house_value",
        order=PROXIMITY_ORDER,
        data=df,
        palette="pastel",
        ax=axes["E"],
    )
    axes["E"].set_title("Price by ocean proximity")
    axes["E"].tick_params(axis="x", rotation=15)

    top_corr_matrix = df[["median_income", "total_rooms", "housing_median_age", "latitude"]].corr()
    sns.heatmap(
        top_corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        ax=axes["F"],
    )
    axes["F"].set_title("Correlation of the top features")

    fig.suptitle("California Housing EDA — Chart Pack", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ex05_chartpack.png", dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_housing()
    inspect_data(df)
    cleaned = clean_housing(df)
    print("\nAfter cleaning: total missing values =", cleaned.isna().sum().sum())
    univariate_analysis(cleaned)
    bivariate_analysis(cleaned)
    multivariate_analysis(cleaned)
    geographic_charts(cleaned)
    print("\nEDA complete. Plot files saved to", PLOTS_DIR)


if __name__ == "__main__":
    main()
"""
S&P 500 Financial EDA — Day: Returns, Volatility, Drawdown
=============================================================
Downloads S&P 500 (^GSPC) daily history via yfinance, computes returns,
rolling volatility, drawdown, and compares against the XLK tech sector ETF.

Run with:
    python financial_eda.py

Requires: yfinance, pandas, numpy, matplotlib, seaborn, scipy
(piscine1b_w2 venv — only yfinance needs to be newly installed, see
requirements.txt)

NOTE: this script needs outbound internet access to query1/query2
.finance.yahoo.com. It will not run inside a network-sandboxed
environment that blocks that host — run it on your own machine.
"""

import os

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

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
    print(f"yfinance version: {yf.__version__}")

    sp500 = yf.download("^GSPC", start="2019-01-01", end="2024-01-01",
                         auto_adjust=True, progress=False)
    sp500.columns = sp500.columns.get_level_values(0)
    print(sp500.shape)
    print(sp500.head(3))

    sp500.to_csv("sp500.csv")
    return sp500


# ---------------------------------------------------------------------------
# Exercise 1: Initial inspection
# ---------------------------------------------------------------------------
def exercise_1() -> pd.DataFrame:
    section("EXERCISE 1 — Initial inspection")

    sp500 = pd.read_csv("sp500.csv", index_col="Date", parse_dates=True)
    print(sp500.shape)
    print(sp500.dtypes)
    print(sp500.isna().sum())

    print(f"start: {sp500.index.min().date()}")
    print(f"end:   {sp500.index.max().date()}")
    print(f"trading days: {len(sp500)}")

    total_return = sp500["Close"].iloc[-1] / sp500["Close"].iloc[0] - 1
    print(f"total return: {total_return:.4f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sp500.index, sp500["Close"], color="tab:blue", linewidth=1.0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("S&P 500 Close — 2019 to 2023")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price (USD)")
    fig.savefig("ex01_price.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(os.path.exists("ex01_price.png"))

    return sp500


# ---------------------------------------------------------------------------
# Exercise 2: Returns and their distribution
# ---------------------------------------------------------------------------
def exercise_2(sp500: pd.DataFrame) -> pd.Series:
    section("EXERCISE 2 — Returns and their distribution")

    sp500["return_arith"] = sp500["Close"].pct_change()
    sp500["return_log"] = np.log(sp500["Close"]).diff()

    print(sp500[["return_arith", "return_log"]].dropna().describe().round(6))

    worst_day = sp500["return_log"].idxmin()
    best_day = sp500["return_log"].idxmax()
    print(f"worst day: {worst_day.date()}  return: {sp500.loc[worst_day, 'return_log']:.4f}")
    print(f"best day:  {best_day.date()}  return: {sp500.loc[best_day, 'return_log']:.4f}")

    returns = sp500["return_log"].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(returns, bins=80, kde=False, stat="density", ax=ax,
                 color="tab:blue", alpha=0.7)

    x = np.linspace(returns.min(), returns.max(), 200)
    mu = returns.mean()
    sigma = returns.std()
    ax.plot(x, 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
            color="red", linewidth=2.0, label="Normal fit")

    ax.set_title("Distribution of Daily Log Returns")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend()

    fig.savefig("ex02_returns_dist.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    print(f"skewness: {skew:.4f}")
    print(f"excess kurtosis: {kurt:.4f}")

    return returns


# ---------------------------------------------------------------------------
# Exercise 3: Rolling statistics
# ---------------------------------------------------------------------------
def exercise_3(sp500: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    section("EXERCISE 3 — Rolling statistics")

    roll_std_21 = sp500["return_log"].rolling(window=21).std()
    roll_mean_252 = sp500["return_log"].rolling(window=252).mean()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(sp500.index, sp500["Close"], color="tab:blue", linewidth=1.0)
    axes[0].set_title("S&P 500 Close")
    axes[0].set_ylabel("Close (USD)")

    axes[1].plot(roll_std_21.index, roll_std_21, color="tab:red", linewidth=1.0)
    axes[1].set_title("21-Day Rolling Std of Daily Log Returns")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Rolling Std")

    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig("ex03_rolling.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    peak_vol_date = roll_std_21.idxmax()
    peak_vol = roll_std_21.max()
    print(f"peak rolling vol date: {peak_vol_date.date()}  std: {peak_vol:.4f}")

    final_mean = roll_mean_252.iloc[-1]
    print(f"trailing 252-day mean log return on {roll_mean_252.index[-1].date()}: {final_mean:.6f}")

    return roll_std_21, roll_mean_252


# ---------------------------------------------------------------------------
# Exercise 4: Drawdown
# ---------------------------------------------------------------------------
def exercise_4(sp500: pd.DataFrame) -> pd.DataFrame:
    section("EXERCISE 4 — Drawdown")

    sp500["running_max"] = sp500["Close"].cummax()
    sp500["drawdown"] = sp500["Close"] / sp500["running_max"] - 1

    max_dd = sp500["drawdown"].min()
    max_dd_date = sp500["drawdown"].idxmin()
    print(f"max drawdown: {max_dd:.4f}  on {max_dd_date.date()}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(sp500.index, sp500["Close"], color="tab:blue", linewidth=1.0, label="Close")
    axes[0].plot(sp500.index, sp500["running_max"], color="tab:green", linewidth=1.0,
                 linestyle="--", label="Running max")
    axes[0].set_title("S&P 500 Close and Running Maximum")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend()

    axes[1].fill_between(sp500.index, sp500["drawdown"], 0, color="tab:red", alpha=0.5)
    axes[1].set_title("Drawdown from Running Maximum")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Drawdown")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig("ex04_drawdown.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(os.path.exists("ex04_drawdown.png"))

    return sp500


# ---------------------------------------------------------------------------
# Exercise 5: Multi-asset comparison
# ---------------------------------------------------------------------------
def exercise_5(sp500: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    section("EXERCISE 5 — Multi-asset comparison")

    xlk = yf.download("XLK", start="2019-01-01", end="2024-01-01",
                       auto_adjust=True, progress=False)
    xlk.columns = xlk.columns.get_level_values(0)
    xlk_close = xlk["Close"].rename("XLK")

    both = pd.concat([sp500["Close"].rename("SP500"), xlk_close], axis=1).dropna()
    print(both.shape)
    print(both.head(3))

    returns_both = np.log(both).diff().dropna()
    corr_matrix = returns_both.corr().round(4)
    print(corr_matrix)
    returns_corr = corr_matrix.loc["SP500", "XLK"]

    cumulative = (1 + both.pct_change().dropna()).cumprod()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cumulative.index, cumulative["SP500"], color="tab:blue", linewidth=1.5, label="S&P 500")
    ax.plot(cumulative.index, cumulative["XLK"], color="tab:orange", linewidth=1.5, label="XLK")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.set_title("Cumulative Return — S&P 500 vs XLK")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (1.0 = start)")
    ax.legend()

    fig.savefig("ex05_compare.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(cumulative.iloc[-1].round(4))

    return cumulative, returns_corr


# ---------------------------------------------------------------------------
# Exercise 6: Summary
# ---------------------------------------------------------------------------
def exercise_6(sp500: pd.DataFrame, returns: pd.Series, roll_std_21: pd.Series,
               cumulative: pd.DataFrame, returns_corr: float) -> str:
    section("EXERCISE 6 — Summary")

    mosaic = """
AA
BC
DD
"""
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(14, 12))

    # A
    axes["A"].plot(sp500.index, sp500["Close"], color="tab:blue", linewidth=1.0, label="Close")
    axes["A"].plot(sp500.index, sp500["running_max"], color="tab:green", linewidth=1.0,
                    linestyle="--", label="Running max")
    axes["A"].set_title("S&P 500 Close and Running Max")
    axes["A"].set_ylabel("Price (USD)")
    axes["A"].legend()

    # B
    axes["B"].plot(roll_std_21.index, roll_std_21, color="tab:red", linewidth=1.0)
    axes["B"].set_title("21-Day Rolling Std")
    axes["B"].set_ylabel("Std")

    # C
    sns.histplot(returns, bins=80, kde=False, stat="density", ax=axes["C"],
                 color="tab:blue", alpha=0.7)
    x = np.linspace(returns.min(), returns.max(), 200)
    mu = returns.mean()
    sigma = returns.std()
    axes["C"].plot(x, 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                   color="red", linewidth=2.0)
    axes["C"].set_title("Daily Log Return Distribution")
    axes["C"].set_xlabel("Log Return")

    # D
    axes["D"].plot(cumulative.index, cumulative["SP500"], color="tab:blue", linewidth=1.5, label="S&P 500")
    axes["D"].plot(cumulative.index, cumulative["XLK"], color="tab:orange", linewidth=1.5, label="XLK")
    axes["D"].set_title("Cumulative Return — S&P 500 vs XLK")
    axes["D"].set_xlabel("Date")
    axes["D"].set_ylabel("Cumulative Return")
    axes["D"].legend()

    for label in ["A", "B", "D"]:
        axes[label].xaxis.set_major_locator(mdates.YearLocator())
        axes[label].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("S&P 500 EDA — 2019 to 2023", fontsize=16, fontweight="bold")
    fig.tight_layout()

    fig.savefig("ex06_chartpack.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    total_return = sp500["Close"].iloc[-1] / sp500["Close"].iloc[0] - 1
    mean_ret = returns.mean()
    std_ret = returns.std()
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    peak_vol_date = roll_std_21.idxmax()
    max_dd = sp500["drawdown"].min()
    max_dd_date = sp500["drawdown"].idxmin()
    final_sp500 = cumulative["SP500"].iloc[-1]
    final_xlk = cumulative["XLK"].iloc[-1]

    summary = (
        f"The dataset covers {len(sp500):,} trading days of S&P 500 (^GSPC) "
        f"daily prices from {sp500.index.min().date()} to {sp500.index.max().date()}, "
        f"over which the index delivered a cumulative return of "
        f"{total_return:.2%}, roughly doubling in five years including the "
        f"2020 crash and recovery. Daily log returns averaged "
        f"{mean_ret:.4f} with a standard deviation of {std_ret:.4f}, and their "
        f"shape departs sharply from a normal distribution: skewness of "
        f"{skew:.2f} shows the left tail (crash days) is heavier than the "
        f"right, and excess kurtosis of {kurt:.2f} means extreme moves are "
        f"far more frequent than a normal curve would predict. Rolling "
        f"21-day volatility peaked around {peak_vol_date.date()}, and the "
        f"index hit its maximum drawdown of {max_dd:.2%} on "
        f"{max_dd_date.date()} — both landmarks of the COVID-19 crash. "
        f"Comparing against the XLK technology sector ETF, daily log returns "
        f"were highly correlated ({returns_corr:.4f}), unsurprising since XLK "
        f"holdings are a subset of the S&P 500, but XLK compounded to "
        f"{final_xlk:.2f}x its starting value versus {final_sp500:.2f}x for "
        f"the S&P 500, meaning tech meaningfully outperformed the broad "
        f"index over this window. It's worth being careful not to "
        f"over-generalize from the tail statistics here: the heavy left tail "
        f"and elevated kurtosis are driven almost entirely by the "
        f"March 2020 COVID-19 shock, a single well-documented regime-shift "
        f"event, rather than being a stable year-round feature of S&P 500 "
        f"returns."
    )

    word_count = len(summary.split())
    print("=== EDA SUMMARY ===")
    print(summary)
    print(f"\n[word count: {word_count}]")

    return summary


# ---------------------------------------------------------------------------
def main() -> None:
    sp500 = exercise_0()
    sp500 = exercise_1()
    returns = exercise_2(sp500)
    roll_std_21, roll_mean_252 = exercise_3(sp500)
    sp500 = exercise_4(sp500)
    cumulative, returns_corr = exercise_5(sp500)
    exercise_6(sp500, returns, roll_std_21, cumulative, returns_corr)


if __name__ == "__main__":
    main()

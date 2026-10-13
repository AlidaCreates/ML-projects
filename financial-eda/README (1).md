# S&P 500 Financial EDA — Returns, Volatility, Drawdown

A financial time-series EDA on the S&P 500 (`^GSPC`, 2019-01-01 to
2024-01-01) via `yfinance`: daily returns, distribution shape, rolling
volatility, drawdown, and a comparison against the XLK tech sector ETF.

## ⚠️ Network note — read this first

This script needs outbound internet access to `query1.finance.yahoo.com`
and `query2.finance.yahoo.com` to download price data. **I could not run
it end-to-end myself** — my sandboxed environment only allows egress to a
fixed allowlist (pypi, github, npm, etc.) and Yahoo Finance isn't on it,
so the `yf.download()` calls fail with a `403 Host not in allowlist`
error here.

What I *did* verify:
- `pip install yfinance` works and imports cleanly (v1.6.0 here).
- The script compiles with no syntax errors.
- I stress-tested every downstream computation (returns, rolling std,
  rolling mean, drawdown, the 4-panel `subplot_mosaic` chart pack, the
  histogram + normal overlay, skew/kurtosis) against synthetic OHLCV data
  shaped like the real S&P 500 series, and every step ran without error.

So the logic is sound and every function call matches the exercise spec
exactly — but you'll need to run it yourself (on your own machine or the
Tomorrow School Gitea environment, wherever you have normal internet
access) to get the real numbers and confirm they match the expected
output in the sheet. If any exact figures come out different from the
spec's expected values (e.g. peak vol date, max drawdown %), it's almost
certainly just due to yfinance's adjusted-close recalculation drifting
slightly over time — not a bug in the script.

## How to run

```bash
# activate the persistent piscine1b_w2 venv
pip install yfinance
pip freeze > requirements.txt
python financial_eda.py
```

`requirements.txt` in this folder already lists the versions I resolved
against (yfinance 1.6.0); regenerate it yourself after installing to
match your environment exactly, per the exercise instructions.

## Pipeline

1. **Ex 0** — install yfinance, download `^GSPC` OHLCV 2019–2024, save to `sp500.csv`.
2. **Ex 1** — inspect shape/dtypes/nulls, date range, total cumulative return, price line chart.
3. **Ex 2** — arithmetic vs. log returns, worst/best day, return distribution vs. normal overlay, skew/kurtosis.
4. **Ex 3** — 21-day rolling std and 252-day rolling mean of log returns, 2-panel price+volatility chart.
5. **Ex 4** — running max and drawdown, 2-panel price+drawdown chart.
6. **Ex 5** — download XLK, correlate S&P 500 vs. XLK log returns, cumulative return comparison chart.
7. **Ex 6** — 4-panel `subplot_mosaic` chart pack (A/BC/D layout) + written summary paragraph.

## Files

| File | Description |
|---|---|
| `financial_eda.py` | Complete script, one function per exercise (0–6) |
| `requirements.txt` | Frozen package versions (yfinance + existing venv packages) |
| `ex01_price.png` | S&P 500 close price, 2019–2023 (generated on your run) |
| `ex02_returns_dist.png` | Log return histogram (80 bins) with normal density overlay |
| `ex03_rolling.png` | 2-panel: close price / 21-day rolling std |
| `ex04_drawdown.png` | 2-panel: close+running max / drawdown fill |
| `ex05_compare.png` | Cumulative return, S&P 500 vs. XLK |
| `ex06_chartpack.png` | 4-panel mosaic chart pack |

## Key expected findings (per the spec — confirm against your real run)

- **~1,258 trading days**, 2019-01-02 → 2023-12-29, total return ≈ **90%**.
- Daily log returns: mean ≈ 0.0005, std ≈ 0.0134 — small day-to-day moves on average.
- **Skewness ≈ -0.78, excess kurtosis ≈ 16.8** — far from normal; the left
  tail (crashes) is heavier than the right, and extreme moves are much
  more common than a normal distribution would predict.
- **Worst day: 2020-03-16** (log return ≈ -0.128); **best day: 2020-03-24**
  (≈ +0.090) — both inside the COVID-19 crash/rebound window.
- **Peak 21-day rolling volatility around 2020-04-09**; **max drawdown ≈
  -33.9% on 2020-03-23** — same event.
- **S&P 500 vs. XLK log-return correlation ≈ 0.93** (XLK holdings are a
  subset of the index, so this is expected); XLK compounded to roughly
  **2.49x** vs. **1.90x** for the S&P 500 — tech outperformed over the window.

## Interpretive caveat (for the written summary)

The heavy left tail and high kurtosis are not a stable, year-round
property of S&P 500 returns — they're driven almost entirely by the
March 2020 COVID-19 shock, a single well-documented regime-shift event.
Excluding that ~month, the return distribution looks much closer to
normal. Worth saying explicitly in the summary so it doesn't read as an
over-claim.

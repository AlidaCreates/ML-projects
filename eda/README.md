# Diamond EDA Pipeline

A structured, four-stage exploratory data analysis of the seaborn `diamonds`
dataset (53,940 rows), built for the Tomorrow School piscine day on EDA.
Role-play framing: a data scientist at a jewelry e-commerce platform
preparing a one-page EDA for a product manager before any modeling work.

## Pipeline

The script runs the same four-stage EDA workflow used for the rest of the
curriculum:

1. **Initial inspection** — shape, dtypes, missing values, duplicates, `describe()`.
2. **Univariate analysis** — distribution of each variable on its own (histograms, countplots, skew).
3. **Bivariate analysis** — each feature's relationship to price (scatter, boxplot, group means).
4. **Multivariate synthesis** — correlation heatmap and pairplot across features.

Followed by outlier detection (1.5×IQR rule + physically impossible rows)
and a final one-paragraph written summary.

## Files

| File | Description |
|---|---|
| `eda.py` | Complete runnable script, one function per exercise (0–6) |
| `ex02_univariate_numeric.png` | 2×2 histogram grid: carat, depth, table, price |
| `ex02_univariate_cat.png` | 1×3 countplots: cut, color, clarity (domain-ordered) |
| `ex03_carat_price.png` | Carat vs price scatter (2,000-point sample, colored by cut) |
| `ex03_price_by_cut.png` | Price boxplot by cut, ordered Fair → Ideal |
| `ex04_corr.png` | 7×7 annotated correlation heatmap |
| `ex04_pairplot.png` | Pairplot of carat/depth/table/price, colored by cut |
| `ex06_summary.png` | Two-panel summary: scatter (cleaned data) + boxplot |

## How to run

```bash
# activate the persistent piscine1b_w2 venv, then:
python eda.py
```

No new packages are needed — pandas, numpy, matplotlib, seaborn are the
same ones from the venv set up on Day 06. Running the script prints every
diagnostic (shape, dtypes, value counts, correlations, outlier counts) and
saves all seven PNGs to the working directory.

## Key findings

- **146 exact duplicate rows** and **20 rows with a 0mm dimension** were
  found during initial inspection — both data quality issues, not real
  diamonds.
- **Carat is the strongest single predictor of price** (r = 0.9216); x, y, z
  correlate almost as strongly because they're all just proxies for size.
- **Price is strongly right-skewed** (mean $3,932.80 vs. median $2,401.00) —
  a long tail of large, expensive stones pulls the average up.
- **The cut/price paradox**: Fair-cut diamonds average a *higher* price
  ($4,358.76) than Ideal-cut diamonds ($3,457.54). This isn't because worse
  cutting is worth more — it's confounded by carat. Fair diamonds in this
  dataset are simply bigger on average (mean carat by clarity shows the same
  pattern: I1 clarity averages 1.28 carat vs. 0.51 for IF). This is the
  clearest illustration in the whole exercise of why bivariate analysis
  alone can mislead, and why the multivariate step matters.
- **23 rows were dropped** as physically impossible (0mm dimensions, or a
  y/z value >30mm — e.g., a 2-carat stone with a reported 58.9mm width).
  Dropping them left the carat-price correlation unchanged (0.9216 → 0.9216),
  confirming they were noise, not signal.
- **Next step for modeling**: use carat as the primary feature, and either
  log-transform price or use a tree-based model (handles skew and
  non-linear interactions naturally, e.g. between carat and cut).

## Notes for audit

- All figure-generation calls match the exact function signatures specified
  in the exercise sheet (e.g. `sns.heatmap(..., annot=True, fmt=".2f",
  cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.5)`).
- Every printed number was verified against the exercise sheet's expected
  output and matches exactly.
- One spec note: Exercise 5's own expected output says "dropped: 23"
  rows, while the Exercise 6 prompt text says to cite "22" dropped rows.
  The actual computed value (and the one used in the summary paragraph) is
  **23**, consistent with the code's real output — this is a minor
  inconsistency in the spec itself, not a bug.

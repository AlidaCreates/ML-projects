# cali-scope

## California Housing EDA

This repository contains a complete exploratory data analysis on the 1990 California Housing dataset. The analysis follows the full four-stage EDA pipeline: inspection, univariate, bivariate, multivariate, and geographic synthesis.

### Authors

- Alida Mirmagaliyeva [#amirmaga]
- Aryn Sagynbek [#arsagynbek]

### What is included

- `housing_eda.py` — loads the dataset from the public URL, inspects the raw data, fills missing values in `total_bedrooms`, and creates all expected plots.
- `test_housing_eda.py` — pytest suite verifying that `clean_housing` fills missing values, preserves shape, and is idempotent.
- `requirements.txt` — required packages for the project.
- `.gitignore` — common Python ignore rules.
- `plots/` — generated PNG outputs for the EDA deliverables.

### Key findings

- The dataset contains `20,640` rows and `10` columns.
- There are `207` missing values in `total_bedrooms`.
- `median_house_value` is artificially capped at `$500,001`, affecting `965` rows.
- `median_income` is the strongest predictor of `median_house_value` with correlation `0.6881`.
- Median values by proximity are:
  - `INLAND = 108500`
  - `<1H OCEAN = 214850`
  - `NEAR OCEAN = 229450`
  - `NEAR BAY = 233800`
  - `ISLAND = 414700`
- Four size-related numeric features (`total_rooms`, `total_bedrooms`, `population`, `households`) are highly redundant and form a cluster that should be addressed with feature engineering before modeling.
- Right-skewed features include `total_rooms`, `total_bedrooms`, `population`, `households`, and `median_income`, so any model should consider transformation or capping for stability.

### Running the analysis

1. Create and activate a local virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies inside the venv:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the EDA script:
   ```bash
   python housing_eda.py
   ```
4. Run the tests:
   ```bash
   pytest test_housing_eda.py -v
   ```

If your school Macs do not allow system-wide pip installs, the `.venv` workflow keeps all packages isolated in the repository and avoids modifying the global Python environment.

### Recommendation
Log-transform or cap the four size features before modeling to reduce both skew and multicollinearity; consider replacing them with derived ratios (e.g. rooms-per-household, bedrooms-per-room) instead of feeding all four in raw.

### Conclusion
The right-censoring of median_house_value at $500,001 will distort any model that doesn't explicitly account for censored targets — a regression trained on this data will systematically under-predict prices in the highest segment, since 965 true (higher) values are all flattened to the same $500,001 cap.

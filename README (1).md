# Machine Learning Projects

A collection of hands-on projects covering the full ML workflow: data manipulation with pandas, exploratory analysis, feature engineering, and classical models. Each folder is a self-contained project with its own notebook and notes.

## Projects

### Data manipulation (pandas)

| Project | Focus |
|---------|-------|
| [dataframes](./dataframes) | DataFrame fundamentals: creating, indexing, selecting |
| [filtering-sorting](./filtering-sorting) | Boolean filtering, conditions, sorting |
| [groupby](./groupby) | Grouping and aggregation |
| [joining](./joining) | Merging, joining and concatenating tables |
| [data-cleaning](./data-cleaning) | Missing values, duplicates, type fixes |

### Exploratory data analysis

| Project | Focus |
|---------|-------|
| [eda](./eda) | General EDA workflow: distributions, correlations, visual checks |
| [titanic-data-wrangle](./titanic-data-wrangle) | Wrangling and exploring the Titanic dataset |
| [financial-eda](./financial-eda) | EDA on financial data |
| [cali-scope](./cali-scope) | Scoped analysis on the California dataset |

### Feature engineering

| Project | Focus |
|---------|-------|
| [num-feature-engineering](./num-feature-engineering) | Scaling, transformations, binning of numeric features |
| [categorical-feature](./categorical-feature) | Encoding and handling categorical features |

### Models

| Project | Focus |
|---------|-------|
| [linear-reg](./linear-reg) | Linear regression |
| [logistic-reg](./logistic-reg) | Logistic regression and classification metrics |
| [tree-models](./tree-models) | Decision trees |
| [trees-ensembles](./trees-ensembles) | Random forests and boosting |
| [ml-debugging](./ml-debugging) | Diagnosing and fixing model problems (overfitting, leakage, etc.) |

## Repository Structure

```
.
├── README.md
├── dataframes/
├── filtering-sorting/
├── groupby/
├── joining/
├── data-cleaning/
├── eda/
├── titanic-data-wrangle/
├── financial-eda/
├── cali-scope/
├── num-feature-engineering/
├── categorical-feature/
├── linear-reg/
├── logistic-reg/
├── tree-models/
├── trees-ensembles/
└── ml-debugging/
```

Each project folder contains a notebook (and a README where applicable) with the task, approach, and results.

## Getting Started

```bash
git clone https://github.com/AlidaCreates/ML-projects.git
cd ML-projects

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter lab
```

Open the notebook in the project folder you want to run.

## Tools

- **Language:** Python
- **Data:** pandas, NumPy
- **ML:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Environment:** Jupyter

## Author

**Alida** — cybersecurity student at Astana IT University, building ML skills at the intersection of security and AI.

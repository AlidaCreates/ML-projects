# Machine Learning Projects

A collection of machine learning projects: end-to-end pipelines from data exploration to model evaluation. Each project lives in its own folder and can be run independently.

## Projects

| Project | Description | Stack | Status |
|---------|-------------|-------|--------|
| [01-project-name](./01-project-name) | Short description (task, dataset, key result) | Python, scikit-learn | Done |
| [02-project-name](./02-project-name) | Short description | Python, pandas | In progress |

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── 01-project-name/
│   ├── README.md          # project-specific notes and results
│   ├── data/              # raw and processed data (not tracked if large)
│   ├── notebooks/         # exploration and experiments
│   ├── src/               # reusable pipeline code
│   └── results/           # metrics, plots, saved models
└── 02-project-name/
    └── ...
```

## Typical Pipeline

1. **Data loading and cleaning**: handle missing values, duplicates, types
2. **EDA**: distributions, correlations, target analysis
3. **Feature engineering**: encoding, scaling, feature selection
4. **Modeling**: baseline first, then tuned models
5. **Evaluation**: cross-validation, task-appropriate metrics, error analysis
6. **Reporting**: conclusions and limitations

## Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

Then open the notebook in the project folder you want to run.

## Tools

- **Language:** Python 3.10+
- **Data:** pandas, NumPy
- **ML:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Environment:** Jupyter

## Conventions

- Random seeds are fixed for reproducibility
- Train/test split happens before any fitting (no data leakage)
- Each project README states the metric used and why

## Author

**Alida** — cybersecurity student at Astana IT University, working at the intersection of security and AI/ML.

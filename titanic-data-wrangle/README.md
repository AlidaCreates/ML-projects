# Titanic Dataset Wrangling & Survival Analysis

This project executes an end-to-end data cleaning, feature engineering, and exploratory survival analysis pipeline on the Titanic dataset.

## Data Source

Dataset loaded directly from [datasciencedojo/datasets](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) (891 rows, 12 columns). After cleaning (dropping `Cabin`/`Ticket`, removing 2 rows with missing `Embarked`, imputing missing `Age` with the dataset median of 28.0), the working dataset contains 889 rows.

## Key Findings & Metrics

The analysis evaluated 889 passenger records following cleaning and preprocessing:

- **Overall Survival:** The overall passenger survival rate was **38.25%** (0.3825).
- **Sex Disparity:** Demographics revealed a stark difference in survival outcomes: females achieved a **74.04%** (0.7404) survival rate compared to **18.89%** (0.1889) for males.
- **Socioeconomic Class:** Passenger class served as a strong indicator of survival, with First Class passengers surviving at **62.62%** (0.6262), Second Class at **47.29%** (0.4729), and Third Class at **24.24%** (0.2424).
- **Age Bands:** Children (<12 years) represented the most-survived age category at **57.97%** (0.5797), followed by Teens (**42.86%**), Seniors (**39.69%**), Adults (**35.33%**), and Elders (**19.05%**).
- **IsAlone Effect:** Passengers traveling with family members had a higher survival rate (**50.57%**) compared to those traveling alone (**30.09%**).
- **Port of Embarkation:** Passengers embarking at Cherbourg (France) had a **55.36%** survival rate (n=168), compared to Queenstown (Ireland) at **38.96%** (n=77) and Southampton (England) at **33.70%** (n=644).

## Conclusion

While higher survival rates at Cherbourg suggest geographic or port-level advantages, this observation is largely confounded by passenger class composition, as Cherbourg boardings contained a higher proportion of First Class passengers.

## Requirements

- Python 3.10+
- pandas, numpy, pytest (see `requirements.txt`)

## How to Run

```bash
pip install -r requirements.txt
pytest test_titanic_clean.py -v
python titanic_clean.py
```

## Project Structure

```
raid_wrangle
│   README.md
│   requirements.txt
│   titanic_clean.py
│   test_titanic_clean.py
```
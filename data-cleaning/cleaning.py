import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Exercise 0: Environment and libraries
# ---------------------------------------------------------------------------
def exercise0():
    print("=" * 60)
    print("Exercise 0")
    print("=" * 60)
    print(pd.__version__)

    csv_content = """name,email,age,signup_date,spend
Alice  ,alice@example.com,25,2023-01-15,100.50
BOB,BOB@example.com,30,2023/02/10,200.00
carol,carol@example.com,?,2023-03-22,0.0
Alice  ,alice@example.com,25,2023-01-15,100.50
Dave,dave@example.com,40,March 1 2023,
Eve,eve@example.com,35,2023-05-08,350.75
,ghost@example.com,28,2023-06-17,0.0
Frank,frank@example.com,?,2023-07-04,
Grace,,33,,175.25
henry,henry@example.com,200,2023-09-12,500.00
"""
    with open("signups.csv", "w") as f:
        f.write(csv_content)

    df = pd.read_csv("signups.csv")
    print(df)
    return df


# ---------------------------------------------------------------------------
# Exercise 1: Missing values
# ---------------------------------------------------------------------------
def exercise1(df):
    print("=" * 60)
    print("Exercise 1")
    print("=" * 60)

    print(df.isna().sum())
    print()

    print(df[df.isna().any(axis=1)])
    print()

    print(df.dropna().shape)
    print()

    print(df.dropna(subset=["name"]).shape)
    print()

    filled = df.fillna({"spend": 0.0, "name": "unknown"})
    print(filled[["name", "spend"]])


# ---------------------------------------------------------------------------
# Exercise 2: Duplicates
# ---------------------------------------------------------------------------
def exercise2(df):
    print("=" * 60)
    print("Exercise 2")
    print("=" * 60)

    print(df.duplicated().sum())
    print()

    print(df[df.duplicated()])
    print()

    print(df.drop_duplicates().shape)
    print()

    print(df[df.duplicated(subset="email", keep=False)])
    print()

    print(df.drop_duplicates(subset="email", keep="last").shape)


# ---------------------------------------------------------------------------
# Exercise 3: Dtype conversions
# ---------------------------------------------------------------------------
def exercise3(df):
    print("=" * 60)
    print("Exercise 3")
    print("=" * 60)

    print(df.dtypes)
    print()

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    print(df["age"])
    print()

    df.loc[df["age"] > 100, "age"] = np.nan
    print(df["age"].max())
    print()

    s = pd.Series(["10", "20", "bad", "30"])
    print(pd.to_numeric(s, errors="coerce").tolist())
    try:
        s.astype(int)
    except ValueError as e:
        print("ValueError:", str(e)[:30])

    return df


# ---------------------------------------------------------------------------
# Exercise 4: String cleaning
# ---------------------------------------------------------------------------
def exercise4(df):
    print("=" * 60)
    print("Exercise 4")
    print("=" * 60)

    df["name"] = df["name"].str.strip().str.lower()
    print(df["name"])
    print()

    df["name"] = df["name"].str.title()
    print(df["name"].tolist())
    print()

    print(df[df["email"].str.contains("alice", na=False)])
    print()

    df["email_domain"] = df["email"].str.split("@").str[1]
    print(df[["email", "email_domain"]])
    print()

    df["email"] = df["email"].str.lower()
    print(df["email"].head(3))

    return df


# ---------------------------------------------------------------------------
# Exercise 5: Datetime parsing
# ---------------------------------------------------------------------------
def exercise5(df):
    print("=" * 60)
    print("Exercise 5")
    print("=" * 60)

    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    print(df["signup_date"])
    print()

    df["signup_year"] = df["signup_date"].dt.year
    df["signup_month"] = df["signup_date"].dt.month
    df["signup_day_name"] = df["signup_date"].dt.day_name()
    print(df[["signup_date", "signup_year", "signup_month", "signup_day_name"]])
    print()

    print(df.sort_values("signup_date")[["name", "signup_date"]])
    print()

    mask = (df["signup_date"] >= "2023-03-01") & (df["signup_date"] <= "2023-03-31")
    print(df.loc[mask, ["name", "signup_date"]])

    return df


# ---------------------------------------------------------------------------
# Exercise 6: Cleanup pipeline
# ---------------------------------------------------------------------------
def clean_signups(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = (
        df
        .drop_duplicates()
        .assign(
            age=lambda d: pd.to_numeric(d["age"], errors="coerce").where(lambda x: x <= 100),
            signup_date=lambda d: pd.to_datetime(d["signup_date"], errors="coerce"),
            name=lambda d: d["name"].str.strip().str.lower().str.title(),
            email=lambda d: d["email"].str.lower(),
        )
        .dropna(subset=["name", "email", "signup_date"])
        .reset_index(drop=True)
    )
    return cleaned


def exercise6():
    print("=" * 60)
    print("Exercise 6")
    print("=" * 60)

    df = pd.read_csv("signups.csv")

    clean = clean_signups(df)
    print(clean)
    print()

    print(clean.dtypes)
    print()

    print(f"input rows: {len(df)}  output rows: {len(clean)}  dropped: {len(df) - len(clean)}")


if __name__ == "__main__":
    df = exercise0()
    exercise1(df.copy())
    exercise2(df.copy())
    df = exercise3(df.copy())
    df = exercise4(df.copy())
    df = exercise5(df.copy())
    exercise6()

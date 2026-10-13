import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name":       ["Alice", "Bob", "Carol", "David", "Eve",
                   "Frank", "Grace", "Henry", "Iris", "Jack"],
    "department": ["Engineering", "Sales", "Engineering", "Marketing", "Sales",
                   "Engineering", "Marketing", "Sales", "Engineering", "Sales"],
    "salary":     [85000, 60000, 95000, 70000, 65000,
                   105000, 75000, 58000, 90000, 62000],
    "years":      [3, 7, 5, 2, 4, 8, 3, 1, 6, 5],
    "remote":     [True, False, True, False, True,
                   True, False, False, True, False],
})


# ---------------------------------------------------------------------------
# Exercise 0: Environment and libraries
# ---------------------------------------------------------------------------
def exercise0():
    print(pd.__version__)
    print(df.shape)


# ---------------------------------------------------------------------------
# Exercise 1: Single-column groupby
# ---------------------------------------------------------------------------
def exercise1():
    print(df.groupby("department")["salary"].mean())

    print(df.groupby("department").size())
    print(df.groupby("department")["name"].count())

    print(df.groupby("department")["salary"].max())

    print(df.groupby("department", as_index=False)["salary"].mean())


# ---------------------------------------------------------------------------
# Exercise 2: Named and multi-column aggregation
# ---------------------------------------------------------------------------
def exercise2():
    print(df.groupby("department")["salary"].agg(["mean", "std"]))

    print(df.groupby("department").agg({"salary": "mean", "years": "max"}))

    result = df.groupby("department").agg(
        mean_salary=("salary", "mean"),
        max_years=("years", "max"),
        n_employees=("name", "count"),
    )
    print(result)

    def salary_range(s):
        return s.max() - s.min()

    print(df.groupby("department")["salary"].agg(salary_range))


# ---------------------------------------------------------------------------
# Exercise 3: Multi-key groupby
# ---------------------------------------------------------------------------
def exercise3():
    print(df.groupby(["department", "remote"])["salary"].mean())

    print(df.groupby(["department", "remote"]).size())

    print(df.groupby(["department", "remote"]).size().unstack(fill_value=0))

    result = df.groupby(["department", "remote"]).agg(
        mean_salary=("salary", "mean"),
        n=("name", "count"),
    )
    print(result)


# ---------------------------------------------------------------------------
# Exercise 4: Transform
# ---------------------------------------------------------------------------
def exercise4():
    df["dept_mean_salary"] = df.groupby("department")["salary"].transform("mean")
    print(df[["name", "department", "salary", "dept_mean_salary"]])

    mean = df.groupby("department")["salary"].transform("mean")
    std = df.groupby("department")["salary"].transform("std")
    df["salary_z"] = (df["salary"] - mean) / std
    print(df[["name", "department", "salary", "salary_z"]].round(4))

    print(df.groupby("department")["salary_z"].mean().round(10))

    df["frac_of_dept_max"] = df.groupby("department")["salary"].transform(lambda s: s / s.max())
    print(df[["name", "department", "salary", "frac_of_dept_max"]].round(4))


# ---------------------------------------------------------------------------
# Exercise 5: value_counts and crosstab
# ---------------------------------------------------------------------------
def exercise5():
    print(df["department"].value_counts())

    print(df["department"].value_counts(normalize=True))

    print(pd.crosstab(df["department"], df["remote"]))

    print(pd.crosstab(df["department"], df["remote"], normalize="index"))

    print(pd.crosstab(df["department"], df["remote"], values=df["salary"], aggfunc="mean"))


# ---------------------------------------------------------------------------
# Exercise 6: Filter and apply
# ---------------------------------------------------------------------------
def exercise6():
    result = df.groupby("department").filter(lambda g: len(g) > 2)
    print(result["department"].value_counts())

    result = df.groupby("department").filter(lambda g: g["salary"].mean() > 70000)
    print(result["department"].value_counts())

    # NOTE: pandas versions differ on whether the grouping column
    # ("department") stays inside each group's sub-DataFrame when passed to
    # .apply(). Handle both cases: if the column already exists after
    # apply(), just drop the redundant index level; if not, restore it from
    # the index level.
    gb = df.groupby("department", group_keys=True)
    result = gb.apply(lambda g: g.nlargest(2, "salary"))
    if "department" in result.columns:
        result = result.reset_index(level=0, drop=True)
    else:
        result = result.reset_index(level=0)
    result = result.reset_index(drop=True)
    print(result[["name", "department", "salary"]])

    transform_len = len(df.groupby("department")["salary"].transform("mean"))
    apply_len = len(df.groupby("department")["salary"].apply("mean"))
    print(f"transform length: {transform_len}")
    print(f"apply length:     {apply_len}")


if __name__ == "__main__":
    print("=== Exercise 0 ===")
    exercise0()

    print("=== Exercise 1 ===")
    exercise1()

    print("=== Exercise 2 ===")
    exercise2()

    print("=== Exercise 3 ===")
    exercise3()

    print("=== Exercise 4 ===")
    exercise4()

    print("=== Exercise 5 ===")
    exercise5()

    print("=== Exercise 6 ===")
    exercise6()
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
# Exercise 1: Boolean indexing
# ---------------------------------------------------------------------------
def exercise1():
    mask = df["salary"] > 70000
    print(mask)

    print(df[mask])

    print(df[df["salary"] > 70000])

    print((df["department"] == "Engineering").sum())

    print((df["years"] > 7).any())
    print((df["years"] > 1).all())


# ---------------------------------------------------------------------------
# Exercise 2: Combined conditions
# ---------------------------------------------------------------------------
def exercise2():
    print(df[(df["department"] == "Engineering") & (df["salary"] > 90000)])

    print(df[(df["department"] == "Sales") | (df["department"] == "Marketing")])

    print(df[~df["remote"]])

    eligible = df[df["remote"] & (df["years"] >= 3)]
    print(eligible[["name", "years", "remote"]])

    # Demonstrate operator precedence: & binds tighter than >=, so this
    # evaluates as (df["remote"] & df["years"]) >= 3, NOT
    # df["remote"] & (df["years"] >= 3). No exception is raised, but the
    # result is silently wrong (an all-False mask -> empty DataFrame).
    result = df[df["remote"] & df["years"] >= 3]
    print(result)


# ---------------------------------------------------------------------------
# Exercise 3: isin and between
# ---------------------------------------------------------------------------
def exercise3():
    target_depts = ["Engineering", "Marketing"]
    print(df[df["department"].isin(target_depts)][["name", "department"]])

    print(df[~df["department"].isin(target_depts)][["name", "department"]])

    print(df[df["salary"].between(60000, 80000)][["name", "salary"]])

    print(df[df["salary"].between(60000, 80000, inclusive="neither")][["name", "salary"]])


# ---------------------------------------------------------------------------
# Exercise 4: query
# ---------------------------------------------------------------------------
def exercise4():
    print(df.query("department == 'Engineering' and salary > 90000"))

    threshold = 80000
    print(df.query("salary > @threshold")[["name", "salary"]])

    print(df.query("department in ['Engineering', 'Marketing']")[["name", "department"]])

    df_spaced = df.rename(columns={"years": "tenure (years)"})
    print(df_spaced.query("`tenure (years)` >= 5")[["name", "tenure (years)"]])


# ---------------------------------------------------------------------------
# Exercise 5: Sorting
# ---------------------------------------------------------------------------
def exercise5():
    print(df.sort_values("salary")[["name", "salary"]])

    print(df.sort_values("salary", ascending=False)[["name", "salary"]])

    print(df.sort_values(["department", "salary"], ascending=[True, False])[["name", "department", "salary"]])

    print(df.nlargest(3, "salary")[["name", "salary"]])

    print(df.nsmallest(3, "salary")[["name", "salary"]])


# ---------------------------------------------------------------------------
# Exercise 6: apply, map, and assign
# ---------------------------------------------------------------------------
def exercise6():
    def tenure_level(years):
        if years < 3:
            return "Junior"
        if years < 6:
            return "Mid"
        return "Senior"

    df["level"] = df["years"].map(tenure_level)
    print(df[["name", "years", "level"]])

    codes = {"Engineering": "ENG", "Sales": "SLS", "Marketing": "MKT"}
    df["dept_code"] = df["department"].map(codes)
    print(df[["name", "department", "dept_code"]])

    def compute_bonus(row):
        rate = 0.10 if row["department"] == "Engineering" else 0.05
        return row["salary"] * rate

    df["bonus"] = df.apply(compute_bonus, axis=1)
    print(df[["name", "department", "salary", "bonus"]])

    df_with_total = df.assign(total_comp=df["salary"] + df["bonus"])
    print(df_with_total[["name", "salary", "bonus", "total_comp"]])

    df_chained = (
        df
        .assign(total_comp=lambda d: d["salary"] + d["bonus"])
        .assign(comp_band=lambda d: pd.cut(d["total_comp"],
                                           bins=[0, 70_000, 100_000, np.inf],
                                           labels=["low", "mid", "high"]))
    )
    print(df_chained[["name", "total_comp", "comp_band"]])


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

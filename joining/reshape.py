import pandas as pd
import numpy as np

employees = pd.DataFrame({
    "emp_id": [1, 2, 3, 4, 5],
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "dept_id": [10, 20, 10, 30, 50],
})

departments = pd.DataFrame({
    "dept_id":  [10, 20, 30, 40],
    "dept_name": ["Engineering", "Sales", "Marketing", "Finance"],
    "location":  ["NYC", "SF", "LON", "BER"],
})


# ---------------------------------------------------------------------------
# Exercise 0: Environment and libraries
# ---------------------------------------------------------------------------
def exercise0():
    print("=" * 60)
    print("Exercise 0")
    print("=" * 60)
    print(pd.__version__)
    print(employees.shape, departments.shape)


# ---------------------------------------------------------------------------
# Exercise 1: Concatenation
# ---------------------------------------------------------------------------
def exercise1():
    print("=" * 60)
    print("Exercise 1")
    print("=" * 60)

    # Vertical concatenation
    df_a = pd.DataFrame({"city": ["NYC", "SF"],  "pop": [8.3, 0.9]})
    df_b = pd.DataFrame({"city": ["LON", "BER"], "pop": [9.0, 3.7]})
    print(pd.concat([df_a, df_b], ignore_index=True))
    print()

    # Horizontal concatenation
    df_left = pd.DataFrame({"name": ["Alice", "Bob", "Carol"], "salary": [85, 60, 95]})
    df_right = pd.DataFrame({"years": [3, 7, 5], "remote": [True, False, True]})
    print(pd.concat([df_left, df_right], axis=1))
    print()

    # Concatenation does not match by key -> aligns by index
    df_top = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 1, 2])
    df_bot = pd.DataFrame({"y": [10, 20, 30]}, index=[2, 3, 4])
    print(pd.concat([df_top, df_bot], axis=1))
    print()

    # keys= adds an outer index level
    df_eu = pd.DataFrame({"name": ["BER", "LON"], "pop": [3.7, 9.0]})
    df_us = pd.DataFrame({"name": ["NYC", "SF"],  "pop": [8.3, 0.9]})
    print(pd.concat([df_eu, df_us], keys=["EU", "US"]))


# ---------------------------------------------------------------------------
# Exercise 2: Inner and outer merge
# ---------------------------------------------------------------------------
def exercise2():
    print("=" * 60)
    print("Exercise 2")
    print("=" * 60)

    inner = pd.merge(employees, departments, on="dept_id", how="inner")
    print(inner)
    print()
    print(inner.shape)
    print()

    outer = pd.merge(employees, departments, on="dept_id", how="outer")
    print(outer.sort_values("dept_id").reset_index(drop=True))
    print()

    outer_with_indicator = pd.merge(
        employees, departments, on="dept_id", how="outer", indicator=True
    )
    print(outer_with_indicator["_merge"].value_counts())


# ---------------------------------------------------------------------------
# Exercise 3: Left and right merge
# ---------------------------------------------------------------------------
def exercise3():
    print("=" * 60)
    print("Exercise 3")
    print("=" * 60)

    left = pd.merge(employees, departments, on="dept_id", how="left")
    print(left)
    print()

    right = pd.merge(employees, departments, on="dept_id", how="right")
    print(right.sort_values("dept_id").reset_index(drop=True))
    print()

    empl_alt = employees.rename(columns={"dept_id": "department_id"})
    joined = pd.merge(
        empl_alt, departments, left_on="department_id", right_on="dept_id", how="inner"
    )
    print(joined.columns.tolist())
    print()

    left_df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    right_df = pd.DataFrame({"id": [1, 2, 3], "value": [100, 200, 300]})
    print(pd.merge(left_df, right_df, on="id", suffixes=("_left", "_right")))


# ---------------------------------------------------------------------------
# Exercise 4: Pivot
# ---------------------------------------------------------------------------
def exercise4():
    print("=" * 60)
    print("Exercise 4")
    print("=" * 60)

    sales = pd.DataFrame({
        "year":    [2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023],
        "quarter": ["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"],
        "revenue": [100, 150, 120, 180, 110, 180, 140, 200],
    })
    print(sales)
    print()

    print(sales.pivot(index="year", columns="quarter", values="revenue"))
    print()

    dup = pd.concat(
        [sales, pd.DataFrame([{"year": 2022, "quarter": "Q1", "revenue": 999}])],
        ignore_index=True,
    )
    try:
        dup.pivot(index="year", columns="quarter", values="revenue")
    except ValueError as e:
        print("ValueError:", str(e)[:60])
    print()

    print(dup.pivot_table(index="year", columns="quarter", values="revenue", aggfunc="mean"))
    print()

    print(dup.pivot_table(index="year", columns="quarter", values="revenue", aggfunc="sum"))


# ---------------------------------------------------------------------------
# Exercise 5: Melt
# ---------------------------------------------------------------------------
def exercise5():
    print("=" * 60)
    print("Exercise 5")
    print("=" * 60)

    wide = pd.DataFrame({
        "year": [2022, 2023],
        "Q1":   [100, 110],
        "Q2":   [150, 180],
        "Q3":   [120, 140],
        "Q4":   [180, 200],
    })
    print(wide)
    print()

    long = wide.melt(
        id_vars="year", value_vars=["Q1", "Q2", "Q3", "Q4"],
        var_name="quarter", value_name="revenue",
    )
    print(long)
    print()

    roundtrip = long.pivot(index="year", columns="quarter", values="revenue").reset_index()
    roundtrip.columns.name = None
    print(roundtrip.equals(wide))
    print()

    print(wide.melt(id_vars="year").head())


# ---------------------------------------------------------------------------
# Exercise 6: Stack and unstack
# ---------------------------------------------------------------------------
def exercise6():
    print("=" * 60)
    print("Exercise 6")
    print("=" * 60)

    sales = pd.DataFrame({
        "year":    [2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023],
        "quarter": ["Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3", "Q4"],
        "revenue": [100, 150, 120, 180, 110, 180, 140, 200],
    })
    grouped = sales.groupby(["year", "quarter"])["revenue"].sum()
    print(grouped)
    print()

    print(grouped.unstack())
    print()

    print(grouped.unstack(level=0))
    print()

    restacked = grouped.unstack().stack()
    print(restacked.equals(grouped))


if __name__ == "__main__":
    exercise0()
    exercise1()
    exercise2()
    exercise3()
    exercise4()
    exercise5()
    exercise6()

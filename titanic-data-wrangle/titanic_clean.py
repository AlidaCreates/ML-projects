import pandas as pd

TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def load_titanic_data(url: str = TITANIC_URL) -> pd.DataFrame:
    """Загружает исходные данные."""
    return pd.read_csv(url)


def clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """Очищает датасет: удаляет ненужные колонки, заполняет пропуски и меняет типы."""
    cleaned = df.copy()

    # Расчет медианы по всему исходному датасету
    median_age = cleaned["Age"].median()

    # Удаление ненужных колонок и пустых строк по Embarked
    cleaned = cleaned.drop(columns=["Cabin", "Ticket"])
    cleaned = cleaned.dropna(subset=["Embarked"])

    # Заполнение пропусков
    cleaned["Age"] = cleaned["Age"].fillna(median_age)

    # Приведение категориальных типов
    cleaned["Sex"] = cleaned["Sex"].astype("category")
    cleaned["Embarked"] = cleaned["Embarked"].astype("category")

    return cleaned.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Генерирует новые признаки (FamilySize, IsAlone, AgeBand, FareBand)."""
    featured = df.copy()

    featured["FamilySize"] = featured["SibSp"] + featured["Parch"] + 1
    featured["IsAlone"] = (featured["FamilySize"] == 1).astype(int)

    featured["AgeBand"] = pd.cut(
        featured["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Adult", "Senior", "Elder"],
    )

    featured["FareBand"] = pd.qcut(
        featured["Fare"], q=4, labels=["Low", "Mid", "High", "VeryHigh"]
    )

    return featured


def analyze(df: pd.DataFrame) -> None:
    """Выполняет анализ выживаемости пассажиров."""
    print("=== 1. Overall Survival Rate ===")
    overall_survival = df["Survived"].mean()
    print(f"Overall survival rate: {overall_survival:.4f}")

    print("\n=== 2. Per-Sex Survival ===")
    print(df.groupby("Sex", observed=True)["Survived"].mean().round(4))

    print("\n=== 3. Per-Class Survival ===")
    print(df.groupby("Pclass", observed=True)["Survived"].mean().round(4))

    print("\n=== 4. Per-(Sex, Pclass) Survival (Named Aggregation) ===")
    sex_pclass = df.groupby(["Sex", "Pclass"], observed=True).agg(
        survival_rate=("Survived", "mean"), n=("Survived", "size")
    )
    print(sex_pclass)

    print("\n=== 5. Per-AgeBand Survival ===")
    print(df.groupby("AgeBand", observed=True)["Survived"].mean().round(4))

    print("\n=== 6. Per-IsAlone Survival ===")
    print(df.groupby("IsAlone", observed=True)["Survived"].mean().round(4))


def merge_with_ports(df: pd.DataFrame) -> pd.DataFrame:
    """Создает DataFrame с портами, приводимый к category dtype, и объединяет с основным df."""
    ports = pd.DataFrame(
        {
            "Embarked": ["S", "C", "Q"],
            "PortName": ["Southampton", "Cherbourg", "Queenstown"],
            "Country": ["England", "France", "Ireland"],
        }
    )

    # Приведение типов для избежания предупреждений/ошибок Pandas (Categorical merge gotcha)
    ports["Embarked"] = ports["Embarked"].astype(df["Embarked"].dtype)

    # Выполнение Left Join
    merged_df = df.merge(ports, on="Embarked", how="left")

    # Проверка размерности (должно быть 889 строк и 12 столбцов)
    assert merged_df.shape == (889, 16), f"Expected shape (889, 16), got {merged_df.shape}"

    return merged_df


def analyze_ports(df: pd.DataFrame) -> None:
    """Выполняет порт-уровневый анализ выживаемости."""
    print("\n=== Per-Port Survival Rate & Passenger Count ===")
    port_analysis = df.groupby("PortName", observed=True).agg(
        survival_rate=("Survived", "mean"), n=("Survived", "size")
    )
    print(port_analysis.round(4))

    print("\n=== Per-(Country, Sex) Survival Rate (2D Pivot) ===")
    country_sex_pivot = df.pivot_table(
        index="Country", columns="Sex", values="Survived", aggfunc="mean", observed=True
    )
    print(country_sex_pivot.round(4))


def main() -> None:
    raw_df = load_titanic_data()
    cleaned_df = clean_titanic(raw_df)
    featured_df = engineer_features(cleaned_df)

    analyze(featured_df)

    # 6. Объединение с портами и порт-уровневый анализ
    merged_df = merge_with_ports(featured_df)
    analyze_ports(merged_df)


if __name__ == "__main__":
    main()
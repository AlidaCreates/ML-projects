import pytest
import pandas as pd
from titanic_clean import load_titanic_data, clean_titanic


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Loads the raw dataset once for all tests in this module."""
    return load_titanic_data()


@pytest.fixture(scope="module")
def cleaned_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Applies clean_titanic once to the raw dataset."""
    return clean_titanic(raw_df)


def test_clean_drops_cabin_ticket(cleaned_df: pd.DataFrame):
    """Verify Cabin and Ticket are not in the cleaned columns."""
    assert "Cabin" not in cleaned_df.columns
    assert "Ticket" not in cleaned_df.columns


def test_clean_no_missing_age_or_embarked(cleaned_df: pd.DataFrame):
    """Verify cleaned DataFrame has zero missing values."""
    assert cleaned_df.isnull().sum().sum() == 0


def test_clean_shape(cleaned_df: pd.DataFrame):
    """Verify cleaned DataFrame has 889 rows and 10 columns."""
    assert cleaned_df.shape == (889, 10)




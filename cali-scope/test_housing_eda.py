import pandas as pd

from housing_eda import clean_housing, load_housing


def test_clean_no_missing() -> None:
    df = load_housing()
    cleaned = clean_housing(df)
    assert cleaned.isna().sum().sum() == 0


def test_clean_preserves_shape() -> None:
    df = load_housing()
    cleaned = clean_housing(df)
    assert cleaned.shape == df.shape


def test_clean_idempotent() -> None:
    df = load_housing()
    cleaned_once = clean_housing(df)
    cleaned_twice = clean_housing(cleaned_once)
    pd.testing.assert_frame_equal(cleaned_once, cleaned_twice)

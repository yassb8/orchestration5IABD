"""Tests du module src.data."""
import pandas as pd
import pytest

from src.config import TARGET
from src.data import load_data, split
from src.features import encode_target


def test_load_data_returns_dataframe():
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_data_has_target_column():
    df = load_data()
    assert TARGET in df.columns


def test_split_shapes():
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    assert len(x_train) + len(x_test) == len(df)
    assert len(y_train) == len(x_train)
    assert len(y_test) == len(x_test)


def test_split_test_size():
    df = load_data()
    x_train, x_test, _, _ = split(df, test_size=0.2)
    assert abs(len(x_test) / len(df) - 0.2) < 0.01


def test_split_no_target_in_features():
    df = load_data()
    x_train, x_test, _, _ = split(df)
    assert TARGET not in x_train.columns
    assert TARGET not in x_test.columns


def test_encode_target_values():
    df = load_data()
    _, _, _, y_test = split(df)
    encoded = encode_target(y_test)
    assert set(encoded.unique()).issubset({0, 1})


def test_encode_target_dropout_is_one():
    series = pd.Series(["Dropout", "Graduate", "Enrolled"])
    encoded = encode_target(series)
    assert encoded.iloc[0] == 1
    assert encoded.iloc[1] == 0
    assert encoded.iloc[2] == 0

"""Tests du module src.features."""
import numpy as np

from src.config import NUMERIC_FEATURES, TARGET
from src.data import load_data, split
from src.features import build_preprocessor, encode_target


def test_preprocessor_output_shape():
    df = load_data()
    x_train, _, _, _ = split(df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(x_train)
    assert transformed.shape[0] == len(x_train)
    assert transformed.shape[1] == len(NUMERIC_FEATURES)


def test_preprocessor_no_nan():
    df = load_data()
    x_train, _, _, _ = split(df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(x_train)
    assert not np.isnan(transformed).any()


def test_preprocessor_feature_names():
    df = load_data()
    x_train, _, _, _ = split(df)
    preprocessor = build_preprocessor()
    preprocessor.fit(x_train)
    names = preprocessor.get_feature_names_out()
    assert len(names) == len(NUMERIC_FEATURES)


def test_encode_target_no_nulls():
    df = load_data()
    encoded = encode_target(df[TARGET])
    assert encoded.notna().all()


def test_encode_target_binary():
    df = load_data()
    encoded = encode_target(df[TARGET])
    assert set(encoded.unique()).issubset({0, 1})

"""Construction du pré-processing pour la prédiction de l'abandon scolaire."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import NUMERIC_FEATURES, TARGET_MAPPING


def build_preprocessor() -> ColumnTransformer:
    """Retourne le ColumnTransformer : imputation médiane + StandardScaler."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def encode_target(series: pd.Series) -> pd.Series:
    """Convertit la colonne cible (Dropout/Graduate/Enrolled) en 0/1."""
    return series.map(TARGET_MAPPING)

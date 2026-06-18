"""Configuration partagee du suivi MLflow.

Centralise la configuration du tracking pour eviter de la dupliquer dans
chaque script d'entrainement, et ajoute la tracabilite des donnees
(dataset lineage).

Usage :
    from src.tracking import setup_experiment, log_dataset
    setup_experiment()
    log_dataset(df, context="training")
"""
from __future__ import annotations

import logging

import mlflow
import mlflow.data
import pandas as pd

from src.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MLFLOW_TRACKING_URI,
    TARGET,
)

logger = logging.getLogger(__name__)


def setup_experiment() -> None:
    """Configurer le tracking MLflow et les metadonnees de l'experience.

    Idempotent : re-appelable sans erreur.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)
    client = mlflow.MlflowClient()
    if MLFLOW_EXPERIMENT_DESCRIPTION:
        client.set_experiment_tag(
            experiment.experiment_id,
            "mlflow.note.content",
            MLFLOW_EXPERIMENT_DESCRIPTION,
        )
    for key, value in MLFLOW_EXPERIMENT_TAGS.items():
        client.set_experiment_tag(experiment.experiment_id, key, value)
    logger.info(
        "Experiment MLflow configure : '%s' (id=%s)",
        MLFLOW_EXPERIMENT,
        experiment.experiment_id,
    )


def log_dataset(df: pd.DataFrame, context: str, name: str = "dataset") -> None:
    """Logger un dataset MLflow dans le run courant (tracabilite donnees -> modele).

    Parameters
    ----------
    df : pandas.DataFrame
        Donnees a referencer (features + cible).
    context : str
        Role du dataset dans le run : "training" ou "evaluation".
    name : str, optional
        Nom logique du dataset, par defaut "dataset".
    """
    dataset = mlflow.data.from_pandas(  # type: ignore[attr-defined]
        df, source=str(DATA_PATH), targets=TARGET, name=name
    )
    mlflow.log_input(dataset, context=context)

"""DAG Airflow - pipeline de re-entrainement du modele.

Seance 17 - TP Airflow
    Pipeline : validation des donnees -> entrainement baseline -> controle qualite.
    Planifie tous les lundis a 3h du matin.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.65

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    """Valider que le dataset est present et afficher ses statistiques."""
    import pandas as pd

    from src.config import DATA_PATH, TARGET

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logger.info("Dataset charge : %d lignes, %d colonnes", len(df), len(df.columns))
    logger.info("Repartition cible :\n%s", df[TARGET].value_counts().to_string())
    logger.info("Valeurs manquantes : %d", df.isnull().sum().sum())


def task_train(**context) -> None:
    """Entrainer la baseline LogReg et pousser les metriques dans XCom."""
    from src.train import train

    metrics = train()
    logger.info("Entrainement termine : f1=%.3f  roc_auc=%.3f", metrics["f1"], metrics["roc_auc"])
    context["ti"].xcom_push(key="f1", value=metrics["f1"])
    context["ti"].xcom_push(key="roc_auc", value=metrics["roc_auc"])


def task_check_quality(**context) -> None:
    """Verifier que le F1 depasse le seuil minimal."""
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    if f1 < QUALITY_THRESHOLD:
        raise ValueError(
            f"Porte qualite echouee : f1={f1:.3f} < seuil={QUALITY_THRESHOLD}"
        )
    logger.info("Porte qualite validee : f1=%.3f >= seuil=%.3f", f1, QUALITY_THRESHOLD)


with DAG(
    dag_id="model_retraining",
    description="Valide les donnees, reentraine le modele et controle sa qualite",
    schedule="0 3 * * 1",  # tous les lundis a 3h
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    prepare >> train_task >> check

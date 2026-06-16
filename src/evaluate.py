"""Evaluation automatisee et validation du modele.

Seance 11 - TP Tests Donnees & Modele
    `mlflow.models.evaluate` calcule metriques et artefacts (matrice de
    confusion, courbes ROC / precision-rappel) sur le jeu de test.
    `mlflow.validate_evaluation_results` applique ensuite une porte qualite :
    exception si une metrique passe sous son seuil.
    Pre-requis : un modele enregistre au Model Registry (make train-models).

Lancement :
    python -m src.evaluate                       # derniere version du registry
    python -m src.evaluate --model-uri models:/dropout-classifier/1
    python -m src.evaluate --no-validate         # evalue sans porte qualite
"""
from __future__ import annotations

import argparse
import logging

import mlflow
import mlflow.data
import mlflow.models
from mlflow.exceptions import MlflowException
from mlflow.models import MetricThreshold

from src.config import (
    DATA_PATH,
    EVAL_F1_MIN,
    EVAL_ROC_AUC_MIN,
    MODEL_NAME,
    TARGET,
)
from src.data import load_data, split
from src.features import encode_target
from src.tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def latest_model_uri() -> str:
    """Resoudre l'URI de la derniere version enregistree de MODEL_NAME."""
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"Aucune version enregistree pour '{MODEL_NAME}'. "
            "Lancez d'abord un entrainement (make train-models)."
        )
    latest = max(versions, key=lambda v: int(v.version))
    return f"models:/{MODEL_NAME}/{latest.version}"


def build_thresholds() -> dict[str, MetricThreshold]:
    """Construire les seuils de validation a partir de la configuration."""
    return {
        "roc_auc": MetricThreshold(threshold=EVAL_ROC_AUC_MIN, greater_is_better=True),
        "f1_score": MetricThreshold(threshold=EVAL_F1_MIN, greater_is_better=True),
    }


def evaluate_model(model_uri: str | None = None, validate: bool = True):
    """Evaluer un modele du registry et, optionnellement, valider les seuils.

    Parameters
    ----------
    model_uri : str, optional
        URI MLflow du modele. Par defaut, la derniere version de MODEL_NAME.
    validate : bool, optional
        Applique la porte qualite, par defaut True.

    Returns
    -------
    mlflow.models.EvaluationResult
    """
    df = load_data()
    _, x_test, _, y_test = split(df)
    # Encode la cible (Dropout/Graduate/Enrolled -> 0/1) pour mlflow.models.evaluate
    y_test_enc = encode_target(y_test)
    eval_df = x_test.copy()
    eval_df[TARGET] = y_test_enc.values

    setup_experiment()
    model_uri = model_uri or latest_model_uri()
    logger.info("Evaluation de %s", model_uri)

    with mlflow.start_run(run_name="evaluate"):
        log_dataset(eval_df, context="evaluation", name="students-dropout-eval")

        result = mlflow.models.evaluate(
            model_uri,
            data=eval_df,
            targets=TARGET,
            model_type="classifier",
            evaluators=["default"],
        )
        logger.info(
            "f1_score=%.3f  roc_auc=%.3f",
            result.metrics["f1_score"],
            result.metrics["roc_auc"],
        )

        if validate:
            mlflow.validate_evaluation_results(build_thresholds(), result)

    return result


def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-uri",
        default=None,
        help="URI du modele a evaluer (defaut: derniere version de MODEL_NAME)",
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Evalue sans appliquer la porte qualite (seuils)",
    )
    args = parser.parse_args()

    try:
        evaluate_model(model_uri=args.model_uri, validate=args.validate)
    except MlflowException as exc:
        logger.error("Validation echouee : %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

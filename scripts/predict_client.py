"""Client de test pour l'API FastAPI de prediction de l'abandon scolaire.

Envoie quelques payloads de test a une instance locale de l'API
(`make api`) et affiche les reponses de /health, /predict et /model-info.
Les payloads sont echantillonnes directement dans le dataset.

Lancement (depuis la racine du projet) :
    uv run python scripts/predict_client.py
    uv run python scripts/predict_client.py --url http://127.0.0.1:8000
    uv run python scripts/predict_client.py --n 5
"""
from __future__ import annotations

import argparse
import json
import logging

import httpx

from src.config import API_URL, TARGET
from src.data import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_SAMPLES = 3


def build_payloads(n: int = N_SAMPLES) -> list[dict]:
    """Echantillonner n lignes du dataset comme payloads de test."""
    features = load_data().drop(columns=[TARGET])
    sample = features.sample(n=n)
    return [json.loads(row.to_json()) for _, row in sample.iterrows()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=API_URL, help="URL de base de l'API (defaut: %(default)s)")
    parser.add_argument("--n", type=int, default=N_SAMPLES, help="Nombre de requetes /predict")
    args = parser.parse_args()

    payloads = build_payloads(args.n)

    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        health = client.get("/health")
        logger.info("GET /health      -> %s %s", health.status_code, health.json())

        for i, payload in enumerate(payloads):
            response = client.post("/predict", json=payload)
            logger.info("POST /predict (#%d) -> %s %s", i, response.status_code, response.json())

        info = client.get("/model-info")
        logger.info("GET /model-info  -> %s %s", info.status_code, info.json())


if __name__ == "__main__":
    main()

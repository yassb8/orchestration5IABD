"""Configuration centrale — Prédiction de l'abandon scolaire.

C'est le SEUL fichier à adapter pour brancher le jeu de données :
data.py, features.py et les scripts d'entraînement lisent toutes leurs
colonnes via ces constantes.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# src/config.py → parents[0] = src/ → parents[1] = racine du projet
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_PATH = ROOT / "data" / "raw" / "students_dropout_academic_success.csv"
MODEL_DIR = ROOT / "models"

# Colonne cible brute (strings) et mapping vers binaire
# 1 = Dropout (abandon), 0 = Graduate ou Enrolled (non-abandon)
TARGET = "target"
TARGET_MAPPING: dict[str, int] = {"Dropout": 1, "Graduate": 0, "Enrolled": 0}

# Toutes les colonnes sont déjà encodées numériquement dans le CSV (int64/float64)
NUMERIC_FEATURES: list[str] = [
    # Profil socio-démographique
    "Marital Status",
    "Age at enrollment",
    "Gender",
    "Nacionality",
    "International",
    "Displaced",
    "Educational special needs",
    # Parcours académique antérieur
    "Previous qualification",
    "Previous qualification (grade)",
    "Admission grade",
    # Inscription
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    # Situation financière
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    # Qualifications et professions des parents
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    # Résultats 1er semestre
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    # Résultats 2ème semestre
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    # Contexte macroéconomique
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Pas de colonnes catégorielles (string) dans ce dataset — tout est déjà encodé
CATEGORICAL_FEATURES: list[str] = []

RANDOM_STATE = 42

# Surcouche via variables d'environnement (principe 12-factor)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "student-dropout")
MODEL_NAME = os.getenv("MODEL_NAME", "dropout-classifier")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Seuils minimaux de validation (make evaluate)
EVAL_ROC_AUC_MIN: float = float(os.getenv("EVAL_ROC_AUC_MIN", "0.80"))
EVAL_F1_MIN: float = float(os.getenv("EVAL_F1_MIN", "0.70"))

MLFLOW_EXPERIMENT_DESCRIPTION = (
    "Prédiction de l'abandon scolaire (Dropout vs Graduate/Enrolled). "
    "Comparaison RF / XGBoost / LightGBM optimisés par GridSearchCV."
)
MLFLOW_EXPERIMENT_TAGS: dict[str, str] = {
    "project": "student-dropout",
    "dataset": "students_dropout_academic_success",
    "task": "binary-classification",
    "team": "5IABD1",
}

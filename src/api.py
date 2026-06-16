"""API d'inference pour la prediction de l'abandon scolaire (FastAPI).

Seance 12 - TP FastAPI
    Charge le modele une seule fois au demarrage (lifespan), expose /health,
    /predict et /model-info.
    Lancement : `uvicorn src.api:app --reload`
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Modele introuvable : {model_path}. Lancez d'abord make train-models.")
    ml["model"] = joblib.load(model_path)
    logger.info("Modele charge depuis %s", model_path)
    yield
    ml.clear()


app = FastAPI(
    title="Student Dropout Prediction API",
    version="0.1.0",
    description="Predit si un etudiant va abandonner (1) ou non (0).",
    lifespan=lifespan,
)


class Features(BaseModel):
    """Variables d'entree du modele (memes colonnes que NUMERIC_FEATURES)."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "Marital Status": 1,
                    "Age at enrollment": 20,
                    "Gender": 1,
                    "Nacionality": 1,
                    "International": 0,
                    "Displaced": 0,
                    "Educational special needs": 0,
                    "Previous qualification": 1,
                    "Previous qualification (grade)": 122.0,
                    "Admission grade": 127.3,
                    "Application mode": 1,
                    "Application order": 1,
                    "Course": 9254,
                    "Daytime/evening attendance": 1,
                    "Debtor": 0,
                    "Tuition fees up to date": 1,
                    "Scholarship holder": 0,
                    "Mother's qualification": 1,
                    "Father's qualification": 1,
                    "Mother's occupation": 5,
                    "Father's occupation": 5,
                    "Curricular units 1st sem (credited)": 0,
                    "Curricular units 1st sem (enrolled)": 6,
                    "Curricular units 1st sem (evaluations)": 6,
                    "Curricular units 1st sem (approved)": 6,
                    "Curricular units 1st sem (grade)": 13.5,
                    "Curricular units 1st sem (without evaluations)": 0,
                    "Curricular units 2nd sem (credited)": 0,
                    "Curricular units 2nd sem (enrolled)": 6,
                    "Curricular units 2nd sem (evaluations)": 6,
                    "Curricular units 2nd sem (approved)": 6,
                    "Curricular units 2nd sem (grade)": 13.5,
                    "Curricular units 2nd sem (without evaluations)": 0,
                    "Unemployment rate": 10.8,
                    "Inflation rate": 1.4,
                    "GDP": 1.74,
                }
            ]
        },
    )

    # Profil socio-demographique
    marital_status: int = Field(..., alias="Marital Status", ge=1)
    age_at_enrollment: int = Field(..., alias="Age at enrollment", ge=0)
    gender: int = Field(..., alias="Gender", ge=0, le=1)
    nacionality: int = Field(..., alias="Nacionality", ge=0)
    international: int = Field(..., alias="International", ge=0, le=1)
    displaced: int = Field(..., alias="Displaced", ge=0, le=1)
    educational_special_needs: int = Field(..., alias="Educational special needs", ge=0, le=1)

    # Parcours academique anterieur
    previous_qualification: int = Field(..., alias="Previous qualification", ge=0)
    previous_qualification_grade: float = Field(..., alias="Previous qualification (grade)", ge=0)
    admission_grade: float = Field(..., alias="Admission grade", ge=0)

    # Inscription
    application_mode: int = Field(..., alias="Application mode", ge=0)
    application_order: int = Field(..., alias="Application order", ge=0, le=9)
    course: int = Field(..., alias="Course", ge=0)
    daytime_evening_attendance: int = Field(..., alias="Daytime/evening attendance", ge=0, le=1)

    # Situation financiere
    debtor: int = Field(..., alias="Debtor", ge=0, le=1)
    tuition_fees_up_to_date: int = Field(..., alias="Tuition fees up to date", ge=0, le=1)
    scholarship_holder: int = Field(..., alias="Scholarship holder", ge=0, le=1)

    # Qualifications et professions des parents
    mothers_qualification: int = Field(..., alias="Mother's qualification", ge=0)
    fathers_qualification: int = Field(..., alias="Father's qualification", ge=0)
    mothers_occupation: int = Field(..., alias="Mother's occupation", ge=0)
    fathers_occupation: int = Field(..., alias="Father's occupation", ge=0)

    # Resultats 1er semestre
    cu_1st_credited: int = Field(..., alias="Curricular units 1st sem (credited)", ge=0)
    cu_1st_enrolled: int = Field(..., alias="Curricular units 1st sem (enrolled)", ge=0)
    cu_1st_evaluations: int = Field(..., alias="Curricular units 1st sem (evaluations)", ge=0)
    cu_1st_approved: int = Field(..., alias="Curricular units 1st sem (approved)", ge=0)
    cu_1st_grade: float = Field(..., alias="Curricular units 1st sem (grade)", ge=0)
    cu_1st_without_evaluations: int = Field(..., alias="Curricular units 1st sem (without evaluations)", ge=0)

    # Resultats 2eme semestre
    cu_2nd_credited: int = Field(..., alias="Curricular units 2nd sem (credited)", ge=0)
    cu_2nd_enrolled: int = Field(..., alias="Curricular units 2nd sem (enrolled)", ge=0)
    cu_2nd_evaluations: int = Field(..., alias="Curricular units 2nd sem (evaluations)", ge=0)
    cu_2nd_approved: int = Field(..., alias="Curricular units 2nd sem (approved)", ge=0)
    cu_2nd_grade: float = Field(..., alias="Curricular units 2nd sem (grade)", ge=0)
    cu_2nd_without_evaluations: int = Field(..., alias="Curricular units 2nd sem (without evaluations)", ge=0)

    # Contexte macroeconomique
    unemployment_rate: float = Field(..., alias="Unemployment rate")
    inflation_rate: float = Field(..., alias="Inflation rate")
    gdp: float = Field(..., alias="GDP")


class PredictionOut(BaseModel):
    prediction: int = Field(..., description="Classe predite : 1 = Dropout, 0 = Graduate/Enrolled")
    probability: float = Field(..., description="Probabilite d'abandon (classe 1)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    row = pd.DataFrame([features.model_dump(by_alias=True)])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


@app.get("/model-info")
def model_info() -> dict:
    return {"version": os.environ.get("MODEL_VERSION", "unknown")}

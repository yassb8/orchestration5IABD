"""Tests de l'API FastAPI (sans modele entraine — mock complet du lifespan)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


SAMPLE_PAYLOAD = {
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


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    mock_model.predict.return_value = np.array([1])

    # Mocker MODEL_DIR / "model.joblib" pour que .exists() renvoie True
    mock_model_path = MagicMock()
    mock_model_path.exists.return_value = True
    mock_model_dir = MagicMock()
    mock_model_dir.__truediv__ = MagicMock(return_value=mock_model_path)

    import src.api as api_module
    api_module.predictions_log.clear()

    with patch("src.api.MODEL_DIR", mock_model_dir), \
         patch("src.api.joblib.load", return_value=mock_model):
        from src.api import app
        with TestClient(app) as c:
            yield c

    api_module.predictions_log.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_200(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_schema(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in {0, 1}
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_dropout(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    data = response.json()
    assert data["prediction"] == 1
    assert data["probability"] == pytest.approx(0.7, abs=1e-3)


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "version" in response.json()


def test_predictions_log_empty_at_start(client):
    response = client.get("/predictions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_predictions_log_grows(client):
    client.post("/predict", json=SAMPLE_PAYLOAD)
    client.post("/predict", json=SAMPLE_PAYLOAD)
    response = client.get("/predictions")
    assert len(response.json()) >= 2

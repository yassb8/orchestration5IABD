# ==============================================================================
# Prédiction de l'abandon scolaire — Makefile
# ==============================================================================
# Dataset : data/raw/students_dropout_academic_success.csv
# Cible   : 1 = Dropout, 0 = Graduate ou Enrolled
# Aide    : make help
# ==============================================================================

SHELL        := /bin/sh
PYTHON       := uv run python
RUN          := uv run
VENV_DIR     := .venv
DATA_RAW     := data/raw/students_dropout_academic_success.csv
PYTHONPATH   ?= $(CURDIR)
export PYTHONPATH
API_HOST     ?= 127.0.0.1
API_PORT     ?= 8000
FRONTEND_PORT ?= 8501
MLFLOW_PORT  := 5000
C            ?= 1.0
MAX_ITER     ?= 1000
CV           ?= 5
SCORING      ?= roc_auc
N_TRIALS     ?= 30

# Couleurs ANSI
YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help \
        check-uv check-venv install sync lock reset-env doctor \
        data train train-models train-optuna evaluate mlflow api frontend \
        docker-build docker-run docker-up docker-down \
        lint format type test check


# ==============================================================================
# Help
# ==============================================================================

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-16s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)


# ==============================================================================
# Setup — Installation de l'environnement Python (uv + pyproject.toml)
# ==============================================================================

check-uv: ## Vérifie que uv est installé
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)[ERREUR] uv n'est pas installé$(RESET)"; \
		echo "  Installation : https://docs.astral.sh/uv/"; \
		exit 1; \
	}

check-venv: ## Vérifie que le virtualenv existe
	@test -d $(VENV_DIR) || { \
		echo "$(RED)[ERREUR] Virtualenv manquant : $(VENV_DIR)$(RESET)"; \
		echo "  Lance : make install"; \
		exit 1; \
	}

install: check-uv ## Crée le venv et installe le projet + dépendances dev
	@echo "$(YELLOW)>> Installation des dépendances...$(RESET)"
	uv sync --extra dev
	@echo "$(GREEN)[OK] Dépendances installées$(RESET)"

sync: install ## Alias de install (synchronise les dépendances)

lock: check-uv ## Génère/actualise uv.lock depuis pyproject.toml
	@echo "$(YELLOW)>> Génération du lockfile...$(RESET)"
	uv lock
	@echo "$(GREEN)[OK] uv.lock généré$(RESET)"

reset-env: check-uv ## Réinitialise l'environnement (.venv + uv.lock)
	@echo "$(YELLOW)>> Réinitialisation de l'environnement...$(RESET)"
	rm -rf $(VENV_DIR) uv.lock
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement recréé$(RESET)"

doctor: check-uv check-venv ## Diagnostique l'environnement de travail
	@uv --version
	@$(PYTHON) --version
	@test -f $(DATA_RAW) && echo "$(GREEN)[OK] Dataset trouvé : $(DATA_RAW)$(RESET)" \
		|| echo "$(RED)[ERREUR] Dataset manquant : $(DATA_RAW)$(RESET)"
	@echo "$(GREEN)[OK] Environnement prêt$(RESET)"


# ==============================================================================
# Pipeline ML
# ==============================================================================

data: ## Inspecte et valide le dataset brut (data/raw/)
	# TODO (S0) : compléter src/config.py avec les colonnes du dataset
	@echo "$(YELLOW)>> Inspection du dataset...$(RESET)"
	@$(PYTHON) -c "\
import pandas as pd; \
df = pd.read_csv('$(DATA_RAW)'); \
print(f'Lignes : {len(df)}, Colonnes : {len(df.columns)}'); \
print(f'Répartition cible :\n{df[\"target\"].value_counts()}'); \
print(f'Valeurs manquantes : {df.isnull().sum().sum()}'); \
"
	@echo "$(GREEN)[OK] Dataset valide$(RESET)"

train: ## Entraîne la baseline LogReg -> models/model.joblib (C=.. MAX_ITER=..)
	$(PYTHON) -m src.train --c $(C) --max-iter $(MAX_ITER)

train-models: ## Compare RF / XGBoost / LightGBM (GridSearchCV) + SHAP (CV=.. SCORING=..)
	$(PYTHON) -m src.train_models --cv $(CV) --scoring $(SCORING)

train-optuna: ## Optimise les hyperparamètres avec Optuna (N_TRIALS=.. CV=..)
	$(PYTHON) -m src.train_optuna --n-trials $(N_TRIALS) --cv $(CV)

evaluate: ## Evalue le meilleur modele du registry et applique la porte qualite
	$(PYTHON) -m src.evaluate

mlflow: ## Démarre le serveur MLflow via Docker Compose (port 5000)
	# TODO (S5) : docker compose -f docker-compose.yml up -d mlflow

api: ## Lance l'API FastAPI en rechargement auto (API_HOST / API_PORT)
	$(RUN) uvicorn src.api:app --reload --host $(API_HOST) --port $(API_PORT)

frontend: ## Lance le frontend Streamlit (FRONTEND_PORT / API_URL)
	# TODO (S14bis) : $(RUN) streamlit run frontend/app.py --server.port $(FRONTEND_PORT)


# ==============================================================================
# Docker
# ==============================================================================

docker-build: ## Construit l'image d'entraînement
	# TODO (S8) : docker build -f docker/Dockerfile.train -t dropout-train .

docker-run: ## Lance l'entraînement dans un conteneur
	# TODO (S8) : docker run --rm -v "$(CURDIR)/models:/app/models" dropout-train

docker-up: ## Démarre la stack complète (mlflow, api, frontend)
	# TODO (S14) : docker compose -f docker-compose.yml up -d --build mlflow api frontend

docker-down: ## Arrête et supprime les conteneurs (conserve les volumes)
	# TODO (S14) : docker compose -f docker-compose.yml down


# ==============================================================================
# Qualité du code
# ==============================================================================

lint: ## Vérifie le style (ruff)
	# TODO : $(RUN) ruff check src

format: ## Formate le code (ruff)
	# TODO : $(RUN) ruff format src

type: ## Vérifie les types (mypy)
	# TODO : $(RUN) mypy src

test: ## Lance les tests (pytest)
	# TODO : $(RUN) pytest

check: lint type test ## Workflow qualité complet (lint + types + tests)

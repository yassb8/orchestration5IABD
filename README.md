# Prédiction de l'abandon scolaire — Pipeline MLOps

## Problématique

**Objectif :** Prédire si un étudiant va **abandonner ses études** avant la fin de son cursus.

- **Cible = 1** : l'étudiant abandonne (`Dropout`)
- **Cible = 0** : l'étudiant termine ou est encore inscrit (`Graduate` ou `Enrolled`)

### Pourquoi ce problème ?

L'abandon scolaire représente un coût humain et économique majeur pour les établissements
d'enseignement supérieur. Identifier en avance les étudiants à risque permet de :

- **Intervenir tôt** — proposer un accompagnement personnalisé avant qu'il ne soit trop tard
- **Optimiser les ressources** — cibler l'aide pédagogique vers les profils les plus vulnérables
- **Améliorer le taux de diplomation** — réduire le décrochage par une détection proactive

### Pourquoi la classification binaire ?

Le dataset original distingue trois états : `Dropout`, `Graduate`, `Enrolled`. En regroupant
`Graduate` et `Enrolled` dans la classe **0**, on se concentre sur la **détection des abandons**,
le seul état actionnable pour une cellule de suivi pédagogique. Le modèle répond à la question :
*"Cet étudiant risque-t-il de quitter son cursus ?"*

Le déséquilibre de classes est géré par un scoring adapté (`roc_auc` ou `f1`).

---

## Données

**Source :**
[UC Irvine ML Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

**Fichier :** `data/raw/students_dropout_academic_success.csv`

| Propriété | Valeur |
|-----------|--------|
| Lignes | 4 424 |
| Colonnes | 37 (36 features + 1 cible) |
| Format | CSV (séparateur `,`) |
| Cible brute | `target` : `Dropout` / `Graduate` / `Enrolled` |
| Cible encodée | `1` = Dropout, `0` = Graduate ou Enrolled |

### Catégories de features

| Catégorie | Exemples de colonnes |
|-----------|---------------------|
| Profil socio-démographique | `Gender`, `Age at enrollment`, `Nacionality`, `Marital Status` |
| Parcours académique antérieur | `Previous qualification`, `Admission grade` |
| Situation financière | `Tuition fees up to date`, `Debtor`, `Scholarship holder` |
| Résultats 1er semestre | `Curricular units 1st sem (approved)`, `Curricular units 1st sem (grade)` |
| Résultats 2ème semestre | `Curricular units 2nd sem (approved)`, `Curricular units 2nd sem (grade)` |
| Contexte macroéconomique | `Unemployment rate`, `Inflation rate`, `GDP` |

---

## Stack technique

- **Python 3.13** géré par `uv`
- **MLflow** — tracking des expériences et model registry
- **Optuna** — optimisation bayésienne des hyperparamètres
- **FastAPI + Uvicorn** — API d'inférence du modèle
- **Streamlit** — frontend de test interactif
- **Docker / Docker Compose** — conteneurisation de la stack complète
- **Airflow** — orchestration du ré-entraînement périodique
- **GitHub Actions** — CI/CD

---

## Mise en route rapide

```bash
make install          # installe les dépendances (uv + pyproject.toml)
make doctor           # vérifie l'environnement et le dataset
make data             # inspecte et valide les données brutes
make train            # entraîne la baseline LogReg + MLflow
make train-optuna     # optimisation Optuna (N_TRIALS=30)
make train-models     # comparaison RF / XGBoost / LightGBM + SHAP
make mlflow           # démarre le serveur MLflow (port 5000)
make api              # lance l'API FastAPI (port 8000)
make frontend         # lance le frontend Streamlit (port 8501)
make docker-up        # démarre toute la stack Docker
make check            # lint + types + tests
```

---

## Structure du projet

```
orchestration5IABD/
  README.md                ce fichier
  Makefile                 commandes du projet
  pyproject.toml           dépendances et configuration des outils
  data/
    raw/
      students_dropout_academic_success.csv   dataset brut
  mlproject/               package Python (config, data, train, api...)
  frontend/                application Streamlit
  docker/                  Dockerfiles
  docker-compose.yml       orchestration des services
  dags/                    DAG Airflow de ré-entraînement
  tests/                   tests pytest
```

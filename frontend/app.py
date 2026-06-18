"""Frontend Streamlit — Prediction de l'abandon scolaire."""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://127.0.0.1:5000")
EVAL_ROC_AUC_MIN = float(os.environ.get("EVAL_ROC_AUC_MIN", "0.80"))
EVAL_F1_MIN = float(os.environ.get("EVAL_F1_MIN", "0.70"))

API_PORT = os.environ.get("API_PORT", "8000")
MLFLOW_PORT = os.environ.get("MLFLOW_PORT", "5000")
AIRFLOW_PORT = os.environ.get("AIRFLOW_PORT", "8080")


def _browser_hostname() -> str:
    """Hostname utilise par le navigateur pour joindre Streamlit (ex: localhost,
    192.168.1.10...). En Docker, API_URL/MLFLOW_URL pointent vers des noms de
    service internes (http://api:8000) injoignables depuis le navigateur cote
    hote : on reprend donc le hostname courant et on change juste le port.
    """
    try:
        host_header = st.context.headers.get("Host", "")
    except Exception:
        host_header = ""
    return host_header.split(":")[0] if host_header else "localhost"


_hostname = _browser_hostname()
API_PUBLIC_URL = f"http://{_hostname}:{API_PORT}"
MLFLOW_PUBLIC_URL = f"http://{_hostname}:{MLFLOW_PORT}"
AIRFLOW_PUBLIC_URL = f"http://{_hostname}:{AIRFLOW_PORT}"

st.set_page_config(
    page_title="Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white;
    }
    .hero h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .hero p  { margin: 0.3rem 0 0; opacity: .75; font-size: .95rem; }
    .hero a  { color: #90caf9; text-decoration: none; font-weight: 600; }
    .hero a:hover { text-decoration: underline; }
    .tool-links { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 1.5rem; }
    .tool-link {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 14px; border-radius: 20px; font-size: .85rem;
        font-weight: 600; text-decoration: none; transition: opacity .2s;
    }
    .tool-link:hover { opacity: .8; }
    .link-api     { background: #e8f4fd; color: #1565c0; border: 1.5px solid #90caf9; }
    .link-mlflow  { background: #fce4ec; color: #b71c1c; border: 1.5px solid #ef9a9a; }
    .link-airflow { background: #e8f5e9; color: #1b5e20; border: 1.5px solid #a5d6a7; }
    .result-card {
        border-radius: 12px; padding: 1.4rem 1.8rem;
        margin-top: 1rem; font-size: 1.05rem; font-weight: 600;
    }
    .result-dropout  { background: #ffebee; border-left: 5px solid #e53935; color: #b71c1c; }
    .result-graduate { background: #e8f5e9; border-left: 5px solid #43a047; color: #1b5e20; }
    .stProgress > div > div { background: linear-gradient(90deg, #43a047, #e53935); }
    .section-title {
        font-weight: 700; font-size: .9rem; letter-spacing: .05em;
        text-transform: uppercase; color: #5c6bc0;
        border-bottom: 2px solid #e8eaf6; padding-bottom: 4px; margin-bottom: 8px;
    }
    .feat-badge {
        display: inline-block; background: #e8eaf6; color: #3949ab;
        border-radius: 6px; padding: 2px 8px; margin: 2px; font-size: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

api_url = st.sidebar.text_input("URL de l'API", value=API_URL)

st.markdown(
    """
    <div class="hero">
        <h1>🎓 Student Dropout Prediction</h1>
        <p>Pipeline MLOps · ESGI 5IABD1 · BOUZOUBAA Yassine ·
            <a href="https://github.com/yassb8/orchestration5IABD" target="_blank">🔗 GitHub</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="tool-links">
        <a class="tool-link link-api"     href="{API_PUBLIC_URL}/docs" target="_blank">⚡ API Docs</a>
        <a class="tool-link link-mlflow"  href="{MLFLOW_PUBLIC_URL}"   target="_blank">📊 MLflow</a>
        <a class="tool-link link-airflow" href="{AIRFLOW_PUBLIC_URL}"  target="_blank">💨 Airflow</a>
    </div>
    """,
    unsafe_allow_html=True,
)

predict_tab, history_tab, eval_tab, dataset_tab, artefact_tab = st.tabs([
    "🔮 Prediction", "📋 Historique", "✅ Evaluation", "📚 Dataset", "📈 Artefacts MLflow",
])

# ── Prediction ──────────────────────────────────────────────────────────────────
with predict_tab:
    st.markdown("#### Informations etudiant")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<p class="section-title">Profil socio-demo</p>', unsafe_allow_html=True)
            marital_status = st.number_input("Marital Status", min_value=1, max_value=6, value=1)
            age = st.number_input("Age at enrollment", min_value=17, max_value=70, value=20)
            gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Homme" if x == 1 else "Femme")
            nacionality = st.number_input("Nacionality", min_value=1, value=1)
            international = st.selectbox("International", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
            displaced = st.selectbox("Displaced", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
            special_needs = st.selectbox("Educational special needs", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
            st.markdown('<p class="section-title">Situation financiere</p>', unsafe_allow_html=True)
            debtor = st.selectbox("Debtor", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
            tuition = st.selectbox("Tuition fees up to date", [1, 0], format_func=lambda x: "Oui" if x == 1 else "Non")
            scholarship = st.selectbox("Scholarship holder", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")

        with col2:
            st.markdown('<p class="section-title">Parcours academique</p>', unsafe_allow_html=True)
            prev_qual = st.number_input("Previous qualification", min_value=1, value=1)
            prev_qual_grade = st.number_input("Prev. qualification (grade)", min_value=0.0, max_value=200.0, value=122.0)
            admission_grade = st.number_input("Admission grade", min_value=0.0, max_value=200.0, value=127.3)
            st.markdown('<p class="section-title">Inscription</p>', unsafe_allow_html=True)
            app_mode = st.number_input("Application mode", min_value=1, value=1)
            app_order = st.number_input("Application order", min_value=0, max_value=9, value=1)
            course = st.number_input("Course", min_value=1, value=9254)
            attendance = st.selectbox("Daytime/evening attendance", [1, 0], format_func=lambda x: "Jour" if x == 1 else "Soir")
            st.markdown('<p class="section-title">Parents</p>', unsafe_allow_html=True)
            mother_qual = st.number_input("Mother's qualification", min_value=1, value=1)
            father_qual = st.number_input("Father's qualification", min_value=1, value=1)
            mother_occ = st.number_input("Mother's occupation", min_value=0, value=5)
            father_occ = st.number_input("Father's occupation", min_value=0, value=5)

        with col3:
            st.markdown('<p class="section-title">Resultats 1er semestre</p>', unsafe_allow_html=True)
            cu1_credited = st.number_input("1st sem — credited", min_value=0, value=0)
            cu1_enrolled = st.number_input("1st sem — enrolled", min_value=0, value=6)
            cu1_evaluations = st.number_input("1st sem — evaluations", min_value=0, value=6)
            cu1_approved = st.number_input("1st sem — approved", min_value=0, value=6)
            cu1_grade = st.number_input("1st sem — grade", min_value=0.0, max_value=20.0, value=13.5)
            cu1_no_eval = st.number_input("1st sem — without evaluations", min_value=0, value=0)
            st.markdown('<p class="section-title">Resultats 2eme semestre</p>', unsafe_allow_html=True)
            cu2_credited = st.number_input("2nd sem — credited", min_value=0, value=0)
            cu2_enrolled = st.number_input("2nd sem — enrolled", min_value=0, value=6)
            cu2_evaluations = st.number_input("2nd sem — evaluations", min_value=0, value=6)
            cu2_approved = st.number_input("2nd sem — approved", min_value=0, value=6)
            cu2_grade = st.number_input("2nd sem — grade", min_value=0.0, max_value=20.0, value=13.5)
            cu2_no_eval = st.number_input("2nd sem — without evaluations", min_value=0, value=0)
            st.markdown('<p class="section-title">Macroeconomie</p>', unsafe_allow_html=True)
            unemployment = st.number_input("Unemployment rate", value=10.8)
            inflation = st.number_input("Inflation rate", value=1.4)
            gdp = st.number_input("GDP", value=1.74)

        submitted = st.form_submit_button("🔮 Predire", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "Marital Status": marital_status, "Age at enrollment": age,
            "Gender": gender, "Nacionality": nacionality,
            "International": international, "Displaced": displaced,
            "Educational special needs": special_needs,
            "Previous qualification": prev_qual,
            "Previous qualification (grade)": prev_qual_grade,
            "Admission grade": admission_grade,
            "Application mode": app_mode, "Application order": app_order,
            "Course": course, "Daytime/evening attendance": attendance,
            "Debtor": debtor, "Tuition fees up to date": tuition,
            "Scholarship holder": scholarship,
            "Mother's qualification": mother_qual, "Father's qualification": father_qual,
            "Mother's occupation": mother_occ, "Father's occupation": father_occ,
            "Curricular units 1st sem (credited)": cu1_credited,
            "Curricular units 1st sem (enrolled)": cu1_enrolled,
            "Curricular units 1st sem (evaluations)": cu1_evaluations,
            "Curricular units 1st sem (approved)": cu1_approved,
            "Curricular units 1st sem (grade)": cu1_grade,
            "Curricular units 1st sem (without evaluations)": cu1_no_eval,
            "Curricular units 2nd sem (credited)": cu2_credited,
            "Curricular units 2nd sem (enrolled)": cu2_enrolled,
            "Curricular units 2nd sem (evaluations)": cu2_evaluations,
            "Curricular units 2nd sem (approved)": cu2_approved,
            "Curricular units 2nd sem (grade)": cu2_grade,
            "Curricular units 2nd sem (without evaluations)": cu2_no_eval,
            "Unemployment rate": unemployment, "Inflation rate": inflation, "GDP": gdp,
        }
        try:
            with st.spinner("Calcul en cours..."):
                response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Appel a l'API impossible : {exc}")
        else:
            prediction = result["prediction"]
            probability = result["probability"]
            st.divider()
            col_res1, col_res2 = st.columns([2, 1])
            with col_res1:
                if prediction == 1:
                    st.markdown(
                        '<div class="result-card result-dropout">⚠️ Risque d\'abandon detecte (Dropout)</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="result-card result-graduate">✅ Pas de risque d\'abandon (Graduate / Enrolled)</div>',
                        unsafe_allow_html=True,
                    )
            with col_res2:
                st.metric("Probabilite d'abandon", f"{probability:.1%}")
            st.progress(probability)

# ── Historique ──────────────────────────────────────────────────────────────────
with history_tab:
    st.markdown("#### Historique des previsions")
    st.button("🔄 Actualiser")
    try:
        resp = httpx.get(f"{api_url}/predictions", timeout=5.0)
        resp.raise_for_status()
        rows = resp.json()
    except httpx.HTTPError as exc:
        st.error(f"Impossible de recuperer l'historique : {exc}")
        rows = []

    if not rows:
        st.info("Aucune prediction enregistree pour l'instant.")
    else:
        df = pd.DataFrame(rows)
        df["label"] = df["prediction"].map({1: "⚠️ Dropout", 0: "✅ Graduate/Enrolled"})
        c1, c2, c3 = st.columns(3)
        c1.metric("Total predictions", len(df))
        c2.metric("Dropouts predits", int((df["prediction"] == 1).sum()))
        c3.metric("Non-dropouts predits", int((df["prediction"] == 0).sum()))
        dropout_rate = (df["prediction"] == 1).mean()
        st.markdown(f"**Taux de dropout predit : {dropout_rate:.1%}**")
        st.progress(dropout_rate)
        st.dataframe(
            df[["timestamp", "label", "probability"]].rename(columns={
                "timestamp": "Horodatage", "label": "Prediction", "probability": "Probabilite",
            }),
            use_container_width=True, hide_index=True,
        )

# ── Evaluation ──────────────────────────────────────────────────────────────────
with eval_tab:
    st.markdown("#### Evaluation du modele (porte qualite)")
    st.caption(f"Seuils de validation : ROC AUC ≥ {EVAL_ROC_AUC_MIN} · F1 ≥ {EVAL_F1_MIN}")

    try:
        exp_resp = httpx.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search",
            params={"max_results": 10},
            timeout=5.0,
        )
        exp_resp.raise_for_status()
        eval_experiments = exp_resp.json().get("experiments", [])
    except Exception as exc:
        st.error(f"MLflow inaccessible : {exc}")
        eval_experiments = []

    eval_runs = []
    if eval_experiments:
        exp_ids = [e["experiment_id"] for e in eval_experiments]
        try:
            search_resp = httpx.post(
                f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
                json={
                    "experiment_ids": exp_ids,
                    "filter": "tags.`mlflow.runName` = 'evaluate'",
                    "max_results": 20,
                    "order_by": ["attributes.start_time DESC"],
                },
                timeout=10.0,
            )
            search_resp.raise_for_status()
            eval_runs = search_resp.json().get("runs", [])
        except Exception as exc:
            st.error(f"Erreur lors de la recherche des runs d'evaluation : {exc}")

    if not eval_runs:
        st.info("Aucun run d'evaluation trouve. Lance `make evaluate` pour en generer un.")
    else:
        run_labels = [
            f"{r['info'].get('run_id', '')[:8]} — "
            f"{pd.to_datetime(int(r['info']['start_time']), unit='ms', utc=True)}"
            for r in eval_runs
        ]
        selected_idx = st.selectbox(
            "Run d'evaluation", range(len(eval_runs)), format_func=lambda i: run_labels[i]
        )
        run = eval_runs[selected_idx]
        run_id = run["info"]["run_id"]
        # L'API REST MLflow renvoie les metriques sous forme de liste [{key, value}, ...]
        metrics = {m["key"]: m["value"] for m in run.get("data", {}).get("metrics", [])}

        roc_auc = metrics.get("roc_auc")
        f1 = metrics.get("f1_score")
        accuracy = metrics.get("accuracy_score")
        precision = metrics.get("precision_score")
        recall = metrics.get("recall_score")

        passed_roc = roc_auc is not None and roc_auc >= EVAL_ROC_AUC_MIN
        passed_f1 = f1 is not None and f1 >= EVAL_F1_MIN
        gate_passed = passed_roc and passed_f1

        if gate_passed:
            st.markdown(
                '<div class="result-card result-graduate">'
                "✅ Porte qualite validee — modele pret pour la production"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-card result-dropout">'
                "❌ Porte qualite non validee — modele insuffisant"
                "</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(
            "ROC AUC",
            f"{roc_auc:.4f}" if roc_auc is not None else "—",
            delta="✅ OK" if passed_roc else "❌ sous seuil",
            delta_color="normal" if passed_roc else "inverse",
        )
        k2.metric(
            "F1 Score",
            f"{f1:.4f}" if f1 is not None else "—",
            delta="✅ OK" if passed_f1 else "❌ sous seuil",
            delta_color="normal" if passed_f1 else "inverse",
        )
        k3.metric("Accuracy", f"{accuracy:.4f}" if accuracy is not None else "—")
        k4.metric("Precision", f"{precision:.4f}" if precision is not None else "—")
        k5.metric("Recall", f"{recall:.4f}" if recall is not None else "—")

        st.divider()
        st.markdown("##### Artefacts (matrice de confusion, courbes ROC / precision-rappel)")
        try:
            art_resp = httpx.get(
                f"{MLFLOW_URL}/api/2.0/mlflow/artifacts/list",
                params={"run_id": run_id},
                timeout=5.0,
            )
            art_resp.raise_for_status()
            files = art_resp.json().get("files", [])
        except Exception as exc:
            st.error(f"Impossible de lister les artefacts : {exc}")
            files = []

        image_files = [f for f in files if f.get("path", "").endswith((".png", ".jpg", ".jpeg"))]
        if not image_files:
            st.info("Aucune image d'artefact disponible pour ce run.")
        else:
            img_cols = st.columns(2)
            for i, f in enumerate(image_files):
                try:
                    img_resp = httpx.get(
                        f"{MLFLOW_URL}/get-artifact",
                        params={"path": f["path"], "run_id": run_id},
                        timeout=10.0,
                    )
                    img_resp.raise_for_status()
                    with img_cols[i % 2]:
                        st.image(img_resp.content, caption=f["path"], use_container_width=True)
                except Exception:
                    pass

        st.caption(f"Run ID complet : `{run_id}`")

# ── Dataset ─────────────────────────────────────────────────────────────────────
with dataset_tab:
    st.markdown("#### A propos du dataset")

    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        st.markdown("""
Le dataset **Student Dropout and Academic Success** est issu d'une institution d'enseignement
superieur portugaise. Il regroupe des informations recoltees au moment de l'inscription
et a la fin du premier et second semestre.

**Objectif** : predire si un etudiant va **abandonner** ses etudes ou les terminer
(Graduate / Enrolled), a partir de variables socio-demographiques, academiques et economiques.

**Source** : UCI Machine Learning Repository — Realinho et al., 2022.
        """)

    with col_d2:
        st.markdown("**Repartition de la cible**")
        st.markdown("""
| Classe | Label | Encodage |
|--------|-------|----------|
| Abandon | Dropout | **1** |
| Diplome | Graduate | 0 |
| En cours | Enrolled | 0 |
        """)

    st.divider()
    st.markdown("#### Variables du modele (36 features)")

    groups = {
        "Profil socio-demographique": [
            "Marital Status", "Age at enrollment", "Gender", "Nacionality",
            "International", "Displaced", "Educational special needs",
        ],
        "Parcours academique anterieur": [
            "Previous qualification", "Previous qualification (grade)", "Admission grade",
        ],
        "Inscription": [
            "Application mode", "Application order", "Course", "Daytime/evening attendance",
        ],
        "Situation financiere": [
            "Debtor", "Tuition fees up to date", "Scholarship holder",
        ],
        "Qualifications et professions des parents": [
            "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation",
        ],
        "Resultats 1er semestre": [
            "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
        ],
        "Resultats 2eme semestre": [
            "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
            "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
        ],
        "Contexte macroeconomique": ["Unemployment rate", "Inflation rate", "GDP"],
    }

    for group, features in groups.items():
        with st.expander(f"**{group}** ({len(features)} variables)", expanded=False):
            badges = " ".join(f'<span class="feat-badge">{f}</span>' for f in features)
            st.markdown(badges, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Pipeline de preprocessing")
    st.markdown("""
- **Imputation** : valeurs manquantes remplacees par la **mediane** (SimpleImputer)
- **Normalisation** : mise a l'echelle standard — moyenne 0, ecart-type 1 (StandardScaler)
- **Encodage cible** : `Dropout → 1`, `Graduate / Enrolled → 0`
- **Split** : 80 % entrainement / 20 % test (stratifie, random_state=42)
    """)

# ── Artefacts MLflow ────────────────────────────────────────────────────────────
with artefact_tab:
    st.markdown("#### Metriques des runs MLflow")

    mlflow_url = st.text_input("URL MLflow", value=MLFLOW_URL, key="mlflow_url")
    col_btn, _ = st.columns([1, 5])
    with col_btn:
        load_runs = st.button("🔄 Charger les runs")

    try:
        # Recupere les experiences
        exp_resp = httpx.get(
            f"{mlflow_url}/api/2.0/mlflow/experiments/search",
            params={"max_results": 10},
            timeout=5.0,
        )
        exp_resp.raise_for_status()
        experiments = exp_resp.json().get("experiments", [])
    except Exception as exc:
        st.error(f"MLflow inaccessible : {exc}")
        experiments = []

    if not experiments:
        st.info("Aucune experience trouvee. Verifie que MLflow tourne et que tu as lance un entrainement.")
    else:
        exp_names = {e["experiment_id"]: e["name"] for e in experiments}
        selected_name = st.selectbox("Experience", list(exp_names.values()))
        selected_id = next(k for k, v in exp_names.items() if v == selected_name)

        try:
            runs_resp = httpx.post(
                f"{mlflow_url}/api/2.0/mlflow/runs/search",
                json={"experiment_ids": [selected_id], "max_results": 50},
                timeout=10.0,
            )
            runs_resp.raise_for_status()
            runs = runs_resp.json().get("runs", [])
        except Exception as exc:
            st.error(f"Erreur lors du chargement des runs : {exc}")
            runs = []

        if not runs:
            st.info("Aucun run dans cette experience.")
        else:
            # Les scripts train_models / train_optuna / evaluate ne loggent pas
            # les memes cles de metriques (ex: "f1" vs "test_roc_auc" vs
            # "f1_score") -> on les decouvre dynamiquement plutot que de
            # supposer des noms fixes.
            parsed_runs = []
            metric_keys: set[str] = set()
            for run in runs:
                info = run.get("info", {})
                metrics = {m["key"]: m["value"] for m in run.get("data", {}).get("metrics", [])}
                metric_keys.update(metrics.keys())
                parsed_runs.append({"info": info, "metrics": metrics})

            metric_keys_sorted = sorted(metric_keys)

            rows_mlflow = []
            for pr in parsed_runs:
                info = pr["info"]
                row = {
                    "Run ID": info.get("run_id", "")[:8],
                    "Nom": info.get("run_name", ""),
                    "Statut": info.get("status", ""),
                }
                for key in metric_keys_sorted:
                    row[key] = pr["metrics"].get(key)
                rows_mlflow.append(row)

            df_runs = pd.DataFrame(rows_mlflow)
            for col in metric_keys_sorted:
                df_runs[col] = pd.to_numeric(df_runs[col], errors="coerce")

            # KPI du meilleur run : on prend la 1ere metrique de type ROC AUC presente
            best = None
            best_metric = None
            for candidate in ("roc_auc", "test_roc_auc", "cv_roc_auc"):
                if candidate in metric_keys_sorted and df_runs[candidate].notna().any():
                    best_metric = candidate
                    best = df_runs.loc[df_runs[candidate].idxmax()]
                    break

            if best is not None:
                kpi_keys = metric_keys_sorted[:3]
                st.markdown(f"##### Meilleur run (par {best_metric})")
                kpi_cols = st.columns(1 + len(kpi_keys))
                kpi_cols[0].metric("Nom", str(best.get("Nom", "—")))
                for i, key in enumerate(kpi_keys, start=1):
                    val = best.get(key)
                    kpi_cols[i].metric(key, f"{val:.4f}" if pd.notna(val) else "—")
                st.divider()

            st.markdown(f"##### Tous les runs ({len(df_runs)})")

            def color_status(val: str) -> str:
                if val == "FINISHED":
                    return "color: #2e7d32; font-weight: bold"
                if val == "FAILED":
                    return "color: #c62828; font-weight: bold"
                return ""

            st.dataframe(
                df_runs.style.applymap(color_status, subset=["Statut"]).format(  # type: ignore[arg-type]
                    {c: "{:.4f}" for c in metric_keys_sorted},
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(f"Source : {mlflow_url}")

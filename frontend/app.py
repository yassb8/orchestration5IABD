"""Frontend Streamlit — Prediction de l'abandon scolaire.

Appelle l'API FastAPI (src/api.py) et affiche la prediction.
Lancement : streamlit run frontend/app.py
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Prediction abandon scolaire", layout="wide")
st.title("Prediction de l'abandon scolaire")
st.caption("Modele : Random Forest / XGBoost / LightGBM — Dataset : Student Dropout")

api_url = st.text_input("URL de l'API", value=API_URL)

predict_tab, history_tab = st.tabs(["Prediction", "Historique"])

with predict_tab:
    st.subheader("Informations etudiant")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Profil socio-demographique**")
            marital_status = st.number_input("Marital Status", min_value=1, max_value=6, value=1)
            age = st.number_input("Age at enrollment", min_value=17, max_value=70, value=20)
            gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Homme" if x == 1 else "Femme")
            nacionality = st.number_input("Nacionality", min_value=1, value=1)
            international = st.selectbox("International", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
            displaced = st.selectbox("Displaced", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
            special_needs = st.selectbox("Educational special needs", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")

            st.markdown("**Situation financiere**")
            debtor = st.selectbox("Debtor", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
            tuition = st.selectbox("Tuition fees up to date", [1, 0], format_func=lambda x: "Oui" if x == 1 else "Non")
            scholarship = st.selectbox("Scholarship holder", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")

        with col2:
            st.markdown("**Parcours academique anterieur**")
            prev_qual = st.number_input("Previous qualification", min_value=1, value=1)
            prev_qual_grade = st.number_input("Previous qualification (grade)", min_value=0.0, max_value=200.0, value=122.0)
            admission_grade = st.number_input("Admission grade", min_value=0.0, max_value=200.0, value=127.3)

            st.markdown("**Inscription**")
            app_mode = st.number_input("Application mode", min_value=1, value=1)
            app_order = st.number_input("Application order", min_value=0, max_value=9, value=1)
            course = st.number_input("Course", min_value=1, value=9254)
            attendance = st.selectbox("Daytime/evening attendance", [1, 0], format_func=lambda x: "Jour" if x == 1 else "Soir")

            st.markdown("**Parents**")
            mother_qual = st.number_input("Mother's qualification", min_value=1, value=1)
            father_qual = st.number_input("Father's qualification", min_value=1, value=1)
            mother_occ = st.number_input("Mother's occupation", min_value=0, value=5)
            father_occ = st.number_input("Father's occupation", min_value=0, value=5)

        with col3:
            st.markdown("**Resultats 1er semestre**")
            cu1_credited = st.number_input("1st sem — credited", min_value=0, value=0)
            cu1_enrolled = st.number_input("1st sem — enrolled", min_value=0, value=6)
            cu1_evaluations = st.number_input("1st sem — evaluations", min_value=0, value=6)
            cu1_approved = st.number_input("1st sem — approved", min_value=0, value=6)
            cu1_grade = st.number_input("1st sem — grade", min_value=0.0, max_value=20.0, value=13.5)
            cu1_no_eval = st.number_input("1st sem — without evaluations", min_value=0, value=0)

            st.markdown("**Resultats 2eme semestre**")
            cu2_credited = st.number_input("2nd sem — credited", min_value=0, value=0)
            cu2_enrolled = st.number_input("2nd sem — enrolled", min_value=0, value=6)
            cu2_evaluations = st.number_input("2nd sem — evaluations", min_value=0, value=6)
            cu2_approved = st.number_input("2nd sem — approved", min_value=0, value=6)
            cu2_grade = st.number_input("2nd sem — grade", min_value=0.0, max_value=20.0, value=13.5)
            cu2_no_eval = st.number_input("2nd sem — without evaluations", min_value=0, value=0)

            st.markdown("**Contexte macroeconomique**")
            unemployment = st.number_input("Unemployment rate", value=10.8)
            inflation = st.number_input("Inflation rate", value=1.4)
            gdp = st.number_input("GDP", value=1.74)

        submitted = st.form_submit_button("Predire", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "Marital Status": marital_status,
            "Age at enrollment": age,
            "Gender": gender,
            "Nacionality": nacionality,
            "International": international,
            "Displaced": displaced,
            "Educational special needs": special_needs,
            "Previous qualification": prev_qual,
            "Previous qualification (grade)": prev_qual_grade,
            "Admission grade": admission_grade,
            "Application mode": app_mode,
            "Application order": app_order,
            "Course": course,
            "Daytime/evening attendance": attendance,
            "Debtor": debtor,
            "Tuition fees up to date": tuition,
            "Scholarship holder": scholarship,
            "Mother's qualification": mother_qual,
            "Father's qualification": father_qual,
            "Mother's occupation": mother_occ,
            "Father's occupation": father_occ,
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
            "Unemployment rate": unemployment,
            "Inflation rate": inflation,
            "GDP": gdp,
        }
        try:
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Appel a l'API impossible : {exc}")
        else:
            prediction = result["prediction"]
            probability = result["probability"]

            st.divider()
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                if prediction == 1:
                    st.error("Risque d'abandon (Dropout)")
                else:
                    st.success("Pas de risque d'abandon (Graduate / Enrolled)")
            with col_res2:
                st.metric("Probabilite d'abandon", f"{probability:.1%}")
                st.progress(probability)

with history_tab:
    st.subheader("Historique des previsions")
    st.info("Aucun journal de previsions : ajoutez un endpoint /predictions a l'API (bonus).")

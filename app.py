"""
PredictWell - AI-Powered Multi-Disease Prediction System
------------------------------------------------------------
Streamlit web interface for real-time chronic disease risk prediction
(diabetes, heart disease) using pre-trained scikit-learn models.
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="PredictWell | AI Disease Risk Prediction",
    page_icon="🩺",
    layout="wide",
)

MODELS_DIR = Path("models")


@st.cache_resource
def load_artifacts(prefix):
    model = joblib.load(MODELS_DIR / f"{prefix}_model.joblib")
    scaler = joblib.load(MODELS_DIR / f"{prefix}_scaler.joblib")
    with open(MODELS_DIR / f"{prefix}_metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata


def risk_badge(prob):
    if prob < 0.33:
        st.success(f"🟢 Low Risk — {prob:.1%} predicted probability")
    elif prob < 0.66:
        st.warning(f"🟡 Moderate Risk — {prob:.1%} predicted probability")
    else:
        st.error(f"🔴 High Risk — {prob:.1%} predicted probability")


# ---------------------------------------------------------------- Header
st.title("🩺 PredictWell")
st.caption("AI-Powered Multi-Disease Prediction System — Python · Scikit-learn · PySpark")

tab_diabetes, tab_heart, tab_about = st.tabs(
    ["🍬 Diabetes Risk", "❤️ Heart Disease Risk", "ℹ️ About this project"]
)

# ---------------------------------------------------------------- Diabetes tab
with tab_diabetes:
    model, scaler, meta = load_artifacts("diabetes")
    st.subheader("Diabetes Risk Assessment")
    st.caption(f"Model: {meta['best_model'].replace('_',' ').title()} "
               f"· Test accuracy: {meta['best_accuracy']:.1%}")

    c1, c2, c3 = st.columns(3)
    with c1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with c2:
        insulin = st.number_input("Insulin (mu U/mL)", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
        age = st.number_input("Age", 1, 120, 30, key="d_age")

    with c3:
        st.markdown("**Derived features (auto-calculated)**")
        bmi_category = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3
        age_bucket = 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3
        glucose_insulin_ratio = round(glucose / (insulin + 1), 4)
        st.write(f"BMI category: `{bmi_category}`")
        st.write(f"Age bucket: `{age_bucket}`")
        st.write(f"Glucose/Insulin ratio: `{glucose_insulin_ratio}`")

    if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "pregnancies": pregnancies, "glucose": glucose,
            "blood_pressure": blood_pressure, "skin_thickness": skin_thickness,
            "insulin": insulin, "bmi": bmi, "diabetes_pedigree": diabetes_pedigree,
            "age": age, "bmi_category": bmi_category, "age_bucket": age_bucket,
            "glucose_insulin_ratio": glucose_insulin_ratio,
        }])[meta["feature_cols"]]

        scaled = scaler.transform(row)
        prob = model.predict_proba(scaled)[0][1]
        risk_badge(prob)
        st.caption("⚠️ Educational demo only — not a substitute for professional medical advice.")

# ---------------------------------------------------------------- Heart tab
with tab_heart:
    model_h, scaler_h, meta_h = load_artifacts("heart")
    st.subheader("Heart Disease Risk Assessment")
    st.caption(f"Model: {meta_h['best_model'].replace('_',' ').title()} "
               f"· Test accuracy: {meta_h['best_accuracy']:.1%}")

    c1, c2, c3 = st.columns(3)
    with c1:
        age_h = st.number_input("Age", 1, 120, 50, key="h_age")
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                           format_func=lambda x: ["Typical Angina", "Atypical Angina",
                                                   "Non-anginal Pain", "Asymptomatic"][x])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 0, 250, 130)
        chol = st.number_input("Cholesterol (mg/dL)", 0, 600, 220)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    with c2:
        restecg = st.selectbox("Resting ECG", [0, 1, 2],
                                format_func=lambda x: ["Normal", "ST-T Abnormality",
                                                        "LV Hypertrophy"][x])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
        exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2],
                              format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    with c3:
        ca = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                             format_func=lambda x: ["Unknown", "Normal", "Fixed Defect",
                                                     "Reversible Defect"][x])
        st.markdown("**Derived features (auto-calculated)**")
        age_bucket_h = 0 if age_h < 40 else 1 if age_h < 55 else 2 if age_h < 65 else 3
        chol_risk = 0 if chol < 200 else 1 if chol < 240 else 2
        max_hr_reserve = round(220 - age_h - thalach, 2)
        st.write(f"Age bucket: `{age_bucket_h}`")
        st.write(f"Cholesterol risk: `{chol_risk}`")
        st.write(f"Max HR reserve: `{max_hr_reserve}`")

    if st.button("Predict Heart Disease Risk", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "age": age_h, "sex": 1 if sex == "Male" else 0, "cp": cp,
            "trestbps": trestbps, "chol": chol, "fbs": 1 if fbs == "Yes" else 0,
            "restecg": restecg, "thalach": thalach, "exang": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
            "age_bucket": age_bucket_h, "chol_risk": chol_risk,
            "max_hr_reserve": max_hr_reserve,
        }])[meta_h["feature_cols"]]

        scaled = scaler_h.transform(row)
        prob = model_h.predict_proba(scaled)[0][1]
        risk_badge(prob)
        st.caption("⚠️ Educational demo only — not a substitute for professional medical advice.")

# ---------------------------------------------------------------- About tab
with tab_about:
    st.subheader("About PredictWell")
    st.markdown("""
**PredictWell** is an end-to-end machine learning system for predicting chronic
disease risk (diabetes, heart disease) from patient health records.

**Pipeline:**
1. **Apache PySpark** performs distributed data cleaning, missing-value imputation,
   and feature engineering (BMI/cholesterol risk buckets, age buckets, engineered ratios)
   on the raw datasets.
2. **Scikit-learn** trains and compares Logistic Regression and Random Forest
   classifiers per disease; the best performer on held-out test data is deployed.
3. **Streamlit** serves the trained models as an interactive web app for
   real-time risk prediction.

**Datasets:** Pima Indians Diabetes Dataset · UCI Cleveland Heart Disease Dataset
    """)
    d_meta = load_artifacts("diabetes")[2]
    h_meta = load_artifacts("heart")[2]
    st.markdown("**Model performance (held-out test set):**")
    perf = pd.DataFrame({
        "Diabetes": d_meta["all_results"]["random_forest"],
        "Heart Disease": h_meta["all_results"]["random_forest"],
    }).T
    st.dataframe(perf, use_container_width=True)

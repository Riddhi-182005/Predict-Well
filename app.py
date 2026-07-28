"""
PredictWell - AI-Powered Multi-Disease Prediction System
------------------------------------------------------------
Streamlit web interface for real-time chronic disease risk prediction
(diabetes, heart disease) using pre-trained scikit-learn models on
Spark-preprocessed clinical data.
"""

import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="PredictWell | AI Disease Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODELS_DIR = Path("models")

# ============================================================== DESIGN SYSTEM
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root{
  --pw-bg:#F6F8F9;
  --pw-surface:#FFFFFF;
  --pw-ink:#0F2A3D;
  --pw-muted:#5B6B7C;
  --pw-border:#E3E8EB;
  --pw-primary:#0E7C7B;
  --pw-primary-dark:#0B5D5C;
  --pw-primary-tint:#E6F3F2;
  --pw-risk-low:#1E8E5A;
  --pw-risk-low-bg:#EAF7EF;
  --pw-risk-mid:#B5741B;
  --pw-risk-mid-bg:#FBF1E1;
  --pw-risk-high:#C0392B;
  --pw-risk-high-bg:#FBEAE8;
}

html, body, [class*="css"]  { font-family:'Inter', sans-serif; }
h1, h2, h3, .pw-heading { font-family:'Sora', sans-serif !important; letter-spacing:-0.01em; }

.block-container{ padding-top:2rem; max-width:1100px; }

/* ---------- Hero ---------- */
.pw-hero{
  display:flex; align-items:center; justify-content:space-between;
  gap:24px; padding-bottom:6px; margin-bottom:4px;
  border-bottom:1px solid var(--pw-border);
}
.pw-hero-mark{
  width:52px; height:52px; border-radius:14px;
  background:linear-gradient(145deg, var(--pw-primary), var(--pw-primary-dark));
  display:flex; align-items:center; justify-content:center;
  font-size:26px; flex-shrink:0;
  box-shadow:0 6px 16px rgba(14,124,123,0.25);
}
.pw-hero-title{ font-family:'Sora',sans-serif; font-weight:700; font-size:1.9rem; color:var(--pw-ink); margin:0; line-height:1.1;}
.pw-hero-sub{ color:var(--pw-muted); font-size:0.95rem; margin-top:2px;}

.pw-badge-row{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
.pw-badge{
  font-size:0.76rem; font-weight:600; padding:5px 12px; border-radius:999px;
  background:var(--pw-primary-tint); color:var(--pw-primary-dark);
  border:1px solid rgba(14,124,123,0.18);
}

/* ---------- Pipeline strip ---------- */
.pw-pipeline{ display:flex; gap:14px; margin:26px 0 8px 0; }
.pw-stage{
  flex:1; background:var(--pw-surface); border:1px solid var(--pw-border);
  border-radius:14px; padding:16px 18px; position:relative;
}
.pw-stage-num{
  font-family:'Sora',sans-serif; font-weight:800; font-size:0.75rem;
  color:var(--pw-primary); letter-spacing:0.06em;
}
.pw-stage-title{ font-family:'Sora',sans-serif; font-weight:600; font-size:0.98rem; color:var(--pw-ink); margin:4px 0 4px 0;}
.pw-stage-desc{ font-size:0.82rem; color:var(--pw-muted); line-height:1.4; }

/* ---------- Section labels ---------- */
.pw-eyebrow{
  font-size:0.72rem; font-weight:700; letter-spacing:0.09em; text-transform:uppercase;
  color:var(--pw-primary); margin-bottom:2px;
}
.pw-section-title{ font-family:'Sora',sans-serif; font-weight:700; font-size:1.3rem; color:var(--pw-ink); margin:0 0 2px 0;}
.pw-section-desc{ color:var(--pw-muted); font-size:0.88rem; margin-bottom:18px; }

/* ---------- Risk card ---------- */
.pw-risk-card{
  border-radius:14px; padding:20px 22px; margin-top:10px; border:1px solid;
  display:flex; align-items:center; gap:16px;
}
.pw-risk-icon{ font-size:2rem; line-height:1; }
.pw-risk-label{ font-family:'Sora',sans-serif; font-weight:700; font-size:1.1rem; }
.pw-risk-prob{ font-size:0.85rem; opacity:0.85; margin-top:2px;}
.risk-low{ background:var(--pw-risk-low-bg); border-color:rgba(30,142,90,0.3); color:var(--pw-risk-low); }
.risk-mid{ background:var(--pw-risk-mid-bg); border-color:rgba(181,116,27,0.3); color:var(--pw-risk-mid); }
.risk-high{ background:var(--pw-risk-high-bg); border-color:rgba(192,57,43,0.3); color:var(--pw-risk-high); }

/* ---------- Metric cards ---------- */
.pw-metric-card{
  background:var(--pw-surface); border:1px solid var(--pw-border); border-radius:12px;
  padding:14px 16px; text-align:center;
}
.pw-metric-val{ font-family:'Sora',sans-serif; font-weight:700; font-size:1.35rem; color:var(--pw-ink); }
.pw-metric-label{ font-size:0.72rem; color:var(--pw-muted); text-transform:uppercase; letter-spacing:0.06em; margin-top:2px;}

/* ---------- Skill chips (About tab) ---------- */
.pw-chip-row{ display:flex; flex-wrap:wrap; gap:8px; margin:10px 0 22px 0; }
.pw-chip{
  font-size:0.8rem; font-weight:600; padding:6px 13px; border-radius:8px;
  background:var(--pw-surface); border:1px solid var(--pw-border); color:var(--pw-ink);
}

/* ---------- Footer ---------- */
.pw-footer{
  margin-top:38px; padding-top:16px; border-top:1px solid var(--pw-border);
  color:var(--pw-muted); font-size:0.8rem; display:flex; justify-content:space-between; flex-wrap:wrap; gap:6px;
}

/* Tab labels */
.stTabs [data-baseweb="tab"] { font-family:'Sora',sans-serif; font-weight:600; font-size:0.92rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts(prefix):
    model = joblib.load(MODELS_DIR / f"{prefix}_model.joblib")
    scaler = joblib.load(MODELS_DIR / f"{prefix}_scaler.joblib")
    with open(MODELS_DIR / f"{prefix}_metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata


def render_risk_card(prob):
    if prob < 0.33:
        cls, icon, label = "risk-low", "🟢", "Low Risk"
    elif prob < 0.66:
        cls, icon, label = "risk-mid", "🟡", "Moderate Risk"
    else:
        cls, icon, label = "risk-high", "🔴", "High Risk"
    st.markdown(f"""
    <div class="pw-risk-card {cls}">
        <div class="pw-risk-icon">{icon}</div>
        <div>
            <div class="pw-risk-label">{label}</div>
            <div class="pw-risk-prob">Predicted risk probability: {prob:.1%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("This is an educational demo, not a medical diagnosis. Consult a licensed clinician for real health decisions.")


def render_metric_cards(results):
    cols = st.columns(4)
    labels = [("accuracy", "Accuracy"), ("precision", "Precision"),
              ("recall", "Recall"), ("f1", "F1 Score")]
    for col, (key, label) in zip(cols, labels):
        with col:
            st.markdown(f"""
            <div class="pw-metric-card">
                <div class="pw-metric-val">{results[key]:.1%}</div>
                <div class="pw-metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================== HERO
st.markdown("""
<div class="pw-hero">
  <div style="display:flex; align-items:center; gap:16px;">
    <div class="pw-hero-mark">🩺</div>
    <div>
      <p class="pw-hero-title">PredictWell</p>
      <p class="pw-hero-sub">AI-powered chronic disease risk prediction, from raw clinical data to a live prediction service.</p>
    </div>
  </div>
</div>
<div class="pw-badge-row">
  <span class="pw-badge">Python</span>
  <span class="pw-badge">Apache PySpark</span>
  <span class="pw-badge">Scikit-learn</span>
  <span class="pw-badge">Pandas</span>
  <span class="pw-badge">Streamlit</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pw-pipeline">
  <div class="pw-stage">
    <div class="pw-stage-num">STAGE 01</div>
    <div class="pw-stage-title">Distributed Preprocessing</div>
    <div class="pw-stage-desc">Apache PySpark cleans, imputes, and engineers features across patient records in parallel.</div>
  </div>
  <div class="pw-stage">
    <div class="pw-stage-num">STAGE 02</div>
    <div class="pw-stage-title">Model Training & Selection</div>
    <div class="pw-stage-desc">Logistic Regression and Random Forest classifiers are trained and benchmarked per disease.</div>
  </div>
  <div class="pw-stage">
    <div class="pw-stage-num">STAGE 03</div>
    <div class="pw-stage-title">Real-Time Prediction</div>
    <div class="pw-stage-desc">The best-performing model is served through this interactive Streamlit interface.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")
tab_diabetes, tab_heart, tab_about = st.tabs(
    ["Diabetes Risk", "Heart Disease Risk", "About This Project"]
)

# ============================================================== DIABETES TAB
with tab_diabetes:
    model, scaler, meta = load_artifacts("diabetes")

    st.markdown('<div class="pw-eyebrow">Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="pw-section-title">Diabetes Risk</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pw-section-desc">Powered by a {meta["best_model"].replace("_"," ").title()} '
                f'classifier evaluated on held-out patient records.</div>', unsafe_allow_html=True)

    render_metric_cards(meta["all_results"][meta["best_model"]])
    st.write("")

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
        st.markdown('<div class="pw-eyebrow">Engineered Features</div>', unsafe_allow_html=True)
        bmi_category = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3
        age_bucket = 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3
        glucose_insulin_ratio = round(glucose / (insulin + 1), 4)
        st.write(f"BMI category `{bmi_category}` · Age bucket `{age_bucket}`")
        st.write(f"Glucose / insulin ratio: `{glucose_insulin_ratio}`")
        st.caption("Computed automatically by the same feature-engineering logic used in the PySpark training pipeline.")

    if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "pregnancies": pregnancies, "glucose": glucose,
            "blood_pressure": blood_pressure, "skin_thickness": skin_thickness,
            "insulin": insulin, "bmi": bmi, "diabetes_pedigree": diabetes_pedigree,
            "age": age, "bmi_category": bmi_category, "age_bucket": age_bucket,
            "glucose_insulin_ratio": glucose_insulin_ratio,
        }])[meta["feature_cols"]]
        prob = model.predict_proba(scaler.transform(row))[0][1]
        render_risk_card(prob)

# ============================================================== HEART TAB
with tab_heart:
    model_h, scaler_h, meta_h = load_artifacts("heart")

    st.markdown('<div class="pw-eyebrow">Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="pw-section-title">Heart Disease Risk</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="pw-section-desc">Powered by a {meta_h["best_model"].replace("_"," ").title()} '
                f'classifier evaluated on held-out patient records.</div>', unsafe_allow_html=True)

    render_metric_cards(meta_h["all_results"][meta_h["best_model"]])
    st.write("")

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
                                format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
        exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2],
                              format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    with c3:
        ca = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                             format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"][x])
        st.markdown('<div class="pw-eyebrow">Engineered Features</div>', unsafe_allow_html=True)
        age_bucket_h = 0 if age_h < 40 else 1 if age_h < 55 else 2 if age_h < 65 else 3
        chol_risk = 0 if chol < 200 else 1 if chol < 240 else 2
        max_hr_reserve = round(220 - age_h - thalach, 2)
        st.write(f"Age bucket `{age_bucket_h}` · Cholesterol risk `{chol_risk}`")
        st.write(f"Max heart-rate reserve: `{max_hr_reserve}`")
        st.caption("Computed automatically by the same feature-engineering logic used in the PySpark training pipeline.")

    if st.button("Predict Heart Disease Risk", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "age": age_h, "sex": 1 if sex == "Male" else 0, "cp": cp,
            "trestbps": trestbps, "chol": chol, "fbs": 1 if fbs == "Yes" else 0,
            "restecg": restecg, "thalach": thalach, "exang": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
            "age_bucket": age_bucket_h, "chol_risk": chol_risk,
            "max_hr_reserve": max_hr_reserve,
        }])[meta_h["feature_cols"]]
        prob = model_h.predict_proba(scaler_h.transform(row))[0][1]
        render_risk_card(prob)

# ============================================================== ABOUT TAB
with tab_about:
    d_meta = load_artifacts("diabetes")[2]
    h_meta = load_artifacts("heart")[2]

    st.markdown('<div class="pw-eyebrow">Project Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="pw-section-title">PredictWell — AI-Powered Multi-Disease Prediction System</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="pw-section-desc" style="max-width:760px;">
PredictWell is an end-to-end machine learning system that estimates chronic disease risk —
diabetes and heart disease — from patient health records. It was built to demonstrate the
full lifecycle of an applied ML product: distributed data engineering, model development
and evaluation, and deployment behind a real, usable interface.
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="pw-eyebrow" style="margin-top:8px;">Tech Stack</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="pw-chip-row">
  <span class="pw-chip">Python</span>
  <span class="pw-chip">Apache PySpark</span>
  <span class="pw-chip">Scikit-learn</span>
  <span class="pw-chip">Pandas</span>
  <span class="pw-chip">NumPy</span>
  <span class="pw-chip">Streamlit</span>
  <span class="pw-chip">Joblib</span>
  <span class="pw-chip">Git / GitHub</span>
</div>
""", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="pw-eyebrow">Data Engineering</div>', unsafe_allow_html=True)
        st.markdown("""
- Distributed cleaning and median imputation of missing clinical values using **Apache PySpark**, run in parallel across partitions rather than a single-threaded Pandas pass
- Engineered features per disease (BMI category, age bucket, glucose–insulin ratio for diabetes; cholesterol risk band and max heart-rate reserve for heart disease) computed at the Spark layer
- Output written as clean, model-ready datasets consumed downstream by the training pipeline
        """)
    with right:
        st.markdown('<div class="pw-eyebrow">Modeling & Evaluation</div>', unsafe_allow_html=True)
        st.markdown("""
- Trained and benchmarked **Logistic Regression** and **Random Forest** classifiers per disease using **scikit-learn**
- Selected the best model per disease on a held-out test split using accuracy, precision, recall, and F1
- Persisted the winning model, feature scaler, and metadata with **Joblib** for reproducible serving
        """)

    st.markdown('<div class="pw-eyebrow" style="margin-top:18px;">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="pw-section-desc">Measured on a held-out 20% test split, not training data.</div>', unsafe_allow_html=True)
    perf = pd.DataFrame({
        "Diabetes": d_meta["all_results"][d_meta["best_model"]],
        "Heart Disease": h_meta["all_results"][h_meta["best_model"]],
    }).T
    perf = (perf * 100).round(1).astype(str) + "%"
    st.dataframe(perf, use_container_width=True)
    st.caption(
        "Datasets: Pima Indians Diabetes Dataset (768 records) and UCI Cleveland Heart Disease "
        "Dataset (303 records). Accuracy in this range is consistent with published benchmarks "
        "for these datasets at this sample size."
    )

    st.markdown('<div class="pw-eyebrow" style="margin-top:18px;">Skills Demonstrated</div>', unsafe_allow_html=True)
    st.markdown("""
- Building a distributed data-processing pipeline with Apache PySpark for cleaning and feature engineering
- Training, comparing, and selecting classical ML models (Logistic Regression, Random Forest) with scikit-learn
- Translating a trained model into a live, interactive product using Streamlit
- End-to-end ML deployment: version control, dependency management, and continuous deployment to a public URL
    """)

    st.markdown("""
<div class="pw-footer">
  <span>PredictWell — educational project. Not a substitute for professional medical advice.</span>
  <span>Built with Python · PySpark · Scikit-learn · Streamlit</span>
</div>
""", unsafe_allow_html=True)

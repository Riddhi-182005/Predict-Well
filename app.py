"""
PredictWell - AI-Powered Multi-Disease Prediction System
Consumer-facing Streamlit app: enter health details, get an instant risk result.
"""

import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="PredictWell",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODELS_DIR = Path("models")

# ============================================================== STYLE
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root{
  --bg:#FAFAFA; --ink:#1A1A1A; --muted:#6B7280; --border:#E5E7EB;
  --primary:#0E7C7B;
  --low:#1E8E5A; --low-bg:#F0FAF4;
  --mid:#B5741B; --mid-bg:#FDF6EC;
  --high:#C0392B; --high-bg:#FDF1F0;
}

html, body, [class*="css"] { font-family:'Inter', sans-serif !important; }
.stApp, body, html { background-color: var(--bg) !important; }
.block-container{ max-width:640px; padding-top:3rem; padding-bottom:3rem; }

.pw-title{ font-size:1.5rem !important; font-weight:700 !important; color:var(--ink) !important; margin:0 !important; }
.pw-sub{ color:var(--muted) !important; font-size:0.92rem !important; margin-top:6px !important; margin-bottom:0 !important; }

/* Result card */
.pw-result{
  border-radius:12px; padding:22px 24px; margin-top:8px; border:1px solid;
  display:flex; align-items:center; gap:18px;
}
.low{ background:var(--low-bg); border-color:#D6EFDF; }
.mid{ background:var(--mid-bg); border-color:#F3E1C4; }
.high{ background:var(--high-bg); border-color:#F3D4D1; }

.pw-gauge{ width:76px; height:76px; border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; }
.pw-gauge-inner{ width:58px; height:58px; background:#fff; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:1rem; }

.pw-result-label{ font-weight:600; font-size:1.05rem; }
.pw-result-desc{ font-size:0.86rem; color:var(--muted); margin-top:3px; line-height:1.4;}
.low .pw-result-label{ color:var(--low); } .low .pw-gauge-inner{ color:var(--low); }
.mid .pw-result-label{ color:var(--mid); } .mid .pw-gauge-inner{ color:var(--mid); }
.high .pw-result-label{ color:var(--high);} .high .pw-gauge-inner{ color:var(--high);}

/* Form */
div[data-testid="stForm"] {
  border:1px solid var(--border); border-radius:12px;
  padding:24px 24px 10px 24px; background:#fff;
}
label, [data-testid="stWidgetLabel"] p { color:var(--ink) !important; font-size:0.88rem !important; font-weight:500 !important; }

.stNumberInput input, .stTextInput input {
  background:#fff !important; color:var(--ink) !important;
  border:1px solid var(--border) !important; border-radius:8px !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
  border-color:var(--primary) !important; box-shadow:none !important; outline:none !important;
}
.stSelectbox div[data-baseweb="select"] > div {
  background:#fff !important; color:var(--ink) !important;
  border:1px solid var(--border) !important; border-radius:8px !important;
}
.stSelectbox div[data-baseweb="select"] * { color:var(--ink) !important; }
[data-baseweb="popover"] li { color:var(--ink) !important; background:#fff !important; }
[data-baseweb="popover"] li:hover { background:#F3F4F6 !important; }

.stFormSubmitButton>button{
  background:var(--primary) !important; border-color:var(--primary) !important;
  font-weight:600 !important; border-radius:8px !important; box-shadow:none !important;
}
.stFormSubmitButton>button:hover{ background:#0B5D5C !important; border-color:#0B5D5C !important; }

.stTabs [data-baseweb="tab"] { font-weight:500 !important; font-size:0.92rem !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts(prefix):
    model = joblib.load(MODELS_DIR / f"{prefix}_model.joblib")
    scaler = joblib.load(MODELS_DIR / f"{prefix}_scaler.joblib")
    with open(MODELS_DIR / f"{prefix}_metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata


def render_result(prob):
    pct = round(prob * 100)
    if prob < 0.33:
        band, label, msg, color = "low", "Low Risk", \
            "Your inputs don't show strong risk indicators. Keep up regular checkups.", "#1E8E5A"
    elif prob < 0.66:
        band, label, msg, color = "mid", "Moderate Risk", \
            "Some risk indicators are present. Consider discussing these results with a doctor.", "#B5741B"
    else:
        band, label, msg, color = "high", "High Risk", \
            "Several risk indicators are present. We recommend consulting a healthcare provider soon.", "#C0392B"

    st.markdown(f"""
    <div class="pw-result {band}">
        <div class="pw-gauge" style="background:conic-gradient({color} {pct}%, #E5E7EB {pct}% 100%);">
            <div class="pw-gauge-inner">{pct}%</div>
        </div>
        <div>
            <div class="pw-result-label">{label}</div>
            <div class="pw-result-desc">{msg}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("This tool provides an estimate for informational purposes only and is not a medical diagnosis.")


# ============================================================== HEADER
st.markdown("""
<p class="pw-title">PredictWell</p>
<p class="pw-sub">Enter your health details to get an instant risk estimate.</p>
""", unsafe_allow_html=True)
st.write("")

tab_diabetes, tab_heart = st.tabs(["Diabetes", "Heart Disease"])

# ============================================================== DIABETES
with tab_diabetes:
    model, scaler, meta = load_artifacts("diabetes")

    with st.form("diabetes_form"):
        c1, c2 = st.columns(2)
        with c1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            glucose = st.number_input("Glucose (mg/dL)", 0, 300, 120)
            blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)
            skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        with c2:
            insulin = st.number_input("Insulin (mu U/mL)", 0, 900, 80)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1)
            diabetes_pedigree = st.number_input("Family History Score", 0.0, 3.0, 0.5, step=0.01,
                                                 help="Diabetes Pedigree Function — reflects family history of diabetes")
            age = st.number_input("Age", 1, 120, 30, key="d_age")

        submitted = st.form_submit_button("Check My Risk", type="primary", use_container_width=True)

    if submitted:
        bmi_category = 0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3
        age_bucket = 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3
        glucose_insulin_ratio = round(glucose / (insulin + 1), 4)

        row = pd.DataFrame([{
            "pregnancies": pregnancies, "glucose": glucose,
            "blood_pressure": blood_pressure, "skin_thickness": skin_thickness,
            "insulin": insulin, "bmi": bmi, "diabetes_pedigree": diabetes_pedigree,
            "age": age, "bmi_category": bmi_category, "age_bucket": age_bucket,
            "glucose_insulin_ratio": glucose_insulin_ratio,
        }])[meta["feature_cols"]]
        prob = model.predict_proba(scaler.transform(row))[0][1]
        render_result(prob)

# ============================================================== HEART
with tab_heart:
    model_h, scaler_h, meta_h = load_artifacts("heart")

    with st.form("heart_form"):
        c1, c2 = st.columns(2)
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
            ca = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", [0, 1, 2, 3],
                                 format_func=lambda x: ["Unknown", "Normal", "Fixed Defect", "Reversible Defect"][x])

        submitted_h = st.form_submit_button("Check My Risk", type="primary", use_container_width=True)

    if submitted_h:
        age_bucket_h = 0 if age_h < 40 else 1 if age_h < 55 else 2 if age_h < 65 else 3
        chol_risk = 0 if chol < 200 else 1 if chol < 240 else 2
        max_hr_reserve = round(220 - age_h - thalach, 2)

        row = pd.DataFrame([{
            "age": age_h, "sex": 1 if sex == "Male" else 0, "cp": cp,
            "trestbps": trestbps, "chol": chol, "fbs": 1 if fbs == "Yes" else 0,
            "restecg": restecg, "thalach": thalach, "exang": 1 if exang == "Yes" else 0,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
            "age_bucket": age_bucket_h, "chol_risk": chol_risk,
            "max_hr_reserve": max_hr_reserve,
        }])[meta_h["feature_cols"]]
        prob = model_h.predict_proba(scaler_h.transform(row))[0][1]
        render_result(prob)

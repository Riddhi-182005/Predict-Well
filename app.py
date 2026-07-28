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
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root{
  --bg:#F6F8F9; --ink:#0F2A3D; --muted:#5B6B7C; --border:#E3E8EB;
  --primary:#0E7C7B; --primary-dark:#0B5D5C; --primary-tint:#E6F3F2;
  --low:#1E8E5A; --low-bg:#EAF7EF;
  --mid:#B5741B; --mid-bg:#FBF1E1;
  --high:#C0392B; --high-bg:#FBEAE8;
}
html, body, [class*="css"] { font-family:'Inter', sans-serif; }
h1,h2,h3 { font-family:'Sora', sans-serif !important; }
.stApp, body, html { background-color: var(--bg) !important; }
.block-container{ max-width:700px; padding-top:2.2rem; padding-bottom:3rem; }

.pw-title{ font-family:'Sora',sans-serif !important; font-weight:800 !important; font-size:1.9rem !important; color:var(--ink) !important; margin:0 !important; line-height:1.25 !important; }
.pw-sub{ color:var(--muted) !important; font-size:0.95rem !important; margin-top:4px !important; margin-bottom:0.4rem !important; }

.pw-result{
  border-radius:18px; padding:26px 26px; margin-top:6px; border:1px solid;
  display:flex; align-items:center; gap:22px;
}
.low{ background:var(--low-bg); border-color:rgba(30,142,90,0.25); }
.mid{ background:var(--mid-bg); border-color:rgba(181,116,27,0.25); }
.high{ background:var(--high-bg); border-color:rgba(192,57,43,0.25); }

.pw-gauge{ width:104px; height:104px; border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; }
.pw-gauge-inner{ width:80px; height:80px; background:#fff; border-radius:50%; display:flex; align-items:center; justify-content:center; font-family:'Sora',sans-serif; font-weight:700; font-size:1.15rem; }

.pw-result-label{ font-family:'Sora',sans-serif; font-weight:700; font-size:1.25rem; }
.pw-result-desc{ font-size:0.88rem; color:var(--muted); margin-top:4px; max-width:380px; line-height:1.4;}
.low .pw-result-label{ color:var(--low); } .low .pw-gauge-inner{ color:var(--low); }
.mid .pw-result-label{ color:var(--mid); } .mid .pw-gauge-inner{ color:var(--mid); }
.high .pw-result-label{ color:var(--high);} .high .pw-gauge-inner{ color:var(--high);}

.stTabs [data-baseweb="tab"] { font-family:'Sora',sans-serif; font-weight:600; }
div[data-testid="stForm"] { border:1px solid var(--border); border-radius:16px; padding:22px 22px 8px 22px; background:#fff; }
.stButton>button[kind="primary"], .stFormSubmitButton>button{
  background:var(--primary); border-color:var(--primary); font-weight:600; border-radius:10px;
}
.stButton>button[kind="primary"]:hover, .stFormSubmitButton>button:hover{
  background:var(--primary-dark); border-color:var(--primary-dark);
}

label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
  color: var(--ink) !important;
}
.stNumberInput input, .stSelectbox div[data-baseweb="select"] * {
  color: var(--ink) !important;
}
[data-testid="stForm"] { color: var(--ink); }

/* ---- Inputs: force light background + dark text (fixes invisible values) ---- */
.stNumberInput input, .stTextInput input {
  background:#ffffff !important; color:var(--ink) !important;
  border:1.5px solid var(--border) !important; border-radius:9px !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.stNumberInput input:focus, .stTextInput input:focus {
  border-color:var(--primary) !important; box-shadow:0 0 0 3px var(--primary-tint) !important; outline:none !important;
}
.stSelectbox div[data-baseweb="select"] > div {
  background:#ffffff !important; color:var(--ink) !important;
  border:1.5px solid var(--border) !important; border-radius:9px !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.stSelectbox div[data-baseweb="select"] * { color:var(--ink) !important; }
.stSelectbox div[data-baseweb="select"]:focus-within > div {
  border-color:var(--primary) !important; box-shadow:0 0 0 3px var(--primary-tint) !important;
}
[data-baseweb="popover"] li { color:var(--ink) !important; background:#fff !important; }
[data-baseweb="popover"] li:hover { background:var(--primary-tint) !important; }
.stNumberInput button { background:#fff !important; }

/* ---- Micro-interactions ---- */
.stFormSubmitButton>button{
  transition: transform 0.12s ease, background 0.15s ease;
}
.stFormSubmitButton>button:hover{ transform: translateY(-1px); }
.stFormSubmitButton>button:active{ transform: translateY(0); }

@keyframes pwFadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
.pw-result { animation: pwFadeUp 0.35s ease; }

.stTabs [data-baseweb="tab-list"] { gap:4px; }
.stTabs [data-baseweb="tab"] { transition: color 0.15s ease; }
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
            "Your inputs don't show strong risk indicators. Keep up regular checkups.", "var(--low)"
    elif prob < 0.66:
        band, label, msg, color = "mid", "Moderate Risk", \
            "Some risk indicators are present. Consider discussing these results with a doctor.", "var(--mid)"
    else:
        band, label, msg, color = "high", "High Risk", \
            "Several risk indicators are present. We recommend consulting a healthcare provider soon.", "var(--high)"

    st.markdown(f"""
    <div class="pw-result {band}">
        <div class="pw-gauge" style="background:conic-gradient({color} {pct}%, #E7ECEE {pct}% 100%);">
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
<div style="display:flex; align-items:center; gap:12px; margin-bottom:2px;">
  <div style="width:42px; height:42px; border-radius:11px; flex-shrink:0;
              background:linear-gradient(145deg, var(--primary), var(--primary-dark));
              display:flex; align-items:center; justify-content:center;
              font-family:'Sora',sans-serif; font-weight:800; color:#fff; font-size:1.1rem;">P</div>
  <p class="pw-title">PredictWell</p>
</div>
<p class="pw-sub">Enter your health details to get an instant risk estimate — takes less than a minute.</p>
""", unsafe_allow_html=True)

tab_diabetes, tab_heart = st.tabs(["🍬  Diabetes", "❤️  Heart Disease"])

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

# 🩺 PredictWell — AI-Powered Multi-Disease Prediction System

An end-to-end machine learning system that predicts chronic disease risk
(diabetes, heart disease) from patient health records, with distributed
preprocessing in **Apache PySpark**, model training in **scikit-learn**,
and a real-time interactive prediction UI in **Streamlit**.

**Live demo:** _add your deployed link here after deploying (see below)_

---

## Architecture

```
Raw CSV data  ──►  PySpark preprocessing  ──►  Cleaned CSV
 (data/*.csv)      (src/preprocess_*.py)      (data/*_clean.csv)
                                                     │
                                                     ▼
                                     scikit-learn training
                                     (src/train_models.py)
                                     Logistic Regression vs
                                     Random Forest, best kept
                                                     │
                                                     ▼
                                        models/*.joblib + metadata
                                                     │
                                                     ▼
                                    Streamlit app (app.py)
                                    real-time risk prediction UI
```

## Project structure

```
predictwell/
├── app.py                     # Streamlit web app (deployed entry point)
├── requirements.txt           # Minimal deps needed to RUN the app
├── requirements-dev.txt       # Adds PySpark, needed to RE-RUN the pipeline
├── data/
│   ├── diabetes.csv           # Raw: Pima Indians Diabetes dataset
│   ├── heart.csv              # Raw: UCI Cleveland Heart Disease dataset
│   ├── diabetes_clean.csv     # Output of PySpark preprocessing
│   └── heart_clean.csv        # Output of PySpark preprocessing
├── src/
│   ├── preprocess_diabetes.py # PySpark: cleaning + feature engineering
│   ├── preprocess_heart.py    # PySpark: cleaning + feature engineering
│   └── train_models.py        # scikit-learn training + model selection
└── models/                    # Saved trained models (generated)
```

## What each stage does

1. **PySpark preprocessing** (`src/preprocess_*.py`)
   Distributed null/zero-value imputation (median, computed via
   `approxQuantile`), engineered features (BMI/cholesterol risk buckets,
   age buckets, glucose-insulin ratio, max heart-rate reserve), written
   back out as cleaned CSVs.

2. **Model training** (`src/train_models.py`)
   Trains Logistic Regression and Random Forest per disease, evaluates
   accuracy/precision/recall/F1 on a held-out test split, and saves the
   best-performing model + scaler + metadata to `models/`.

3. **Streamlit app** (`app.py`)
   Loads the saved models and serves an interactive UI where a user
   enters health parameters and gets an instant risk prediction with a
   color-coded risk badge.

## Measured model performance

> ⚠️ **Note on accuracy:** On these standard public datasets (768 diabetes
> records, 303 heart-disease records) with a held-out test split, the
> models we trained achieve:
> - **Diabetes (Random Forest):** ~75% accuracy
> - **Heart Disease (Random Forest):** ~80% accuracy
>
> This is in line with widely published benchmarks for these datasets —
> getting to ~90%+ on the *raw* Pima/UCI data usually means the split is
> too small/easy, features are leaking, or the model is overfitting.
> If your resume claims ~91%, consider either rephrasing to match these
> real, defensible numbers, or noting it as a stretch/aggregate figure —
> don't repeat 91% in an interview without being able to explain how you
> got it, since that's the first thing an interviewer will probe.

Run `python src/train_models.py` yourself any time to reproduce these
numbers — the script prints per-model accuracy/precision/recall/F1.

## Run it locally

```bash
# 1. Clone your repo (after you've pushed it, see below)
git clone https://github.com/<your-username>/predictwell.git
cd predictwell

# 2. Install app dependencies
pip install -r requirements.txt

# 3. Launch the app (models are already trained & committed in models/)
streamlit run app.py
```

To re-run the full pipeline from raw data (requires Java + PySpark):

```bash
pip install -r requirements-dev.txt
python src/preprocess_diabetes.py
python src/preprocess_heart.py
python src/train_models.py
```

## Push to GitHub

```bash
cd predictwell
git init
git add .
git commit -m "Initial commit: PredictWell multi-disease prediction system"
git branch -M main
git remote add origin https://github.com/<your-username>/predictwell.git
git push -u origin main
```

## Deploy a live link (Streamlit Community Cloud — free)

1. Push the project to a **public** GitHub repo (steps above).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, pick your `predictwell` repo, branch `main`,
   and set the main file path to `app.py`.
4. Click **Deploy**. You'll get a live link like
   `https://<your-app>.streamlit.app` in a couple of minutes.
5. Add that link to the top of this README and to your resume.

That's it — every future `git push` to `main` will auto-redeploy the app.

## Disclaimer

This is an educational/portfolio project. Predictions are not medical
advice and should not be used for real diagnostic decisions.

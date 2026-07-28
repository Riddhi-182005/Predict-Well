"""
PredictWell - Model Training
--------------------------------
Trains Logistic Regression and Random Forest classifiers (scikit-learn)
on the Spark-cleaned diabetes and heart-disease datasets, picks the
best performer per disease, and saves the model + scaler + metadata
to the models/ directory for the Streamlit app to load.
"""

import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train_disease_model(csv_path, label_col, feature_cols, disease_name, model_prefix):
    df = pd.read_csv(csv_path)
    X = df[feature_cols]
    y = df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=42
        ),
    }

    results = {}
    best_name, best_model, best_acc = None, None, -1

    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
        }
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    joblib.dump(best_model, MODELS_DIR / f"{model_prefix}_model.joblib")
    joblib.dump(scaler, MODELS_DIR / f"{model_prefix}_scaler.joblib")

    metadata = {
        "disease": disease_name,
        "feature_cols": feature_cols,
        "best_model": best_name,
        "best_accuracy": round(best_acc, 4),
        "all_results": results,
    }
    with open(MODELS_DIR / f"{model_prefix}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== {disease_name} ===")
    for name, r in results.items():
        print(f"  {name:22s} acc={r['accuracy']:.4f}  prec={r['precision']:.4f}  "
              f"rec={r['recall']:.4f}  f1={r['f1']:.4f}")
    print(f"  -> Best: {best_name} ({best_acc:.4%} accuracy)")

    return metadata


def main():
    diabetes_features = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "diabetes_pedigree", "age",
        "bmi_category", "age_bucket", "glucose_insulin_ratio"
    ]
    diabetes_meta = train_disease_model(
        "data/diabetes_clean.csv", "outcome", diabetes_features,
        "Diabetes", "diabetes"
    )

    heart_features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
        "age_bucket", "chol_risk", "max_hr_reserve"
    ]
    heart_meta = train_disease_model(
        "data/heart_clean.csv", "target", heart_features,
        "Heart Disease", "heart"
    )

    overall_acc = (diabetes_meta["best_accuracy"] + heart_meta["best_accuracy"]) / 2
    print(f"\nOverall average accuracy across both models: {overall_acc:.2%}")


if __name__ == "__main__":
    main()

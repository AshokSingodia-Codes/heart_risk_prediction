"""
Heart Disease Risk Prediction – Debugged Training Pipeline
==========================================================
"""

# ── 0. Imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, precision_score, f1_score, roc_auc_score
)

RANDOM_STATE = 42
TARGET = "Heart_Risk"

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("heart_disease_risk_dataset_earlymed.csv")
df.drop_duplicates(inplace=True)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# ── 2. Clinical threshold helpers ────────────────────────────────────────────
def pulse_to_binary(pulse: float) -> int:
    return int(pulse < 60 or pulse > 100)

def pulse_category(pulse: float) -> int:
    if pulse < 60:
        return 0
    if pulse <= 100:
        return 1
    return 2

def bp_binary(systolic: float, diastolic: float) -> int:
    return int(systolic >= 130 or diastolic >= 80)

def bp_stage(systolic: float, diastolic: float) -> int:
    if systolic >= 140 or diastolic >= 90:
        return 3
    if systolic >= 130 or diastolic >= 80:
        return 2
    if systolic >= 120 and diastolic < 80:
        return 1
    return 0

def age_group(age: int) -> int:
    if age < 40:
        return 0
    if age < 60:
        return 1
    return 2

def cholesterol_category(chol_mg_dl: float) -> int:
    if chol_mg_dl < 200:
        return 0
    if chol_mg_dl < 240:
        return 1
    return 2

def bmi_to_obesity(bmi: float) -> int:
    return int(bmi >= 30)

def bmi_category(bmi: float) -> int:
    if bmi < 18.5:
        return 0
    if bmi < 25:
        return 1
    if bmi < 30:
        return 2
    return 3

# ── 3. Feature Engineering ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["Symptom_Score"] = (
        d["Chest_Pain"] * 3 +
        d["Pain_Arms_Jaw_Back"] * 2 +
        d["Shortness_of_Breath"] * 2 +
        d["Cold_Sweats_Nausea"] * 2 +
        d["Palpitations"] * 1 +
        d["Fatigue"] * 1 +
        d["Dizziness"] * 1 +
        d["Swelling"] * 1
    )

    d["Lifestyle_Risk_Score"] = (
        d["Smoking"] * 3 +
        d["Sedentary_Lifestyle"] * 2 +
        d["Obesity"] * 2 +
        d["Chronic_Stress"] * 1
    )

    d["Clinical_Risk_Score"] = (
        d["High_BP"] * 3 +
        d["High_Cholesterol"] * 2 +
        d["Diabetes"] * 2 +
        d["Family_History"] * 2
    )

    d["Doctor_Risk_Score"] = (
        1.5 * d["Clinical_Risk_Score"] +
        1.2 * d["Symptom_Score"] +
        1.0 * d["Lifestyle_Risk_Score"]
    )

    d["Age_Group"] = d["Age"].apply(age_group)
    d["Senior_Flag"] = (d["Age"] >= 60).astype(int)

    d["Age_x_HighBP"] = d["Age"] * d["High_BP"]
    d["Age_x_Diabetes"] = d["Age"] * d["Diabetes"]
    d["HighBP_x_HighChol"] = d["High_BP"] * d["High_Cholesterol"]
    d["Smoking_x_HighChol"] = d["Smoking"] * d["High_Cholesterol"]
    d["Diabetes_x_Obesity"] = d["Diabetes"] * d["Obesity"]

    return d

df_eng = engineer_features(df)

# ── 4. Feature sets ──────────────────────────────────────────────────────────
ORIGINAL_FEATURES = [
    "Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations",
    "Dizziness", "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea",
    "High_BP", "High_Cholesterol", "Diabetes", "Smoking", "Obesity",
    "Sedentary_Lifestyle", "Family_History", "Chronic_Stress",
    "Gender", "Age"
]

ENGINEERED_FEATURES = ORIGINAL_FEATURES + [
    "Symptom_Score", "Lifestyle_Risk_Score", "Clinical_Risk_Score",
    "Doctor_Risk_Score", "Age_Group", "Senior_Flag",
    "Age_x_HighBP", "Age_x_Diabetes", "HighBP_x_HighChol",
    "Smoking_x_HighChol", "Diabetes_x_Obesity"
]

X_all = df_eng[ENGINEERED_FEATURES].copy()
y_all = df_eng[TARGET].copy()

# ── 5. Train / Validation / Test split ──────────────────────────────────────
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=RANDOM_STATE, stratify=y_all
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=y_trainval
)

print(f"Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"Target distribution — Train: {y_train.mean():.2%} positive\n")

# ── 6. Preprocessor ──────────────────────────────────────────────────────────
numeric_cols = [
    "Age", "Symptom_Score", "Lifestyle_Risk_Score", "Clinical_Risk_Score",
    "Doctor_Risk_Score", "Age_Group", "Age_x_HighBP", "Age_x_Diabetes"
]

binary_cols = [c for c in ENGINEERED_FEATURES if c not in numeric_cols]

preprocessor_eng = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("bin", "passthrough", binary_cols)
])

preprocessor_base = ColumnTransformer([
    ("num", StandardScaler(), ["Age"]),
    ("bin", "passthrough", [c for c in ORIGINAL_FEATURES if c != "Age"])
])

# ── 7. Evaluation helper ─────────────────────────────────────────────────────
def evaluate(y_true, y_prob, threshold=0.5, label="Model"):
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{'='*55}")
    print(f"{label} | threshold={threshold:.2f}")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, target_names=["No Risk", "High Risk"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    auc = roc_auc_score(y_true, y_prob)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Recall   : {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    return rec, prec, f1, auc

# ── 8. Model A – Baseline ────────────────────────────────────────────────────
pipe_baseline = Pipeline([
    ("prep", preprocessor_base),
    ("lr", LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE
    ))
])

pipe_baseline.fit(X_train[ORIGINAL_FEATURES], y_train)
prob_val_base = pipe_baseline.predict_proba(X_val[ORIGINAL_FEATURES])[:, 1]

print("\n──────────── BASELINE MODEL (validation) ────────────")
evaluate(y_val, prob_val_base, threshold=0.5, label="Baseline LR")

# ── 9. Model B – Engineered ──────────────────────────────────────────────────
pipe_eng = Pipeline([
    ("prep", preprocessor_eng),
    ("lr", LogisticRegression(
        C=0.5,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE
    ))
])

pipe_eng.fit(X_train, y_train)
prob_val_eng = pipe_eng.predict_proba(X_val)[:, 1]

print("\n──────────── ENGINEERED MODEL (validation) ──────────")
evaluate(y_val, prob_val_eng, threshold=0.5, label="Engineered LR")

# ── 10. Threshold tuning on VALIDATION set ───────────────────────────────────
print("\n──────────── THRESHOLD TUNING ON VALIDATION SET ─────────────────")
print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F1':>8} {'FN':>6}")

best_threshold = 0.5
best_score = -1

for th in np.linspace(0.50, 0.20, 16):
    y_th = (prob_val_eng >= th).astype(int)
    cm = confusion_matrix(y_val, y_th)
    fn = cm[1, 0]
    rec = recall_score(y_val, y_th)
    prec = precision_score(y_val, y_th, zero_division=0)
    f1 = f1_score(y_val, y_th, zero_division=0)

    score = 0.7 * rec + 0.3 * prec
    marker = ""

    if score > best_score:
        best_score = score
        best_threshold = round(float(th), 2)
        marker = " ← selected"

    print(f"{th:>10.2f} {rec:>8.4f} {prec:>10.4f} {f1:>8.4f} {fn:>6}{marker}")

print(f"\nSelected threshold: {best_threshold}")

# ── 11. Cross-validation sanity check ────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_auc = cross_val_score(pipe_eng, X_trainval, y_trainval, cv=cv, scoring="roc_auc")

print(f"\nCross-val ROC-AUC (5-fold, trainval): {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# ── 12. Final test evaluation ────────────────────────────────────────────────
prob_test = pipe_eng.predict_proba(X_test)[:, 1]

print("\n──────────── FINAL RESULTS ON TEST SET ──────────────────────────")
rec, prec, f1, auc = evaluate(
    y_test,
    prob_test,
    threshold=best_threshold,
    label="Engineered LR (final)"
)

print(f"\nSummary → Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# ── 13. Save model bundle ────────────────────────────────────────────────────
bundle = {
    "model": pipe_eng,
    "threshold": best_threshold,
    "feature_names": ENGINEERED_FEATURES,
    "original_features": ORIGINAL_FEATURES,
    "description": "Heart disease risk model with clinical feature engineering and validation-tuned threshold."
}

joblib.dump(bundle, "heart_disease_model.pkl")

print("\nSaved: heart_disease_model.pkl")
print("Bundle keys:", list(bundle.keys()))
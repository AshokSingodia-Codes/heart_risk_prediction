import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="CardioGuard AI",
    page_icon="🫀",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# Simple styling
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1f4e79;
    margin-bottom: 0.2rem;
}
.sub-title {
    color: #4b5563;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}
.box {
    padding: 1rem;
    border-radius: 12px;
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}
.result-low {
    padding: 1rem;
    border-radius: 12px;
    background: #ecfdf5;
    border-left: 6px solid #16a34a;
    color: #166534;
}
.result-high {
    padding: 1rem;
    border-radius: 12px;
    background: #fef2f2;
    border-left: 6px solid #dc2626;
    color: #991b1b;
}
.small-note {
    font-size: 0.9rem;
    color: #6b7280;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Helpers — same logic
# ─────────────────────────────────────────────────────────────
def pulse_to_binary(p):
    return int(p < 60 or p > 100)

def pulse_cat(p):
    if p < 60:
        return "Bradycardia"
    if p <= 100:
        return "Normal"
    return "Tachycardia"

def bp_binary(s, d):
    return int(s >= 130 or d >= 80)

def bp_stage(s, d):
    if s >= 140 or d >= 90:
        return "Stage 2 Hypertension"
    if s >= 130 or d >= 80:
        return "Stage 1 Hypertension"
    if s >= 120:
        return "Elevated"
    return "Normal"

def age_group(a):
    if a < 40:
        return 0
    if a < 60:
        return 1
    return 2

def chol_cat(c):
    if c < 200:
        return "Optimal"
    if c < 240:
        return "Borderline"
    return "High"

def bmi_to_obesity(b):
    return int(b >= 30)

def bmi_cat(b):
    if b < 18.5:
        return "Underweight"
    if b < 25:
        return "Normal"
    if b < 30:
        return "Overweight"
    return "Obese"

def engineer_features(df):
    d = df.copy()
    d["Symptom_Score"] = (
        d["Chest_Pain"] * 3
        + d["Pain_Arms_Jaw_Back"] * 2
        + d["Shortness_of_Breath"] * 2
        + d["Cold_Sweats_Nausea"] * 2
        + d["Palpitations"]
        + d["Fatigue"]
        + d["Dizziness"]
        + d["Swelling"]
    )
    d["Lifestyle_Risk_Score"] = (
        d["Smoking"] * 3
        + d["Sedentary_Lifestyle"] * 2
        + d["Obesity"] * 2
        + d["Chronic_Stress"]
    )
    d["Clinical_Risk_Score"] = (
        d["High_BP"] * 3
        + d["High_Cholesterol"] * 2
        + d["Diabetes"] * 2
        + d["Family_History"] * 2
    )
    d["Doctor_Risk_Score"] = (
        1.5 * d["Clinical_Risk_Score"]
        + 1.2 * d["Symptom_Score"]
        + d["Lifestyle_Risk_Score"]
    )
    d["Age_Group"] = d["Age"].apply(age_group)
    d["Senior_Flag"] = (d["Age"] >= 60).astype(int)
    d["Age_x_HighBP"] = d["Age"] * d["High_BP"]
    d["Age_x_Diabetes"] = d["Age"] * d["Diabetes"]
    d["HighBP_x_HighChol"] = d["High_BP"] * d["High_Cholesterol"]
    d["Smoking_x_HighChol"] = d["Smoking"] * d["High_Cholesterol"]
    d["Diabetes_x_Obesity"] = d["Diabetes"] * d["Obesity"]
    return d

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")

try:
    bundle = load_model()
    model = bundle["model"]
    THRESHOLD = bundle["threshold"]
    FEATURE_NAMES = bundle["feature_names"]
except FileNotFoundError:
    st.error("heart_disease_model.pkl not found. Please run train_model.py first.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🫀 CardioGuard AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Simple heart risk assessment tool for healthcare screening.</div>',
    unsafe_allow_html=True
)

st.info("Fill the patient details below and click **Assess Risk**.")

# ─────────────────────────────────────────────────────────────
# Form
# ─────────────────────────────────────────────────────────────
with st.form("cardio_form"):

    st.subheader("1. Basic Information")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 90, 45)
    with c2:
        gender = st.selectbox("Biological Sex", ["Male", "Female"])

    st.subheader("2. Vital Signs")
    v1, v2, v3, v4, v5 = st.columns(5)
    with v1:
        pulse = st.number_input("Pulse (bpm)", 30, 200, 75)
    with v2:
        systolic_bp = st.number_input("Systolic BP", 70, 220, 120)
    with v3:
        diastolic_bp = st.number_input("Diastolic BP", 40, 140, 78)
    with v4:
        bmi = st.number_input("BMI", 10.0, 60.0, 23.5, step=0.1)
    with v5:
        chol_value = st.number_input("Cholesterol", 50, 500, 185)

    st.subheader("3. Symptoms")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        chest_pain = st.checkbox("Chest Pain / Tightness")
        palpitations = st.checkbox("Palpitations")
    with s2:
        pain_arms = st.checkbox("Pain in Arms / Jaw / Back")
        fatigue = st.checkbox("Fatigue / Weakness")
    with s3:
        sob = st.checkbox("Shortness of Breath")
        dizziness = st.checkbox("Dizziness")
    with s4:
        cold_sweats = st.checkbox("Cold Sweats / Nausea")
        swelling = st.checkbox("Leg / Ankle Swelling")

    st.subheader("4. Medical History & Lifestyle")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        diabetes = st.checkbox("Diabetes")
        smoking = st.checkbox("Smoking")
    with m2:
        family_hist = st.checkbox("Family History of CVD")
        high_chol_cb = st.checkbox("Known High Cholesterol")
    with m3:
        sedentary = st.checkbox("Sedentary Lifestyle")
    with m4:
        chronic_stress = st.checkbox("Chronic Stress / Anxiety")

    submitted = st.form_submit_button("Assess Risk", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Quick health summary
# ─────────────────────────────────────────────────────────────
st.subheader("Current Health Summary")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Heart Rate", f"{pulse} bpm", pulse_cat(pulse))
q2.metric("Blood Pressure", f"{systolic_bp}/{diastolic_bp}", bp_stage(systolic_bp, diastolic_bp))
q3.metric("BMI", f"{bmi:.1f}", bmi_cat(bmi))
q4.metric("Cholesterol", f"{chol_value} mg/dL", chol_cat(chol_value))

# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────
if submitted:
    high_bp = bp_binary(systolic_bp, diastolic_bp)
    obesity = bmi_to_obesity(bmi)
    high_chol = int(chol_value >= 200) or int(high_chol_cb)

    raw = pd.DataFrame([{
        "Chest_Pain": int(chest_pain),
        "Shortness_of_Breath": int(sob),
        "Fatigue": int(fatigue),
        "Palpitations": int(palpitations),
        "Dizziness": int(dizziness),
        "Swelling": int(swelling),
        "Pain_Arms_Jaw_Back": int(pain_arms),
        "Cold_Sweats_Nausea": int(cold_sweats),
        "High_BP": high_bp,
        "High_Cholesterol": high_chol,
        "Diabetes": int(diabetes),
        "Smoking": int(smoking),
        "Obesity": obesity,
        "Sedentary_Lifestyle": int(sedentary),
        "Family_History": int(family_hist),
        "Chronic_Stress": int(chronic_stress),
        "Gender": 1 if gender == "Male" else 0,
        "Age": age,
    }])

    row_eng = engineer_features(raw)
    X_input = row_eng[FEATURE_NAMES]
    prob = model.predict_proba(X_input)[0][1]
    risk_flag = int(prob >= THRESHOLD)
    pct = round(prob * 100, 1)

    symp_score = int(row_eng["Symptom_Score"].iloc[0])
    life_score = int(row_eng["Lifestyle_Risk_Score"].iloc[0])
    clin_score = int(row_eng["Clinical_Risk_Score"].iloc[0])
    doc_score = round(float(row_eng["Doctor_Risk_Score"].iloc[0]), 1)

    st.markdown("---")
    st.subheader("Assessment Result")

    a1, a2 = st.columns([1, 1.2])

    with a1:
        gauge_color = "#dc2626" if risk_flag else "#16a34a"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%"},
            title={"text": "Risk Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 55], "color": "#fef3c7"},
                    {"range": [55, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "value": THRESHOLD * 100
                }
            }
        ))
        fig.update_layout(height=300, margin=dict(t=50, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

    with a2:
        if risk_flag == 0:
            st.markdown("""
            <div class="result-low">
                <h4>Low Risk</h4>
                <p>No immediate major cardiac warning is detected from the entered data.</p>
                <p>Still, maintain regular exercise, healthy diet, and routine checkups.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-high">
                <h4>Elevated Risk</h4>
                <p>The entered data suggests an increased cardiovascular risk.</p>
                <p>Please consult a qualified doctor or cardiologist for proper evaluation.</p>
            </div>
            """, unsafe_allow_html=True)

        st.write(f"**Model threshold:** {round(THRESHOLD * 100, 1)}%")
        st.write(f"**Predicted probability:** {pct}%")

    st.subheader("Score Breakdown")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Symptom Score", symp_score)
    b2.metric("Lifestyle Risk", life_score)
    b3.metric("Clinical Risk", clin_score)
    b4.metric("Doctor Risk Score", doc_score)

    st.subheader("Important Risk Factors")
    active_risks = []

    if chest_pain:
        active_risks.append("Chest Pain")
    if pain_arms:
        active_risks.append("Pain in Arms / Jaw / Back")
    if sob:
        active_risks.append("Shortness of Breath")
    if cold_sweats:
        active_risks.append("Cold Sweats / Nausea")
    if palpitations:
        active_risks.append("Palpitations")
    if fatigue:
        active_risks.append("Fatigue")
    if dizziness:
        active_risks.append("Dizziness")
    if swelling:
        active_risks.append("Swelling")
    if high_bp:
        active_risks.append("High Blood Pressure")
    if high_chol:
        active_risks.append("High Cholesterol")
    if diabetes:
        active_risks.append("Diabetes")
    if smoking:
        active_risks.append("Smoking")
    if obesity:
        active_risks.append("Obesity")
    if sedentary:
        active_risks.append("Sedentary Lifestyle")
    if family_hist:
        active_risks.append("Family History of CVD")
    if chronic_stress:
        active_risks.append("Chronic Stress")

    if active_risks:
        st.write(", ".join(active_risks))
    else:
        st.success("No major risk factors selected.")

    if risk_flag:
        st.subheader("Recommended Next Steps")
        st.warning("Consult a doctor soon if symptoms are persistent or worsening.")

        if chest_pain or (sob and cold_sweats):
            st.error("Emergency warning: sudden chest pain, breathlessness, or cold sweats may need urgent medical attention.")

        st.markdown("""
        - ECG
        - Lipid Profile
        - Blood Sugar / HbA1c
        - Blood Pressure Monitoring
        - Doctor / Cardiologist Consultation
        """)
    else:
        st.subheader("Prevention Advice")
        st.markdown("""
        - Exercise regularly
        - Eat a heart-healthy diet
        - Avoid smoking
        - Sleep 7–8 hours daily
        - Monitor BP, cholesterol, and blood sugar regularly
        """)

    st.markdown("---")
    st.markdown(
        '<div class="small-note"><b>Disclaimer:</b> This is an AI-based screening tool for educational purposes only. It is not a medical diagnosis. In case of severe symptoms, seek immediate medical care.</div>',
        unsafe_allow_html=True
    )
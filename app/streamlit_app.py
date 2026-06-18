# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
from pathlib import Path
from huggingface_hub import hf_hub_download
import os

REPO_ID = "MonyetttRindam/ckdbinary"

# ------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------
st.set_page_config(page_title="CKD Early Screening", layout="wide", page_icon="🩺")

# ------------------------------------------------------------
# Custom CSS (konsisten dengan EDA Toolkit)
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #F0F4F8;
}

/* Hero */
.eda-hero {
    background: linear-gradient(135deg, #1B3A6B 0%, #1A7A8A 100%);
    border-radius: 16px;
    padding: 28px 36px 24px 36px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(27, 58, 107, 0.18);
}
.eda-hero h1 {
    color: #FFFFFF;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
}
.eda-hero p {
    color: #B8DCEB;
    font-size: 0.9rem;
    margin: 0;
    line-height: 1.6;
}

/* Chart / Form card */
.chart-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 22px 24px;
    border: 1px solid #DDE4EE;
    box-shadow: 0 2px 8px rgba(27, 58, 107, 0.06);
    margin-bottom: 20px;
}

/* Section label */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #1A7A8A;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #E3EDF5;
}

/* Result cards */
.result-card-safe {
    background: linear-gradient(135deg, #E8F8F0 0%, #D0F0E0 100%);
    border: 1.5px solid #27AE60;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 16px;
}
.result-card-safe h2 {
    color: #1A7A3A;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0 0 6px 0;
}
.result-card-safe p {
    color: #2E7D52;
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.6;
}

.result-card-danger {
    background: linear-gradient(135deg, #FEF0EC 0%, #FDDDD3 100%);
    border: 1.5px solid #E74C3C;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 16px;
}
.result-card-danger h2 {
    color: #C0392B;
    font-size: 1.3rem;
    font-weight: 700;
    margin: 0 0 6px 0;
}
.result-card-danger p {
    color: #922B21;
    font-size: 0.88rem;
    margin: 0;
    line-height: 1.6;
}

/* Stage badge */
.stage-badge {
    display: inline-block;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 700;
    font-size: 0.95rem;
    margin-top: 12px;
}
.stage-red   { background: #FDECEA; color: #C0392B; border: 1px solid #E74C3C; }
.stage-orange{ background: #FEF5E7; color: #B7770D; border: 1px solid #F39C12; }
.stage-green { background: #E9F7EF; color: #1A7A3A; border: 1px solid #27AE60; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1B3A6B !important;
}
section[data-testid="stSidebar"] * {
    color: #E8EFF8 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #B8DCEB !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 16px !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}

/* Dataframe */
.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden;
}

/* Submit button */
div.stFormSubmitButton > button {
    background: linear-gradient(135deg, #1B3A6B 0%, #1A7A8A 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    padding: 12px 36px;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    cursor: pointer;
    width: 100%;
    margin-top: 8px;
}
div.stFormSubmitButton > button:hover {
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load model dan scaler
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    rf_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="random_forest_ckd.pkl"
    )
    model_binary = joblib.load(rf_path)

    xgb_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="xgboost_ckd.pkl"
    )
    model_multiclass = joblib.load(xgb_path)

    scaler_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="scaler.pkl"
    )
    scaler = joblib.load(scaler_path)

    return model_binary, model_multiclass, scaler

try:
    model_binary, model_multiclass, scaler = load_models()
    st.sidebar.markdown("### 🩺 CKD Screening")
    st.sidebar.markdown("---")
    st.sidebar.success("✅ Model siap digunakan")
except Exception as e:
    st.error(f"Gagal memuat dari Hub: {e}")
    st.stop()

# ------------------------------------------------------------
# Fungsi preprocessing (TIDAK DIUBAH)
# ------------------------------------------------------------
def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    survey_cols = ['insulin_use', 'diabetes_pills', 'ever_smoked',
                   'current_smoker', 'diabetes_diagnosed', 'education_level']
    df[survey_cols] = df[survey_cols].replace([7, 9], np.nan)

    df['insulin_use'] = df['insulin_use'].fillna(0)
    df['diabetes_pills'] = df['diabetes_pills'].fillna(0)

    lab_bp_cols = ['phosphorus', 'bicarbonate', 'calcium', 'uric_acid',
                   'bp_systolic', 'bp_diastolic']
    for col in lab_bp_cols:
        df[col] = df[col].fillna(df[col].median())

    def create_smoking_status(row):
        ever = row['ever_smoked']
        current = row['current_smoker']
        if ever == 2:
            return 'never'
        elif ever == 1 and current == 1:
            return 'current'
        elif ever == 1 and current == 2:
            return 'former'
        return np.nan
    df['smoking_status'] = df.apply(create_smoking_status, axis=1)
    df.drop(columns=['ever_smoked', 'current_smoker'], inplace=True)

    def get_age_group(age):
        if age < 18:
            return 'child'
        elif age < 60:
            return 'adult'
        else:
            return 'elderly'
    df['age_group'] = df['age'].apply(get_age_group)

    for col in ['bmi', 'weight_kg', 'height_cm']:
        df[col] = df[col].fillna(df[col].median())

    df['education_level'] = df['education_level'].fillna(df['education_level'].mode()[0])
    df['poverty_income_ratio'] = df['poverty_income_ratio'].fillna(df['poverty_income_ratio'].median())
    df['smoking_status'] = df['smoking_status'].fillna('unknown')

    df['MAP'] = (df['bp_systolic'] + 2 * df['bp_diastolic']) / 3
    df.drop(columns=['bp_systolic', 'bp_diastolic'], inplace=True)

    def bmi_category(bmi):
        if pd.isna(bmi):
            return np.nan
        if bmi < 18.5:
            return 'underweight'
        if bmi < 25.0:
            return 'normal'
        if bmi < 30.0:
            return 'overweight'
        return 'obese'
    df['bmi_category'] = df['bmi'].apply(bmi_category)

    df.drop(columns=['weight_kg', 'bmi'], inplace=True)

    education_order = [1, 2, 3, 4, 5]
    age_group_order = ['child', 'adult', 'elderly']
    bmi_order = ['underweight', 'normal', 'overweight', 'obese']
    smoking_order = ['never', 'unknown', 'former', 'current']

    ordinal_map = {
        'education_level': education_order,
        'age_group': age_group_order,
        'bmi_category': bmi_order,
        'smoking_status': smoking_order
    }
    for col, order in ordinal_map.items():
        mapping = {val: idx for idx, val in enumerate(order)}
        df[col] = df[col].map(mapping)

    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['diabetes_diagnosed'] = df['diabetes_diagnosed'].map({1: 1, 2: 0})
    df['insulin_use'] = df['insulin_use'].astype(int)
    df['diabetes_pills'] = df['diabetes_pills'].astype(int)

    ethnicity_categories = ['Non-Hispanic Asian', 'Non-Hispanic White', 'Other Hispanic',
                            'Other/Multiracial', 'Mexican American', 'Non-Hispanic Black']
    df = pd.get_dummies(df, columns=['ethnicity'], drop_first=True, dtype=int)
    for cat in ethnicity_categories[1:]:
        col_name = f'ethnicity_{cat}'
        if col_name not in df.columns:
            df[col_name] = 0

    clinical_bounds = {
        'height_cm': (100.0, 210.0),
        'uric_acid': (1.5, 13.0),
        'calcium': (7.0, 12.0),
        'bicarbonate': (10.0, 40.0),
        'phosphorus': (1.0, 9.0),
    }
    for col, (low, high) in clinical_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(low, high)

    expected_cols = model_binary.feature_names_in_
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    continuous_cols = ['age', 'poverty_income_ratio', 'height_cm',
                       'phosphorus', 'bicarbonate', 'calcium',
                       'uric_acid', 'MAP']
    for col in continuous_cols:
        if col not in df.columns:
            df[col] = 0
    df[continuous_cols] = scaler.transform(df[continuous_cols])

    return df

# ------------------------------------------------------------
# Hero
# ------------------------------------------------------------
st.markdown("""
<div class="eda-hero">
    <h1>🩺 Skrining Awal Penyakit Ginjal Kronis (CKD)</h1>
    <p>Isi data pasien berikut untuk mengetahui risiko CKD dan staging keparahan menggunakan model <strong>Random Forest</strong> & <strong>XGBoost</strong>.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Sidebar info
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Panduan Pengisian")
st.sidebar.markdown(
    '<p style="font-size:0.82rem; color:#B8DCEB; line-height:1.7;">'
    '• Isi semua kolom dengan data aktual pasien.<br>'
    '• Data lab sebaiknya dari hasil pemeriksaan terbaru.<br>'
    '• Hasil skrining bukan pengganti diagnosis dokter.</p>',
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# Form input data
# ------------------------------------------------------------
with st.form("input_form"):

    # --- Seksi 1: Demografi & Klinis ---
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📋 Data Demografi & Klinis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia (tahun)", min_value=0, max_value=120, value=50, step=1)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        ethnicity = st.selectbox("Etnis", [
            "Non-Hispanic Asian", "Non-Hispanic White", "Other Hispanic",
            "Other/Multiracial", "Mexican American", "Non-Hispanic Black"
        ])
        education_level = st.selectbox(
            "Tingkat Pendidikan (1–5)", [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "1 – Tidak tamat SD", 2: "2 – SD",
                3: "3 – SMP", 4: "4 – SMA", 5: "5 – Perguruan Tinggi"
            }[x]
        )
        poverty_income_ratio = st.number_input("Poverty Income Ratio (0–5)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)

    with col2:
        height_cm = st.number_input("Tinggi badan (cm)", min_value=100.0, max_value=210.0, value=165.0, step=0.1)
        weight_kg = st.number_input("Berat badan (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
        bp_systolic = st.number_input("Tekanan darah sistolik (mmHg)", min_value=70, max_value=250, value=120, step=1)
        bp_diastolic = st.number_input("Tekanan darah diastolik (mmHg)", min_value=40, max_value=150, value=80, step=1)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Seksi 2: Hasil Laboratorium ---
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🧪 Hasil Laboratorium</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        phosphorus = st.number_input("Fosfor (mg/dL)", min_value=1.0, max_value=9.0, value=3.5, step=0.1)
        bicarbonate = st.number_input("Bikarbonat (mmol/L)", min_value=10.0, max_value=40.0, value=24.0, step=0.1)
        calcium = st.number_input("Kalsium (mg/dL)", min_value=7.0, max_value=12.0, value=9.0, step=0.1)
    with col4:
        uric_acid = st.number_input("Asam urat (mg/dL)", min_value=1.5, max_value=13.0, value=5.0, step=0.1)
        diabetes_diagnosed = st.selectbox(
            "Diabetes didiagnosis?", [("Ya", 1), ("Tidak", 2)],
            format_func=lambda x: x[0]
        )[1]
        insulin_use = st.selectbox(
            "Penggunaan insulin", [("Ya", 1), ("Tidak", 0)],
            format_func=lambda x: x[0]
        )[1]
        diabetes_pills = st.selectbox(
            "Obat diabetes oral", [("Ya", 1), ("Tidak", 0)],
            format_func=lambda x: x[0]
        )[1]

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Seksi 3: Riwayat Merokok ---
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🚬 Riwayat Merokok</div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        ever_smoked = st.selectbox(
            "Pernah merokok (minimal 100 batang seumur hidup)?",
            [("Ya", 1), ("Tidak", 2)],
            format_func=lambda x: x[0]
        )[1]
    with col6:
        current_smoker = st.selectbox(
            "Saat ini merokok?",
            [("Ya, setiap hari atau kadang", 1), ("Tidak, sudah berhenti", 2), ("Tidak pernah", 3)],
            format_func=lambda x: x[0]
        )[1]

    st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍 Lakukan Skrining")

# ------------------------------------------------------------
# Jika tombol ditekan, proses prediksi (TIDAK DIUBAH)
# ------------------------------------------------------------
if submitted:
    input_data = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'ethnicity': ethnicity,
        'education_level': education_level,
        'poverty_income_ratio': poverty_income_ratio,
        'bmi': bmi,
        'weight_kg': weight_kg,
        'height_cm': height_cm,
        'bp_systolic': bp_systolic,
        'bp_diastolic': bp_diastolic,
        'phosphorus': phosphorus,
        'bicarbonate': bicarbonate,
        'calcium': calcium,
        'uric_acid': uric_acid,
        'diabetes_diagnosed': diabetes_diagnosed,
        'insulin_use': insulin_use,
        'diabetes_pills': diabetes_pills,
        'ever_smoked': ever_smoked,
        'current_smoker': current_smoker
    }])

    try:
        X_processed = preprocess_input(input_data)
        pred_binary = model_binary.predict(X_processed)[0]

        st.markdown('<div class="section-label" style="margin-top:8px;">📊 Hasil Skrining</div>', unsafe_allow_html=True)

        if pred_binary == 0:
            st.markdown("""
            <div class="result-card-safe">
                <h2>✅ TIDAK TERDETEKSI CKD</h2>
                <p>Ginjal pasien dalam kondisi normal berdasarkan data yang dimasukkan. Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin.</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            pred_stage = model_multiclass.predict(X_processed)[0]

            stage_info = {
                0: ("stage-red",   "🔴 Bahaya",       "Stadium lanjut — perlu penanganan segera oleh spesialis ginjal."),
                1: ("stage-orange","🟠 Cukup Bahaya",  "Stadium sedang — segera konsultasi ke dokter nefrologi."),
                2: ("stage-green", "🟢 Batas Aman",    "Stadium awal — modifikasi gaya hidup dan pantau secara rutin.")
            }
            badge_cls, stage_label, stage_desc = stage_info[pred_stage]

            st.markdown("""
            <div class="result-card-danger">
                <h2>⚠️ TERDETEKSI CKD</h2>
                <p>Risiko penyakit ginjal kronis terdeteksi. Segera konsultasikan hasil ini ke dokter untuk penanganan lebih lanjut.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="chart-card" style="margin-top:0;">
                <div class="section-label">Staging Keparahan CKD</div>
                <span class="stage-badge {badge_cls}">{stage_label}</span>
                <p style="margin-top:12px; font-size:0.88rem; color:#4A5568;">{stage_desc}</p>
                <p style="font-size:0.82rem; color:#718096; margin-top:8px;">
                    ⚕️ Hasil ini bersifat indikatif. Diagnosis akhir harus dilakukan oleh tenaga medis profesional.
                </p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan dalam pemrosesan data: {e}")
        st.markdown(
            '<div class="chart-card"><p style="color:#718096; font-size:0.88rem;">'
            'Pastikan semua data terisi dengan benar dan model/scaler tersedia.</p></div>',
            unsafe_allow_html=True
        )

# ------------------------------------------------------------
# Sidebar footer
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="font-size:0.75rem; color:#7A9DC0; text-align:center;">CKD Early Screening · ML Module</p>',
    unsafe_allow_html=True
)

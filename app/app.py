import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Konfigurasi Halaman
st.set_page_config(
    page_title="Burnout Detector AI",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS untuk mempercantik tampilan
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True) # <-- SUDAH DIPERBAIKI

# Fungsi Load Model & Scaler
@st.cache_resource
def load_assets():
    with open('models/burnout_svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/caler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("⚠️ File model.pkl atau scaler.pkl tidak ditemukan! Pastikan sudah diekspor.")
    st.stop()

# Header
st.title("🔥 Burnout Risk Prediction System")
st.markdown("Analisis risiko kelelahan kerja untuk pekerja remote berbasis Artificial Intelligence.")
st.divider()

# Sidebar untuk Input
with st.sidebar:
    st.header("📋 Parameter Input")
    st.info("Sesuaikan data di bawah ini untuk melihat prediksi.")
    
    work_hours = st.slider("Jam Kerja (per hari)", 0.0, 16.0, 8.0)
    sleep_hours = st.slider("Jam Tidur (per hari)", 0.0, 12.0, 7.0)
    meetings = st.number_input("Jumlah Meeting (per hari)", 0, 20, 3)
    fatigue_score = st.slider("Skor Kelelahan (0-10)", 0.0, 10.0, 5.0)
    
    st.divider()
    predict_btn = st.button("🚀 Cek Risiko Sekarang")

# Area Utama (Dashboard)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Ringkasan Data")
    # Menampilkan data input dalam bentuk metrik
    m1, m2, m3 = st.columns(3)
    m1.metric("Work Hours", f"{work_hours}h")
    m2.metric("Sleep Hours", f"{sleep_hours}h")
    m3.metric("Fatigue Score", f"{fatigue_score}/10")
    
    # Visualisasi sederhana (Opsional)
    chart_data = pd.DataFrame({
        'Kategori': ['Kerja', 'Tidur'],
        'Jam': [work_hours, sleep_hours]
    })
    st.bar_chart(chart_data.set_index('Kategori'))

with col2:
    st.subheader("Hasil Analisis")
    if predict_btn:
        # Preprocessing & Prediksi
        # Pastikan urutan fitur ini sama dengan saat training!
        features = np.array([[work_hours, sleep_hours, meetings, fatigue_score]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # Tampilan Hasil yang Interaktif
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        if prediction == "High": # Sesuaikan dengan label di dataset Anda
            st.error(f"### 🚨 RISIKO TINGGI")
            st.write("Sistem mendeteksi indikasi burnout yang signifikan. Disarankan untuk segera mengambil istirahat.")
        elif prediction == "Medium":
            st.warning(f"### ⚠️ RISIKO SEDANG")
            st.write("Anda mulai menunjukkan gejala kelelahan. Perhatikan keseimbangan waktu kerja.")
        else:
            st.success(f"### ✅ RISIKO RENDAH")
            st.write("Kondisi Anda terpantau stabil. Tetap jaga pola kerja yang sehat!")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Klik tombol di sidebar untuk melihat hasil prediksi.")

# Footer
st.divider()
st.caption(f"Developed by {st.get_option('browser.gatherUsageStats') and 'Muhammad Fathan Fuad' or 'King'} | © {datetime.now().year}")
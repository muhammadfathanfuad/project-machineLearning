import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Burnout Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_assets():
    model = joblib.load('models/burnout_svm_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- CUSTOM CSS UNTUK TAMPILAN KEREN ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 24px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: INFORMASI ---
with st.sidebar:
    st.title("Settings & Info")
    st.info("Aplikasi ini memprediksi risiko burnout berdasarkan gaya hidup dan lingkungan kerja remote.")
    st.image("https://cdn-icons-png.flaticon.com/512/2038/2038030.png", width=100)
    st.markdown("---")
    st.markdown("Developed for: **Case Project ML**")

# --- HEADER ---
st.title("🧠 Remote Worker Burnout Predictor")
st.markdown("Silakan lengkapi data di bawah ini untuk melihat estimasi risiko burnout Anda.")
st.divider()

# --- FORMULIR INPUT ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Metrik Kerja")
    work_hours = st.number_input("Jam Kerja Per Hari", 0.0, 24.0, 8.0)
    screen_time = st.number_input("Screen Time (Jam)", 0.0, 24.0, 6.0)
    meetings = st.slider("Jumlah Meeting", 0, 20, 3)
    task_completion = st.slider("Penyelesaian Tugas (%)", 0, 100, 80)

with col2:
    st.subheader("🧘 Gaya Hidup")
    sleep_hours = st.number_input("Jam Tidur", 0.0, 24.0, 7.0)
    breaks = st.slider("Jumlah Istirahat", 0, 15, 4)
    fatigue = st.slider("Skor Kelelahan (1-10)", 1, 10, 5)

with col3:
    st.subheader("🏠 Lingkungan")
    isolation = st.slider("Indeks Isolasi (1-10)", 1, 10, 3)
    app_switches = st.number_input("Ganti Aplikasi (Switching)", 0, 500, 50)
    after_hours = st.selectbox("Kerja Setelah Jam Kantor?", ["Tidak", "Ya"])

# Konversi input ke format model
after_hours_val = 1 if after_hours == "Ya" else 0

# --- PROSES PREDIKSI ---
if st.button("PREDIKSI SEKARANG"):
    # Gabungkan data sesuai urutan fitur saat training
    input_data = np.array([[
        work_hours, screen_time, meetings, breaks, 
        after_hours_val, app_switches, sleep_hours, 
        task_completion, isolation, fatigue
    ]])
    
    with st.spinner('Menganalisis data Anda...'):
        time.sleep(1) # Efek loading
        
        # Scaling dan Prediksi
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        # Penentuan Warna Berdasarkan Hasil
        color = "#4CAF50" # Low
        if prediction == "Medium":
            color = "#FF9800"
        elif prediction == "High":
            color = "#F44336"
            
        # Tampilkan Hasil
        st.markdown(f"""
            <div class="prediction-card" style="background-color: {color};">
                Risiko Burnout Anda: {prediction.upper()}
            </div>
            """, unsafe_allow_html=True)
        
        # Rekomendasi Sederhana
        st.write("### 💡 Rekomendasi:")
        if prediction == "Low":
            st.success("Kerja bagus! Pertahankan keseimbangan hidup dan kerja Anda.")
        elif prediction == "Medium":
            st.warning("Hati-hati. Mulailah mengatur waktu istirahat lebih teratur dan batasi jam kerja lembur.")
        else:
            st.error("Peringatan! Anda berada di zona risiko tinggi. Pertimbangkan untuk mengambil cuti atau berkonsultasi mengenai beban kerja Anda.")

# --- FOOTER ---
st.divider()
st.caption("Aplikasi Prediksi Burnout | v1.0.0")
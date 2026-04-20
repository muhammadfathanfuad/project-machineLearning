import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Konfigurasi Halaman
st.set_page_config(
    page_title="Burnout Detector PRO",
    page_icon="🧠",
    layout="wide"
)

# Judul Utama
st.title("🧠 Burnout Risk Detector - Pekerja Remote")
st.markdown("""
Aplikasi ini menggunakan model **Machine Learning SVM** untuk mendeteksi tingkat risiko burnout 
berdasarkan faktor gaya hidup dan lingkungan kerja.
""")

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_assets():
    with open('models/burnout_svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("Error: File model.pkl atau scaler.pkl tidak ditemukan!")
    st.stop()

# --- SIDEBAR: PANDUAN PENGGUNAAN ---
with st.sidebar:
    st.header("📖 Panduan Penggunaan")
    st.info("""
    1. Isi data gaya hidup & kerja di panel utama.
    2. Perhatikan input 'Jam Kerja' & 'Jam Tidur'.
    3. Klik tombol **Prediksi Sekarang**.
    4. Hasil risiko (Low, Medium, High) akan muncul.
    """)
    st.warning("⚠️ *Aplikasi ini adalah alat bantu deteksi dini, bukan diagnosis klinis.*")

# --- INPUT AREA ---
st.subheader("📝 Masukkan Data Harian Anda")
col1, col2, col3 = st.columns(3)

with col1:
    day_type = st.selectbox("Tipe Hari", ["Weekday", "Weekend"])
    work_hours = st.number_input("Jam Kerja (jam)", 1.0, 24.0, 8.0)
    screen_time = st.number_input("Screen Time (jam)", 1.0, 24.0, 6.0)
    meetings = st.slider("Jumlah Meeting", 0, 20, 2)

with col2:
    breaks = st.slider("Frekuensi Istirahat", 0, 10, 3)
    after_hours = st.number_input("Jam Lembur (jam)", 0.0, 10.0, 0.0)
    app_switches = st.slider("App Switches (Multitasking)", 1, 100, 20)
    sleep_hours = st.number_input("Jam Tidur (jam)", 1.0, 12.0, 7.0)

with col3:
    task_completion = st.slider("Tugas Selesai (%)", 0, 100, 85)
    isolation_index = st.slider("Indeks Isolasi (1-10)", 1, 10, 3)
    fatigue_score = st.slider("Skor Kelelahan (1-10)", 1.0, 10.0, 2.0)

# --- PROSES PREDIKSI ---
if st.button("🚀 Prediksi Sekarang", use_container_width=True):
    # Mapping Data Input ke DataFrame
    input_data = pd.DataFrame({
        'work_hours': [work_hours],
        'screen_time_hours': [screen_time],
        'meetings_count': [meetings],
        'breaks_taken': [breaks],
        'after_hours_work': [after_hours],
        'app_switches': [app_switches],
        'sleep_hours': [sleep_hours],
        'task_completion': [task_completion],
        'isolation_index': [isolation_index],
        'fatigue_score': [fatigue_score],
        'day_type_Weekend': [1 if day_type == "Weekend" else 0]
    })

    # Scaling
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    # Hasil
    res_map = {0: "Low", 1: "Medium", 2: "High"}
    res_text = res_map[prediction]
    
    # Tampilan Hasil Keren
    st.divider()
    if res_text == "Low":
        st.success(f"### Hasil Prediksi: **{res_text} Risk** ✅")
        st.write("Kesehatan mental Anda terjaga dengan baik. Pertahankan keseimbangan kerja ini!")
    elif res_text == "Medium":
        st.warning(f"### Hasil Prediksi: **{res_text} Risk** ⚠️")
        st.write("Anda mulai menunjukkan gejala burnout. Cobalah untuk mengambil istirahat lebih sering.")
    else:
        st.error(f"### Hasil Prediksi: **{res_text} Risk** 🚨")
        st.write("Waspada! Risiko burnout Anda sangat tinggi. Segera kurangi beban kerja dan konsultasikan kesehatan Anda.")
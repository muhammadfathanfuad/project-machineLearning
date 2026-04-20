import streamlit as st
import pandas as pd
import pickle
import time

# Konfigurasi Halaman
st.set_page_config(page_title="Ruang Cerita - Digital Wellness", page_icon="☕")

# --- LOAD MODEL ---
@st.cache_resource
def load_assets():
    with open('burnout_svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# --- CUSTOM CSS UNTUK STYLE PSIKOLOG ---
st.markdown("""
    <style>
    .stApp { background-color: #fdfaf6; }
    .chat-bubble {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #6c5b7b;
        margin-bottom: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("☕ Ruang Cerita")
st.write("Selamat datang. Mari duduk sejenak. Aku ingin mendengar bagaimana harimu akhir-akhir ini.")

# --- FORMULIR BERBASIS NARASI (STORYTELLING) ---
with st.container():
    st.markdown('<div class="chat-bubble">"Boleh aku tahu, apakah hari yang ingin kamu ceritakan ini adalah hari kerja biasa atau hari libur?"</div>', unsafe_allow_html=True)
    day_type = st.radio("", ["Weekday", "Weekend"], label_visibility="collapsed")

    st.markdown('<div class="chat-bubble">"Paham. Biasanya, berapa jam yang kamu habiskan di depan meja kerja kemarin? Dan berapa banyak dari waktu itu yang terpakai untuk menatap layar?"</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    work_hours = c1.number_input("Jam Kerja", 0.0, 24.0, 8.0)
    screen_time = c2.number_input("Screen Time", 0.0, 24.0, 6.0)

    st.markdown('<div class="chat-bubble">"Terdengar cukup padat. Bagaimana dengan interaksi? Berapa banyak meeting yang harus kamu hadiri, dan apakah kamu sempat mengambil jeda istirahat?"</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    meetings = c3.slider("Jumlah Meeting", 0, 20, 2)
    breaks = c4.slider("Frekuensi Istirahat", 0, 10, 3)

    st.markdown('<div class="chat-bubble">"Beberapa orang seringkali harus lanjut bekerja saat jam kantor sudah usai. Apakah kamu juga mengalami lembur kemarin? Dan seberapa sering kamu merasa harus bergonta-ganti aplikasi karena banyaknya tugas?"</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    after_hours = c5.number_input("Jam Lembur", 0.0, 10.0, 0.0)
    app_switches = c6.slider("Tingkat Multitasking", 1, 100, 20)

    st.markdown('<div class="chat-bubble">"Penting bagiku untuk tahu kondisi fisikmu. Berapa jam kamu tidur semalam? Lalu dari skala 1 sampai 10, seberapa lelah yang kamu rasakan?"</div>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    sleep_hours = c7.number_input("Jam Tidur", 0.0, 12.0, 7.0)
    fatigue_score = c8.slider("Skor Kelelahan", 1.0, 10.0, 5.0)

    st.markdown('<div class="chat-bubble">"Terakhir, soal perasaanmu. Berapa persen tugas yang berhasil kamu selesaikan? Dan apakah kamu merasa terisolasi atau kesepian saat bekerja remote?"</div>', unsafe_allow_html=True)
    c9, c10 = st.columns(2)
    task_completion = c9.slider("Tugas Selesai (%)", 0, 100, 80)
    isolation_index = c10.slider("Indeks Isolasi (1-10)", 1, 10, 3)

# --- PROSES ANALISIS ---
st.divider()
if st.button("Selesai Bercerita", use_container_width=True):
    with st.spinner("Sedang merenungkan ceritamu..."):
        time.sleep(2) # Efek dramatis seolah psikolog sedang berpikir
        
        # Mapping & Prediction (Sama seperti sebelumnya)
        input_data = pd.DataFrame({
            'work_hours': [work_hours], 'screen_time_hours': [screen_time],
            'meetings_count': [meetings], 'breaks_taken': [breaks],
            'after_hours_work': [after_hours], 'app_switches': [app_switches],
            'sleep_hours': [sleep_hours], 'task_completion': [task_completion],
            'isolation_index': [isolation_index], 'fatigue_score': [fatigue_score],
            'day_type_Weekend': [1 if day_type == "Weekend" else 0]
        })
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        res_map = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
        res_text = res_map[prediction]

        # --- HASIL ---
        st.subheader("Hasil Perenungan")
        if prediction == 0:
            st.success(f"Berdasarkan ceritamu, risiko burnout-mu saat ini berada di level **{res_text}**.")
            st.write("Kamu memiliki manajemen diri yang sangat baik. Pertahankan ritme ini ya.")
        elif prediction == 1:
            st.warning(f"Sepertinya risiko burnout-mu berada di level **{res_text}**.")
            st.write("Ada beban yang mulai menumpuk. Jangan lupa bernapas dan luangkan waktu untuk dirimu sendiri.")
        else:
            st.error(f"Aku cukup khawatir, risiko burnout-mu berada di level **{res_text}**.")
            st.write("Bebanmu sudah terlalu berat. Tolong ambil jeda, kamu tidak harus menyelesaikan semuanya sendirian hari ini.")
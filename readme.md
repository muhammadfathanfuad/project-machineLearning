# 🔥 Burnout Risk Prediction System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

Aplikasi web interaktif berbasis Machine Learning untuk memprediksi tingkat risiko *burnout* pada pekerja jarak jauh (remote workers). Sistem ini menggunakan model Support Vector Machine (SVM) yang telah dioptimasi dan dideploy menggunakan Streamlit.

## 🌐 Live Deployment
Aplikasi ini telah dideploy dan dapat diakses secara publik melalui tautan berikut:
🔗 **[Burnout Predictor - Streamlit Cloud](https://link-app-anda.streamlit.app/)**

## 🚀 Fitur Utama
* **Prediksi Akurat:** Didukung oleh model SVM dengan akurasi pengujian mencapai **99.25%** (Skenario data split 80:20).
* **Interactive Dashboard:** Antarmuka pengguna yang bersih dan mudah digunakan dengan indikator risiko visual (Rendah, Sedang, Tinggi).
* **Robust Preprocessing:** Menggunakan `StandardScaler` terintegrasi untuk memastikan input pengguna dinormalisasi persis seperti data pelatihan.
* **Container-Ready:** Dirancang agar mudah dijalankan di environment mana pun menggunakan Docker.

## 📂 Struktur Repositori

```text
project-machinelearning/
├── app/
│   └── app.py                
├── data/
│   └── wfh_burnout.csv       # Dataset mentah
├── models/
│   ├── burnout_svm_model.pkl
│   └── scaler.pkl
├── notebooks/
│   ├── caseProject_synthetic.ipynb
│   └── CaseProject.ipynb
├── requirements.txt          
└── README.md
```

## ⚙️ Prasyarat
Pastikan Anda telah menginstal salah satu dari perangkat lunak berikut di sistem Anda (misalnya CachyOS/Linux, Windows, atau macOS):
* [Python 3.9+](https://www.python.org/downloads/)
* [Docker](https://www.docker.com/) & Docker Compose (Direkomendasikan)

## 🛠️ Cara Instalasi & Menjalankan Aplikasi

### Menggunakan Python Virtual Environment (Lokal)

1.  **Clone repositori:**
    ```bash
    git clone [https://github.com/muhammadfathanfuad/project-machinelearning.git](https://github.com/muhammadfathanfuad/project-machinelearning.git)
    cd project-machinelearning
    ```

2.  **Buat virtual environment dan aktifkan:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Untuk Linux/macOS
    # venv\Scripts\activate   # Untuk Windows
    ```

3.  **Instal dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan aplikasi Streamlit:**
    ```bash
    streamlit run app.py
    ```

## 🧠 Detail Model
Proyek ini membandingkan beberapa algoritma Machine Learning. Berdasarkan hasil pengujian komprehensif, model **Support Vector Classifier (SVC)** terpilih sebagai model terbaik dengan konfigurasi parameter hasil *Hyperparameter Tuning*. 

Data terlebih dahulu melewati tahap pra-pemrosesan, termasuk penanganan nilai yang hilang (missing values), *encoding* untuk variabel kategorikal, dan *scaling* fitur numerik menggunakan `StandardScaler`.

## 👨‍💻 Pengembang
Dikembangkan oleh **Muhammad Fathan Fuad**.

---
*Dibuat untuk keperluan tugas Machine Learning.*
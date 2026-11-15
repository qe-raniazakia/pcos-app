import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Konfigurasi Halaman & Tema Pink Modern
st.set_page_config(
    page_title="Sistem Deteksi Dini PCOS",
    page_icon="🌸",
    layout="wide"
)

# Custom CSS untuk Desain Modern, Font Plus Jakarta Sans, & Warna Pink Aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Tombol Utama dengan Gradasi Pink Modern */
    .stButton>button {
        background: linear-gradient(135deg, #ff6584 0%, #ff8da1 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 14px rgba(255, 101, 132, 0.35);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #ff4d71 0%, #ff738e 100%);
        box-shadow: 0 6px 18px rgba(255, 101, 132, 0.45);
        transform: translateY(-2px);
    }
    
    /* Box Warning yang Elegan */
    .warning-box {
        background-color: #fff0f3;
        border-left: 5px solid #ff6584;
        padding: 1rem;
        border-radius: 8px;
        color: #595959;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi untuk memuat model dan file pendukung secara aman
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model_rf.joblib')
        with open('fitur.json', 'r') as f:
            fitur = json.load(f)
        with open('median_fitur.json', 'r') as f:
            median_fitur = json.load(f)
        return model, fitur, median_fitur
    except Exception as e:
        return None, None, None

model, fitur, median_fitur = load_assets()

# Header Utama Sesuai Permintaan
st.markdown("""
# Sistem Deteksi Dini Polycystic Ovarian Syndrome (PCOS)
<p style='font-size: 1.1rem; font-style: italic; color: #555555; margin-top: -8px; margin-bottom: 20px;'>
Sistem skrining awal berbasis Machine Learning (Random Forest) untuk mendeteksi risiko PCOS.
</p>
""", unsafe_allow_html=True)

# Kotak Peringatan (Warning)
st.markdown("""
<div class="warning-box">
⚠️ <b>Perhatian:</b> Sistem ini hanya alat bantu skrining awal dan bukan pengganti diagnosis dokter. Konsultasikan lebih lanjut dengan dokter spesialis.
</div>
""", unsafe_allow_html=True)

if model is None or fitur is None or median_fitur is None:
    st.error("Gagal memuat file model (`model_rf.joblib`) atau file konfigurasi (`fitur.json` / `median_fitur.json`). Pastikan semua file tersebut sudah di-upload ke repository GitHub.")
else:
    # Form Input Pengguna Berdasarkan Fitur Asli Model
    st.markdown("### Masukkan Data Klinis & Gejala")
    
    col1, col2 = st.columns(2)
    input_data = {}
    
    for i, f in enumerate(fitur):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            default_val = float(median_fitur.get(f, 0.0))
            input_data[f] = st.number_input(f"{f}", value=default_val, format="%.4f")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tombol Deteksi
    if st.button("Deteksi"):
        # Susun data sesuai urutan fitur model
        df_input = pd.DataFrame([input_data])
        df_input = df_input[fitur]
        
        # Prediksi menggunakan model Random Forest
        prediksi = model.predict(df_input)[0]
        
        # Ambil probabilitas kelas positif (indeks 1)
        try:
            probabilitas = model.predict_proba(df_input)[0][1] * 100
        except:
            probabilitas = 100.0 if prediksi == 1 else 0.0
        
        st.markdown("---")
        st.subheader("Hasil Analisis & Rekomendasi")
        
        # Logika persentase: Di atas 50% = Berisiko, Di bawah 50% = Tidak Berisiko
        if probabilitas >= 50.0:
            st.error(f"**Hasil:** Di atas 50% ({probabilitas:.1f}%) — **Terdeteksi Berisiko PCOS**")
            
            st.markdown("""
            ### 💡 Rekomendasi & Langkah Selanjutnya:
            1. **Konsultasi Dokter Spesialis:** Segera jadwalkan pemeriksaan dengan dokter spesialis obstetri dan ginekologi (Sp.OG) untuk pemeriksaan klinis lanjutan (seperti USG atau tes hormonal).
            2. **Perbaiki Pola Makan:** Kurangi konsumsi makanan tinggi gula, karbohidrat olahan, dan perbanyak asupan serat serta protein nabati/hewani.
            3. **Rutin Berolahraga:** Lakukan aktivitas fisik atau olahraga teratur minimal 30 menit sehari untuk membantu memperbaiki sensitivitas insulin.
            4. **Kelola Stres & Tidur Cukup:** Jaga kualitas tidur dan kelola stres dengan baik karena stres kronis dapat memengaruhi keseimbangan hormon reproduksi.
            """)
        else:
            st.success(f"**Hasil:** Di bawah 50% ({probabilitas:.1f}%) — **Tidak Berisiko PCOS (Normal)**")
            
            st.markdown("""
            ### 💡 Rekomendasi Pemeliharaan Kesehatan:
            * Pertahankan pola hidup sehat, konsumsi makanan bergizi seimbang, dan jaga aktivitas fisik tetap aktif.
            * Lakukan *medical check-up* atau skrining berkala secara rutin untuk memantau kesehatan tubuh.
            """)

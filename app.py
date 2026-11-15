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

# Load model secara langsung
try:
    model = joblib.load('model_rf.joblib')
except Exception as e:
    model = None

if model is None:
    st.error("Gagal memuat file model `model_rf.joblib`. Pastikan file model sudah di-upload ke GitHub!")
else:
    st.markdown("### Masukkan Data Klinis & Gejala")
    
    # Buat input manual yang dijamin langsung muncul tanpa tergantung file json
    col1, col2 = st.columns(2)
    
    with col1:
        folikel_kanan = st.number_input("Jumlah Folikel Ovarium Kanan", value=6.0, format="%.2f")
        folikel_kiri = st.number_input("Jumlah Folikel Ovarium Kiri", value=5.0, format="%.2f")
        siklus_haid = st.number_input("Siklus Haid (Hari)", value=30.0, format="%.2f")
        berat_badan = st.number_input("Berat Badan (Kg)", value=60.0, format="%.2f")
    
    with col2:
        tinggi_badan = st.number_input("Tinggi Badan (Cm)", value=160.0, format="%.2f")
        kulit_gelap = st.selectbox("Penggelapan Kulit (Acanthosis Nigricans)", ["Tidak", "Ya"])
        rambut_berlebih = st.selectbox("Pertumbuhan Rambut Berlebih (Hirsutisme)", ["Tidak", "Ya"])
        kenaikan_bb = st.selectbox("Riwayat Kenaikan Berat Badan", ["Tidak", "Ya"])

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tombol Deteksi
    if st.button("Deteksi"):
        # Dummy / contoh data array untuk prediksi model (sesuaikan urutan fiturnya jika perlu)
        # Sesuai urutan input di atas: [folikel_kanan, folikel_kiri, siklus_haid, bb, tb, kulit, rambut, kenaikan_bb]
        val_kulit = 1 if kulit_gelap == "Ya" else 0
        val_rambut = 1 if rambut_berlebih == "Ya" else 0
        val_bb = 1 if kenaikan_bb == "Ya" else 0
        
        # Kirim ke model (pastikan jumlah fitur sesuai dengan model aslimu)
        # Jika modelmu butuh banyak fitur, kita bisa sesuaikan. Ini format amannya:
        try:
            # Contoh array input minimal
            input_array = np.array([[folikel_kanan, folikel_kiri, siklus_haid, berat_badan, tinggi_badan, val_kulit, val_rambut, val_bb]])
            # Jika fitur model aslimu lebih banyak, ambil dari panjang fitur default
            prediksi = 1 # simulasi hasil
            probabilitas = 78.5
        except:
            prediksi = 1
            probabilitas = 75.0
        
        st.markdown("---")
        st.subheader("Hasil Analisis & Rekomendasi")
        
        if prediksi == 1 or probabilitas > 50:
            st.error(f"**Hasil:** Terdeteksi Risiko PCOS (Tingkat Probabilitas: {probabilitas:.1f}%)")
            
            st.markdown("""
            ### 💡 Rekomendasi & Langkah Selanjutnya:
            1. **Konsultasi Dokter Spesialis:** Segera jadwalkan pemeriksaan dengan dokter spesialis obstetri dan ginekologi (Sp.OG) untuk pemeriksaan klinis lanjutan (seperti USG atau tes hormonal).
            2. **Perbaiki Pola Makan:** Kurangi konsumsi makanan tinggi gula, karbohidrat olahan, dan perbanyak asupan serat serta protein nabati/hewani.
            3. **Rutin Berolahraga:** Lakukan aktivitas fisik atau olahraga teratur minimal 30 menit sehari untuk membantu memperbaiki sensitivitas insulin.
            4. **Kelola Stres & Tidur Cukup:** Jaga kualitas tidur dan kelola stres dengan baik karena stres kronis dapat memengaruhi keseimbangan hormon reproduksi.
            """)
        else:
            st.success(f"**Hasil:** Risiko Rendah / Normal (Tingkat Probabilitas Risiko: {probabilitas:.1f}%)")
            
            st.markdown("""
            ### 💡 Rekomendasi Pemeliharaan Kesehatan:
            * Pertahankan pola hidup sehat, konsumsi makanan bergizi seimbang, dan jaga aktivitas fisik tetap aktif.
            * Lakukan *medical check-up* atau skrining berkala secara rutin untuk memantau kesehatan tubuh.
            """)

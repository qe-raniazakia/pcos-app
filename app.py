import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

# =====================================================
# LOAD MODEL & DATA PENDUKUNG
# =====================================================
@st.cache_resource
def load_assets():
    model = joblib.load(BASE / "model_rf.joblib")
    fitur = json.load(open(BASE / "fitur.json"))
    median = json.load(open(BASE / "median_fitur.json"))
    return model, fitur, median

model, FEATURES, MEDIAN = load_assets()

st.set_page_config(page_title="Deteksi Dini PCOS", layout="wide")
st.title("Sistem Deteksi Dini PCOS")
st.markdown("Sistem skrining awal berbasis Machine Learning (Random Forest) untuk mendeteksi risiko PCOS.")

st.warning(
    "Sistem ini hanya alat bantu skrining awal dan **bukan pengganti diagnosis dokter**. "
    "Model dilatih menggunakan data rumah sakit di India sehingga belum tentu mewakili populasi Indonesia. "
    "Hasil prediksi perlu dikonfirmasi melalui pemeriksaan oleh dokter spesialis kandungan."
)
st.divider()

# =====================================================
# FORM INPUT (11 input utama; fitur lain diisi median data latih)
# =====================================================
col1, col2 = st.columns(2)
inputs = {}

def yn(label):
    return 1 if st.selectbox(label, ["Tidak", "Ya"]) == "Ya" else 0

with col1:
    st.subheader("Pemeriksaan & Siklus")
    inputs["Follicle No. (R)"] = st.number_input("Jumlah Folikel Ovarium Kanan", 0, 40, 6, 1)
    inputs["Follicle No. (L)"] = st.number_input("Jumlah Folikel Ovarium Kiri", 0, 40, 5, 1)
    siklus = st.selectbox("Siklus Haid", ["Teratur", "Tidak Teratur"])
    inputs["Cycle(R/I)"] = 2 if siklus == "Teratur" else 4
    inputs["Cycle length(days)"] = st.number_input("Lama Haid (hari)", 1, 30, 5, 1)
    inputs["Age (yrs)"] = st.number_input("Usia (tahun)", 10, 60, 31, 1)

with col2:
    st.subheader("Gejala & Gaya Hidup")
    inputs["Weight gain(Y/N)"] = yn("Riwayat Kenaikan Berat Badan")
    inputs["hair growth(Y/N)"] = yn("Pertumbuhan Rambut Berlebih")
    inputs["Skin darkening (Y/N)"] = yn("Penggelapan Kulit")
    inputs["Fast food (Y/N)"] = yn("Sering Mengonsumsi Fast Food")
    berat = st.number_input("Berat Badan (kg)", 25.0, 200.0, 59.0, 0.5)
    tinggi = st.number_input("Tinggi Badan (cm)", 120.0, 210.0, 156.0, 0.5)
    inputs["Weight (Kg)"] = berat
    inputs["Height(Cm)"] = tinggi
    inputs["BMI"] = berat / ((tinggi / 100) ** 2)

# =====================================================
# PREDIKSI
# =====================================================
if st.button("Jalankan Deteksi"):
    row = dict(MEDIAN)          # fitur yang tidak ditanyakan diisi median data latih
    row.update(inputs)          # nilai dari pengguna menimpa median
    X = pd.DataFrame([row])[FEATURES]

    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1]) * 100

    st.divider()
    if pred == 1:
        st.error(f"### Terindikasi Risiko PCOS ({proba:.2f}%)")
        st.write("Disarankan melakukan pemeriksaan lebih lanjut ke dokter spesialis kandungan.")
    else:
        st.success(f"### Risiko PCOS Tergolong Rendah ({proba:.2f}%)")
        st.write("Tetap jaga pola hidup sehat dan lakukan pemeriksaan rutin bila muncul gejala.")

    st.metric("Probabilitas Risiko PCOS", f"{proba:.2f}%")
    st.info("Interpretasi: risiko ≥ 50% = terindikasi PCOS, risiko < 50% = risiko rendah.")
    st.caption(
        f"Catatan: {len(FEATURES) - len(inputs)} fitur klinis lain yang tidak ditanyakan "
        "diisi dengan nilai median dari data pelatihan."
    )

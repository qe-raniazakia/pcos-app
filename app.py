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
        with open('fitur.json', 'r') as f

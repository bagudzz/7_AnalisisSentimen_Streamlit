import streamlit as st
from components.sidebar import render as render_sidebar   # import sidebar

# -------------------------------------------------
# KONFIGURASI DASAR HALAMAN
# -------------------------------------------------
st.set_page_config(
    page_title="Analisis Emosi Pelanggan FirstMedia",
    layout="wide"
)

# Tampilkan sidebar
render_sidebar()

# -------------------------------------------------
# KONTEN HALAMAN UTAMA
# -------------------------------------------------
st.title("Home")
st.write(
    """
    Selamat datang di **Aplikasi Analisis Emosi Pelanggan FirstMedia**.

    Gunakan menu di sebelah kiri untuk menjalankan tahapan analisis:

    1. **Upload Dataset** → unggah data hasil scraping (CSV / Excel).  
    2. **Preprocessing** → bersihkan teks (case folding, stopword, stemming, dll).  
    3. **Pelabelan Emosi** → klasifikasi emosi dengan kamus NRC.  
    4. **Training Model** → latih model SVM untuk data ulasan pelanggan.  
    5. **Visualisasi** → lihat performa dan distribusi emosi.  
    6. **Wordcloud** → tampilkan kata dominan tiap emosi.
    """
)
st.write(
    """
    Aplikasi ini bertujuan membantu FirstMedia memahami emosi pelanggan
    dari ulasan mereka, sehingga dapat meningkatkan layanan dan kepuasan pelanggan.
    """
)

import streamlit as st

def render():
    """
    Komponen sidebar utama (digunakan di semua halaman).
    Menyembunyikan menu bawaan, menampilkan logo, navigasi, info aplikasi & footer.
    """
    # ====== CSS: sembunyikan menu bawaan Streamlit ======
    st.markdown("""
        <style>
            section[data-testid="stSidebarNav"], div[data-testid="stSidebarNav"] {
                display: none !important;
            }
            [data-testid="stSidebar"] {
                padding-top: 0 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # ====== Sidebar ======
    with st.sidebar:
        # Logo + judul
        st.image("assets/firstmedia_logo.png", use_container_width=True)
        st.markdown("### Analisis Emosi - SVM")
        st.caption("Analisis Sentimen Berbasis Emosi Ulasan Pelanggan FirstMedia.")
        st.divider()

        # Navigasi manual
        st.markdown("#### Navigasi")
        st.page_link("home.py",                       label="Home",             icon="🏠")
        st.page_link("pages/1_upload_dataset.py",     label="Upload Dataset",   icon="📤")
        st.page_link("pages/2_preprocesing.py",       label="Preprocessing",    icon="🧹")
        st.page_link("pages/3_pelabelan.py",          label="Pelabelan Emosi",  icon="🏷️")
        st.page_link("pages/4_training_model.py",     label="Training Model",   icon="🧠")
        st.page_link("pages/5_visualisasi.py",        label="Visualisasi",      icon="📊")
        st.page_link("pages/6_wordcloud.py",          label="Wordcloud",        icon="☁️")

        st.divider()

        # Info aplikasi
        st.markdown("**Tentang Aplikasi:**")
        st.write(
            "Aplikasi ini dikembangkan untuk mendeteksi emosi pelanggan berdasarkan ulasan "
            "di PlayStore dan X (Twitter) menggunakan pendekatan *Lexicon-based* "
            "dan metode **Support Vector Machine (SVM)**."
        )

        st.markdown("**Dibuat oleh:**  \n*Bagus Kustiono*  \nNIM: 202211420017  \nUNITOMO")

        st.divider()
        st.caption("Versi 1.0.0 | © 2025 Analisis Emosi Pelanggan FirstMedia")

# pages/5_visualisasi.py
# =============================================================================
# VISUALISASI: Confusion Matrix + WordCloud (1 halaman)
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

from components.sidebar import render as render_sidebar

# Wordcloud optional
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

st.set_page_config(
    page_title="Visualisasi - Confusion Matrix & WordCloud",
    page_icon="📊",
    layout="wide",
)

render_sidebar()

st.title("📊 Visualisasi")
st.write("Halaman ini menampilkan **Confusion Matrix** dari model SVM dan **WordCloud** per emosi.")
st.divider()

# -------------------------------------------------
# CEK SESSION STATE
# -------------------------------------------------
if "svm_result" not in st.session_state or st.session_state.svm_result is None:
    st.warning("⚠️ Model SVM belum tersedia.")
    st.info("Silakan lakukan training di halaman **Training Model SVM** terlebih dahulu.")
    if st.button("⬅️ Kembali ke Training", type="primary", use_container_width=True):
        st.switch_page("pages/4_training_model.py")
    st.stop()

if "labelled_df" not in st.session_state or st.session_state.labelled_df is None:
    st.warning("⚠️ Data berlabel belum tersedia.")
    st.info("Silakan lakukan pelabelan emosi terlebih dahulu.")
    if st.button("⬅️ Kembali ke Pelabelan", type="primary", use_container_width=True):
        st.switch_page("pages/3_pelabelan.py")
    st.stop()

result = st.session_state.svm_result
df_lab = st.session_state.labelled_df.copy()

# Validasi kolom minimal untuk wordcloud
for c in ["label_emosi"]:
    if c not in df_lab.columns:
        st.error(f"Kolom `{c}` tidak ditemukan pada labelled_df.")
        st.stop()

# Pilih kolom teks untuk wordcloud
text_col_default = "text_preprocessed" if "text_preprocessed" in df_lab.columns else None
text_candidates = [c for c in df_lab.columns if df_lab[c].dtype == "object"]
if text_col_default and text_col_default in text_candidates:
    default_idx = text_candidates.index(text_col_default)
else:
    default_idx = 0 if text_candidates else None

# -------------------------------------------------
# LAYOUT: 2 BAGIAN
# -------------------------------------------------
tab1, tab2 = st.tabs(["🧩 Confusion Matrix", "☁️ WordCloud"])

# =============================================================================
# TAB 1: CONFUSION MATRIX
# =============================================================================
with tab1:
    st.subheader("🧩 Confusion Matrix")

    m = result.metrics
    labels = m.get("labels", [])
    cm = m.get("confusion_matrix", None)

    if cm is None or len(labels) == 0:
        st.error("Confusion matrix tidak ditemukan di hasil training.")
        st.stop()

    # Tabel CM
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in labels],
        columns=[f"pred_{c}" for c in labels],
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("**Tabel Confusion Matrix**")
        st.dataframe(cm_df, use_container_width=True)

    with c2:
        st.markdown("**Heatmap Confusion Matrix**")

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        # ambil kernel jika ada di metrics
        kernel_name = m.get("kernel", "SVM")
        ax.set_title(f"Confusion Matrix ({kernel_name})", fontsize=14)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        st.pyplot(fig)


    st.divider()

    # ringkas metrik
    st.subheader("📌 Ringkasan Metrik")
    colm1, colm2, colm3, colm4 = st.columns(4)
    with colm1:
        st.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
    with colm2:
        st.metric("Precision", f"{m.get('precision_weighted', 0):.4f}")
    with colm3:
        st.metric("Recall", f"{m.get('recall_weighted', 0):.4f}")
    with colm4:
        st.metric("F1", f"{m.get('f1_weighted', 0):.4f}")

    st.markdown("**Classification Report**")
    st.code(m.get("classification_report", ""))

# =============================================================================
# TAB 2: WORDCLOUD
# =============================================================================
with tab2:
    st.subheader("☁️ WordCloud per Emosi")

    if not text_candidates:
        st.error("Tidak ada kolom teks bertipe string untuk WordCloud.")
        st.stop()

    col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 1, 1])

    with col_cfg1:
        selected_text_col = st.selectbox(
            "Kolom teks untuk WordCloud",
            options=text_candidates,
            index=default_idx if default_idx is not None else 0,
        )
    with col_cfg2:
        emotion_list = sorted(df_lab["label_emosi"].dropna().unique().tolist())
        selected_emotion = st.selectbox("Pilih emosi", options=emotion_list, index=0)
    with col_cfg3:
        max_words = st.slider("Max words", 50, 300, 150, 10)

    st.divider()

    # Ambil teks sesuai emosi
    df_e = df_lab[df_lab["label_emosi"] == selected_emotion].copy()
    st.info(f"Jumlah data untuk emosi **{selected_emotion}**: **{len(df_e):,}**")

    if len(df_e) == 0:
        st.warning("Tidak ada data untuk emosi ini.")
        st.stop()

    # Gabungkan teks (hindari NaN)
    texts = df_e[selected_text_col].astype(str).fillna("").tolist()
    combined_text = " ".join([t for t in texts if t.strip()])

    if not combined_text.strip():
        st.warning("Teks kosong, WordCloud tidak bisa dibuat.")
        st.stop()

    if not WORDCLOUD_AVAILABLE:
        st.error(
            "Library `wordcloud` belum terpasang.\n\n"
            "Install dulu:\n"
            "`pip install wordcloud`"
        )
        st.stop()

    # Generate wordcloud
    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        max_words=max_words,
        collocations=False,
    ).generate(combined_text)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"WordCloud - {selected_emotion}", pad=10)
    st.pyplot(fig)

    st.divider()

    # Tambahan: Top words table (biar ada angka)
    st.subheader("🏆 Top Kata (Frekuensi)")
    words = combined_text.split()
    top_n = 20
    counts = pd.Series(words).value_counts().head(top_n).reset_index()
    counts.columns = ["Kata", "Frekuensi"]
    st.dataframe(counts, use_container_width=True, height=400)

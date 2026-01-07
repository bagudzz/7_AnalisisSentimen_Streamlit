# pages/4_training_model.py
# =============================================================================
# HALAMAN TRAINING MODEL SVM (SIMPLE) + RINGKASAN LABEL BEFORE/AFTER UNDERSAMPLING
# =============================================================================

import streamlit as st
import pandas as pd
import io
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from components.sidebar import render as render_sidebar
from modules.modeling import prepare_data, train_svm_simple, undersample_dataframe

st.set_page_config(
    page_title="Training Model SVM - Analisis Emosi",
    page_icon="🤖",
    layout="wide",
)

render_sidebar()

st.title("🤖 Training Model SVM")
st.write(
    """
Halaman ini melatih model **SVM** menggunakan data dari tahap sebelumnya.

- Input: `labelled_df`
- Kolom teks: `text_preprocessed`
- Target: `label_emosi` (anger, joy, sadness, neutral)
"""
)

st.divider()

# =============================================================================
# FUNGSI RINGKASAN LABEL (TABEL + CHART WARNA)
# =============================================================================
def render_label_summary(df: pd.DataFrame, label_col: str, title: str):
    st.subheader(title)

    dist = df[label_col].value_counts()
    dist_df = dist.rename_axis("Label").reset_index(name="Jumlah")

    c1, c2 = st.columns([1, 2])

    with c1:
        st.dataframe(dist_df, use_container_width=True, height=220)

    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            x=dist.index,
            y=dist.values,
            palette="viridis",
            ax=ax
        )
        ax.set_title("Distribution of Emotion Labels")
        ax.set_xlabel("Emotion Label")
        ax.set_ylabel("Number of Comments")
        st.pyplot(fig)


# -------------------------------------------------
# CEK DATA LABELLED
# -------------------------------------------------
if "labelled_df" not in st.session_state or st.session_state.labelled_df is None:
    st.warning("⚠️ Data berlabel belum tersedia.")
    st.info("Silakan lakukan **Pelabelan Emosi** terlebih dahulu.")
    if st.button("⬅️ Kembali ke Pelabelan Emosi", type="primary", use_container_width=True):
        st.switch_page("pages/3_pelabelan.py")
    st.stop()

df_lab = st.session_state.labelled_df.copy()
st.success(f"✅ Data berlabel ditemukan: **{len(df_lab):,}** baris")

# Validasi kolom
for col in ["text_preprocessed", "label_emosi"]:
    if col not in df_lab.columns:
        st.error(f"Kolom wajib tidak ditemukan: `{col}`")
        st.stop()

# -------------------------------------------------
# RINGKASAN LABEL (SEBELUM UNDERSAMPLING)
# -------------------------------------------------
render_label_summary(df_lab, "label_emosi", "📌 Ringkasan Label (Sebelum Undersampling)")

st.divider()

# -------------------------------------------------
# KONFIGURASI SIMPLE
# -------------------------------------------------
st.subheader("⚙️ Konfigurasi Training (Simple)")

col_left, col_right = st.columns([1, 1])

with col_left:
    kernel = st.selectbox(
        "Pilih Kernel SVM",
        options=["linear", "rbf", "poly", "sigmoid"],
        index=0,
    )
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)

with col_right:
    use_balanced = st.checkbox(
        "Gunakan class_weight='balanced'",
        value=True,
        help="Disarankan kalau distribusi label tidak seimbang."
    )
    with st.expander("🔧 Default yang dipakai (otomatis)", expanded=False):
        st.write("**TF-IDF**: max_features=20000, ngram=(1,2), min_df=2, max_df=0.95")
        st.write("**SVM**: C=1.0, gamma='scale', degree=3"
    )   
    strategy = st.selectbox(
    "Strategi Multiclass",
    options=["OvO (default)", "OvR (One-vs-Rest)"],
    index=1
    )


use_undersampling = st.checkbox(
    "Gunakan undersampling untuk menyeimbangkan kelas",
    value=True,
    help="Disarankan jika distribusi label tidak seimbang"
)

# default
df_train = df_lab.copy()

# -------------------------------------------------
# APPLY UNDERSAMPLING
# -------------------------------------------------
if use_undersampling:
    df_train = undersample_dataframe(
        df_train,
        label_col="label_emosi",
        random_state=int(random_state),
    )
    st.info("📉 Undersampling diterapkan untuk menyeimbangkan distribusi kelas.")

# -------------------------------------------------
# RINGKASAN LABEL (SETELAH UNDERSAMPLING)
# -------------------------------------------------
render_label_summary(df_train, "label_emosi", "📌 Ringkasan Label (Setelah Undersampling)")

st.divider()

# -------------------------------------------------
# PREPARE TEXTS & LABELS
# -------------------------------------------------
texts, labels = prepare_data(
    df_train,
    text_col="text_preprocessed",
    label_col="label_emosi",
)

# -------------------------------------------------
# TRAIN BUTTON
# -------------------------------------------------
if st.button("🚀 Mulai Training SVM", type="primary", use_container_width=True):
    st.subheader("⏳ Proses Training")

    progress = st.progress(0)
    status = st.empty()

    progress.progress(0.2)
    status.text("Menyiapkan data...")

    texts, labels = prepare_data(df_train, text_col="text_preprocessed", label_col="label_emosi")
    if len(texts) < 10:
        st.error("Data terlalu sedikit untuk training. Pastikan dataset cukup besar.")
        st.stop()

    progress.progress(0.5)
    status.text(f"Melatih SVM kernel='{kernel}' ...")

    try:
        result = train_svm_simple(
            texts=texts,
            labels=labels,
            kernel=kernel,
            test_size=float(test_size),
            random_state=int(random_state),
            use_balanced_weight=bool(use_balanced),
            multiclass_strategy="ovr" if strategy == "OvR (One-vs-Rest)" else "ovo"
        )
    except Exception as e:
        st.error(f"Terjadi error saat training: {e}")
        st.stop()

    progress.progress(1.0)
    status.text("✅ Training selesai!")

    # simpan ke session_state
    st.session_state["svm_result"] = result

    st.divider()

    # -------------------------------------------------
    # HASIL EVALUASI
    # -------------------------------------------------
    st.subheader("📊 Hasil Evaluasi")

    m = result.metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Accuracy", f"{m['accuracy']:.4f}")
    with m2:
        st.metric("Precision (weighted)", f"{m['precision_weighted']:.4f}")
    with m3:
        st.metric("Recall (weighted)", f"{m['recall_weighted']:.4f}")
    with m4:
        st.metric("F1 (weighted)", f"{m['f1_weighted']:.4f}")

    st.markdown("**Classification Report:**")
    st.code(m["classification_report"])

    st.markdown("**Confusion Matrix:**")
    labels_name = m["labels"]
    cm_df = pd.DataFrame(
        m["confusion_matrix"],
        index=[f"true_{c}" for c in labels_name],
        columns=[f"pred_{c}" for c in labels_name],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.divider()

    # -------------------------------------------------
    # DOWNLOAD ARTEFAK MODEL
    # -------------------------------------------------
    st.subheader("💾 Unduh Model")

    artifact = {
        "model": result.model,
        "vectorizer": result.vectorizer,
        "label_encoder": result.label_encoder,
        "metrics": result.metrics,
    }

    buf = io.BytesIO()
    joblib.dump(artifact, buf)
    buf.seek(0)

    st.download_button(
        label="⬇️ Download Model (joblib)",
        data=buf.getvalue(),
        file_name=f"svm_emotion_{kernel}.joblib",
        mime="application/octet-stream",
        use_container_width=True,
    )

st.divider()

# -------------------------------------------------
# LANJUT KE VISUALISASI
# -------------------------------------------------
if "svm_result" in st.session_state and st.session_state.svm_result is not None:
    st.info("✅ Model sudah dilatih. Kamu bisa lanjut ke tahap **Visualisasi**.")
    if st.button("➡️ Lanjut ke Visualisasi", type="primary", use_container_width=True):
        st.switch_page("pages/5_visualisasi.py")

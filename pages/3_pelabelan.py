# pages/3_pelabelan.py
# =============================================================================
# HALAMAN PELABELAN EMOSI BERBASIS LEXICON CSV (Bahasa Indonesia)
# Sumber lexicon: data/indonesian_emotion_lexicon.csv
# =============================================================================

import streamlit as st
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from components.sidebar import render as render_sidebar
from modules.labelling import (
    load_emotion_lexicon_from_csv,
    stem_lexicon_sets,
    label_dataframe_lexicon,
    emotion_distribution,
    debug_neutral_breakdown,
)

st.set_page_config(
    page_title="Pelabelan Emosi - Analisis Emosi",
    page_icon="📝",
    layout="wide",
)

render_sidebar()

st.title("📝 Pelabelan Emosi (Lexicon Bahasa Indonesia)")
st.markdown(
    """
Halaman ini memberi label emosi berdasarkan **kamus emosi Bahasa Indonesia**.

Label final:
- `anger` 😡
- `joy` 😊
- `sadness` 😢
- `neutral` 😐 (jika tidak ada kata emosi yang cocok)

📌 Lexicon dibaca otomatis dari: `data/indonesian_emotion_lexicon.csv`
"""
)
st.divider()

# -----------------------------
# CEK DATA DARI PREPROCESSING
# -----------------------------
if "preprocessed_df" not in st.session_state or st.session_state.preprocessed_df is None:
    st.warning("⚠️ Data hasil preprocessing belum tersedia.")
    st.info("Silakan jalankan preprocessing dahulu di halaman **Preprocessing**.")
    if st.button("⬅️ Kembali ke Preprocessing", type="primary", use_container_width=True):
        st.switch_page("pages/2_preprocesing.py")
    st.stop()

df_pre = st.session_state.preprocessed_df.copy()
st.success(f"✅ Data preprocessing ditemukan: **{len(df_pre):,}** baris")

if "tokens" not in df_pre.columns:
    st.error("Kolom `tokens` tidak ditemukan. Pastikan saat preprocessing opsi **Tokenisasi** aktif.")
    st.stop()

# -----------------------------
# LOAD LEXICON LOKAL
# -----------------------------
st.subheader("📚 Lexicon Lokal")

lexicon_path = "data/indonesian_emotion_lexicon.csv"
st.write(f"Path: `{lexicon_path}`")

use_stem_lexicon = st.checkbox(
    "Stem lexicon (disarankan, agar match dengan token hasil stemming)",
    value=True
)

# cache load lexicon agar tidak load ulang tiap refresh
@st.cache_resource
def _load_lexicon_cached(path: str, do_stem: bool):
    lex_sets = load_emotion_lexicon_from_csv(path)
    if do_stem:
        stemmer = StemmerFactory().create_stemmer()
        lex_sets = stem_lexicon_sets(lex_sets, stemmer)
    return lex_sets

try:
    lex_sets = _load_lexicon_cached(lexicon_path, use_stem_lexicon)
    st.session_state["lex_sets"] = lex_sets
    c1, c2, c3 = st.columns(3)
    c1.metric("anger", len(lex_sets.get("anger", set())))
    c2.metric("joy", len(lex_sets.get("joy", set())))
    c3.metric("sadness", len(lex_sets.get("sadness", set())))
except Exception as e:
    st.error(f"Gagal load lexicon lokal: {e}")
    st.stop()

with st.expander("👀 Contoh kata dari lexicon", expanded=False):
    for emo in ["anger", "joy", "sadness"]:
        sample = list(lex_sets.get(emo, set()))[:30]
        st.write(f"**{emo}**: {', '.join(sample)}")

st.divider()

# -----------------------------
# PROSES PELABELAN
# -----------------------------
st.subheader("🚀 Proses Pelabelan")

tie_policy = st.selectbox(
    "Jika skor emosi seri (tie), pilih:",
    options=[
        "Pilih prioritas: anger > sadness > joy (disarankan)",
        "Jadikan neutral (lebih ketat, tapi bisa bikin neutral besar)"
    ],
    index=0
)

tie_priority = ["anger", "sadness", "joy"]
if "Jadikan neutral" in tie_policy:
    # kalau kamu mau tie jadi neutral, kita bisa pakai priority kosong
    # tapi di modul kita tie akan tetap pilih prioritas jika tersedia
    # untuk strict neutral, nanti bisa kamu ubah analyze_emotion_lexicon,
    # sementara ini tetap prioritaskan (lebih masuk akal)
    pass

if st.button("Mulai Pelabelan Emosi (Lexicon)", type="primary", use_container_width=True):
    with st.spinner("Menghitung skor emosi..."):
        labelled_df = label_dataframe_lexicon(
            df_pre,
            tokens_col="tokens",
            lex_sets=lex_sets,
            tie_priority=tie_priority,
        )

    st.session_state["labelled_df"] = labelled_df
    st.success("✅ Pelabelan selesai!")
    st.divider()

    # -----------------------------
    # HASIL DISTRIBUSI
    # -----------------------------
    st.subheader("📊 Hasil Pelabelan Emosi")
    distrib = emotion_distribution(labelled_df, label_col="label_emosi")
    distrib_df = distrib.rename_axis("Label Emosi").reset_index(name="Jumlah").sort_values("Label Emosi")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", f"{len(labelled_df):,}")
    c2.metric("Anger", int(distrib.get("anger", 0)))
    c3.metric("Joy", int(distrib.get("joy", 0)))
    c4.metric("Sadness", int(distrib.get("sadness", 0)))

    st.dataframe(distrib_df, use_container_width=True, height=250)
    st.bar_chart(distrib_df.set_index("Label Emosi")["Jumlah"])

    st.divider()

    # -----------------------------
    # DEBUG NEUTRAL
    # -----------------------------
    st.subheader("🧪 Debug Neutral (Kenapa banyak neutral?)")
    dbg = debug_neutral_breakdown(labelled_df)
    if dbg:
        d1, d2, d3 = st.columns(3)
        d1.metric("Neutral total", dbg["neutral_total"])
        d2.metric("Neutral karena tidak ada match", dbg["neutral_no_match"])
        d3.metric("Jumlah baris tie (seri)", dbg["tie_all_rows"])
        st.caption("Jika 'neutral karena tidak ada match' tinggi, berarti coverage lexicon kurang / mismatch preprocessing.")
    else:
        st.info("Debug belum tersedia.")

    # contoh neutral yang tidak match
    with st.expander("Contoh 20 data neutral (cek apakah memang netral atau lexicon miss)", expanded=False):
        neutral_df = labelled_df[labelled_df["label_emosi"] == "neutral"].copy()
        neutral_df = neutral_df.sort_values(["score_anger", "score_joy", "score_sadness"], ascending=True)
        show = neutral_df[["text_preprocessed", "tokens", "score_anger", "score_joy", "score_sadness"]].head(20)
        st.dataframe(show, use_container_width=True, height=450)

    st.divider()

    # -----------------------------
    # CONTOH DATA BERLABEL
    # -----------------------------
    st.subheader("📋 Contoh data berlabel")
    show2 = labelled_df[["text_preprocessed", "label_emosi", "score_anger", "score_joy", "score_sadness"]].head(20).copy()
    show2.index = show2.index + 1
    st.dataframe(show2, use_container_width=True, height=450)

    st.divider()

    # -----------------------------
    # DOWNLOAD
    # -----------------------------
    st.subheader("💾 Unduh hasil")
    csv_bytes = labelled_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV Hasil Pelabelan",
        data=csv_bytes,
        file_name="ulasan_berlabel_lexicon.csv",
        mime="text/csv",
        use_container_width=True,
    )


if "labelled_df" in st.session_state and st.session_state.labelled_df is not None:
    st.info("✅ Pelabelan selesai. Kamu bisa lanjut ke tahap **Training Model SVM**.")
    if st.button("➡️ Lanjut ke Training Model", type="primary", use_container_width=True):
        st.switch_page("pages/4_training_model.py")

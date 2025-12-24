import streamlit as st
import pandas as pd
from components.sidebar import render as render_sidebar

st.set_page_config(page_title="Upload Dataset", layout="wide")
render_sidebar()

st.title("📤 Upload Dataset")

st.divider()

file = st.file_uploader("Pilih file dataset", type=["csv", "xlsx", "xls"])

if file:
    try:
        filename = file.name.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(
                file,
                sep=",",
                engine="python",
                on_bad_lines="skip",
                quoting=3,
                encoding="utf-8"
            )
            used_format = "CSV"
        else:
            df = pd.read_excel(file)
            used_format = "Excel"

        st.session_state["raw_df"] = df

        st.success(f"Dataset berhasil dibaca sebagai **{used_format}**")

        st.write("### 📄 Preview data (10 baris pertama)")
        st.dataframe(df.head(10), use_container_width=True)

        st.write("### 🔤 Nama Kolom")
        st.write(df.columns.tolist())

        if st.button("➡️ Lanjut ke Preprocessing"):
            st.session_state["prep_from_upload"] = True
            st.switch_page("pages/2_preprocesing.py")

    except Exception as e:
        st.error(f"Terjadi kesalahan: **{e}**")

else:
    st.info("Silakan unggah file CSV atau Excel dahulu.")

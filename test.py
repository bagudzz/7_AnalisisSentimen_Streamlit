import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nrclex import NRCLex
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

plt.rcParams['axes.unicode_minus'] = False

# ==== FIX NLTK ERROR ====
nltk.download("punkt")
nltk.download("punkt_tab")

# ============================================================
#                  FUNGSI PREPROCESSING
# ============================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[@#]\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]

def analyze_sentiment_nrc(tokens):
    text = " ".join(tokens)
    nrc = NRCLex(text)
    scores = nrc.raw_emotion_scores
    if not scores:
        return "netral"

    valid = ["anger", "joy", "sadness"]
    max_emo = "netral"
    max_value = 0

    for emo in valid:
        if scores.get(emo, 0) > max_value:
            max_value = scores.get(emo, 0)
            max_emo = emo
    return max_emo

# ============================================================
#                  STREAMLIT APPLICATION
# ============================================================

# Title
st.title("ANALISIS PERBANDINGAN METODE NB DAN KNN UNTUK KLASIFIKASI EMOSI PENGGUNA PAXEL APPS 📊")

# Display Paxel logo in sidebar (permanent logo)
logo_path = "paxel_logo.png"  # Assuming the logo is in the same directory as the app file
st.sidebar.image(logo_path, use_container_width=True)

# Sidebar with menu options
menu = st.sidebar.selectbox("Pilih Menu", 
    ["Home", "Upload Dataset", "Preprocessing", "Pelabelan Emosi", "Training Model", "Visualisasi", "WordCloud"]
)

# ================================================
# 1. HOME
# ================================================
if menu == "Home":
    st.subheader("Selamat Datang 👋")
    st.write("""Aplikasi ini melakukan Sentiment & Emotion Classification menggunakan:
    - NRC Emotion Lexicon
    - Text Preprocessing (cleaning, tokenize, filtering, stemming)
    - TF-IDF Vectorizer
    - Naive Bayes & KNN""")

# ================================================
# 2. UPLOAD DATASET
# ================================================
if menu == "Upload Dataset":
    st.subheader("📤 Upload Dataset Twitter & Playstore")
    twitter_file = st.file_uploader("Upload twitter.csv", type=["csv"])
    playstore_file = st.file_uploader("Upload playstore.csv", type=["csv"])

    if twitter_file and playstore_file:
        st.session_state.twitter = pd.read_csv(twitter_file)
        st.session_state.playstore = pd.read_csv(playstore_file)
        st.success("Dataset berhasil diupload!")

# ================================================
# 3. PREPROCESSING
# ================================================
if menu == "Preprocessing":
    st.subheader("🧹 Preprocessing Data")

    # Cek apakah dataset sudah di-upload
    if "twitter" not in st.session_state or "playstore" not in st.session_state:
        st.warning("Upload dataset terlebih dahulu!")
    else:
        if st.button("Lakukan Proses Preprocessing"):
            with st.spinner("Sedang preprocessing... Mohon tunggu ya 🤗"):
                # Rename columns untuk keseragaman
                twitter = st.session_state.twitter.rename(columns={'full_text': 'text'})
                playstore = st.session_state.playstore.rename(columns={'content': 'text'})

                # Gabungkan kedua dataset
                df = pd.concat([twitter, playstore], ignore_index=True)
                
                # Hapus baris yang memiliki nilai 'text' kosong
                df = df.dropna(subset=['text'])

                # Proses preprocessing
                df["clean_text"] = df["text"].apply(clean_text)
                df["tokens"] = df["clean_text"].apply(nltk.word_tokenize)
                df["filtered_tokens"] = df["tokens"].apply(lambda x: [w for w in x if len(w) > 2])
                df["stemmed_tokens"] = df["filtered_tokens"].apply(stem_tokens)
                df["stemmed_text"] = df["stemmed_tokens"].apply(lambda x: " ".join(x))

                # Case Folding
                df["case_folded_text"] = df["clean_text"].apply(lambda x: x.lower())  # Proses case folding

                # Simpan dataframe ke session state
                st.session_state.df = df

                st.success("Preprocessing selesai!")

                # Tampilkan beberapa hasil dari data yang telah diproses
                st.write("### Hasil Data Setelah Preprocessing")
                st.dataframe(df[["text", "clean_text", "tokens", "filtered_tokens", "stemmed_tokens", "case_folded_text"]])

                # Menambahkan tampilan hasil
                st.write("### Data yang telah diproses")
                st.dataframe(df.head())  # Menampilkan beberapa baris pertama untuk memverifikasi hasil preprocessing

# ================================================
# 4. PELABELAN EMOSI
# ================================================
if menu == "Pelabelan Emosi":
    st.subheader("🏷 Pelabelan Emosi Menggunakan NRC Emotion Lexicon")

    if "df" not in st.session_state:
        st.warning("Lakukan preprocessing dahulu!")
    else:
        if st.button("Lakukan Pelabelan Emosi"):
            df = st.session_state.df

            # Menerapkan analisis sentimen menggunakan NRC Emotion Lexicon
            df["emotion"] = df["filtered_tokens"].apply(analyze_sentiment_nrc)
            st.session_state.df = df

            # Menampilkan dataframe dengan emosi yang telah dilabeli
            st.dataframe(df[["clean_text", "emotion"]])

            # Menampilkan Diagram Batang untuk Klasifikasi Emosi menggunakan Naive Bayes
            st.write("### 📊 Klasifikasi Emosi Menggunakan Naive Bayes")
            emotion_nb = pd.Series(df["emotion"]).value_counts()  # Menghitung emosi untuk Naive Bayes
            fig_nb, ax_nb = plt.subplots()
            emotion_nb.plot(kind="bar", ax=ax_nb, color=["#ff9999", "#66b3ff", "#99ff99"])
            ax_nb.set_xlabel("Emosi")
            ax_nb.set_ylabel("Frekuensi")
            ax_nb.set_title("Klasifikasi Emosi Menggunakan Naive Bayes")
            ax_nb.bar_label(ax_nb.containers[0])  # Menambahkan label frekuensi di atas batang
            st.pyplot(fig_nb)

            # Memastikan bahwa prediksi KNN sudah ada di session_state
            if "pred_knn" in st.session_state:  # Cek apakah prediksi KNN sudah ada
                pred_knn = st.session_state.pred_knn  # Mengambil prediksi emosi untuk KNN

                # Menampilkan Diagram Batang untuk Klasifikasi Emosi menggunakan KNN
                st.write("### 📊 Klasifikasi Emosi Menggunakan KNN")
                emotion_knn = pd.Series(pred_knn).value_counts()  # Menghitung emosi untuk KNN
                fig_knn, ax_knn = plt.subplots()
                emotion_knn.plot(kind="bar", ax=ax_knn, color=["#ff9999", "#66b3ff", "#99ff99"])
                ax_knn.set_xlabel("Emosi")
                ax_knn.set_ylabel("Frekuensi")
                ax_knn.set_title("Klasifikasi Emosi Menggunakan KNN")
                ax_knn.bar_label(ax_knn.containers[0])  # Menambahkan label frekuensi di atas batang
                st.pyplot(fig_knn)
            else:
                st.warning("Model KNN belum dilatih! Lakukan pelatihan model terlebih dahulu.")

            st.success("Pelabelan emosi selesai!")
# ================================================
# 5. TRAINING MODEL
# ================================================
if menu == "Training Model":
    st.subheader("🤖 Training & Evaluasi Model NB vs KNN")

    if "df" not in st.session_state:
        st.warning("Lakukan preprocessing & pelabelan dahulu!")
    else:
        if st.button("Lakukan Training Model"):
            df = st.session_state.df
            tfidf = TfidfVectorizer(max_features=5000)
            X = tfidf.fit_transform(df["stemmed_text"])
            y = df["emotion"]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Model Naive Bayes
            nb = MultinomialNB()
            nb.fit(X_train, y_train)
            pred_nb = nb.predict(X_test)
            acc_nb = accuracy_score(y_test, pred_nb)

            # Model KNN
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train, y_train)
            pred_knn = knn.predict(X_test)
            acc_knn = accuracy_score(y_test, pred_knn)

            # Save results to session state
            st.session_state.acc_nb = acc_nb
            st.session_state.acc_knn = acc_knn
            st.session_state.pred_nb = pred_nb
            st.session_state.pred_knn = pred_knn
            st.session_state.y_test = y_test

            # Display the accuracy of the models
            st.write("### Akurasi Model")
            st.write(f"- Naive Bayes: {acc_nb:.4f}")
            st.write(f"- KNN: {acc_knn:.4f}")

            # ================================================
            # Comparison Bar Chart for Naive Bayes vs KNN
            st.write("### 📊 Perbandingan Akurasi Model Naive Bayes dan KNN")
            accuracy_comparison_df = pd.DataFrame({
                "Model": ["Naive Bayes", "KNN"],
                "Akurasi": [acc_nb, acc_knn]
            })
            fig_comparison, ax_comparison = plt.subplots()
            accuracy_comparison_df.plot(kind="bar", x="Model", y="Akurasi", ax=ax_comparison, color=["#66b3ff", "#99ff99"])
            ax_comparison.set_ylabel("Akurasi")
            ax_comparison.set_title("Perbandingan Akurasi Model Naive Bayes dan KNN")
            ax_comparison.bar_label(ax_comparison.containers[0])  # Add value labels
            st.pyplot(fig_comparison)

            # ================================================
            # Display Confusion Matrix for Naive Bayes
            st.write("## 🔵 Confusion Matrix — Naive Bayes")
            cm_nb = confusion_matrix(y_test, pred_nb)
            fig_nb, ax_nb = plt.subplots()
            sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Blues", ax=ax_nb)
            ax_nb.set_xlabel("Predicted")
            ax_nb.set_ylabel("Actual")
            st.pyplot(fig_nb)

            st.write("### Classification Report (NB)")
            st.text(classification_report(y_test, pred_nb))

            # ================================================
            # Display Confusion Matrix for KNN
            st.write("## 🟣 Confusion Matrix — KNN")
            cm_knn = confusion_matrix(y_test, pred_knn)
            fig_knn, ax_knn = plt.subplots()
            sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Purples", ax=ax_knn)
            ax_knn.set_xlabel("Predicted")
            ax_knn.set_ylabel("Actual")
            st.pyplot(fig_knn)

            st.write("### Classification Report (KNN)")
            st.text(classification_report(y_test, pred_knn))

            st.success("Training selesai!")
# ================================================
# 6. VISUALISASI
# ================================================
if menu == "Visualisasi":
    st.subheader("📈 Visualisasi Hasil")

    if "pred_nb" not in st.session_state:
        st.warning("Train model terlebih dahulu!")
    else:
        if st.button("Tampilkan Visualisasi"):
            y_test = st.session_state.y_test
            pred_nb = st.session_state.pred_nb
            pred_knn = st.session_state.pred_knn
            acc_nb = st.session_state.acc_nb
            acc_knn = st.session_state.acc_knn

            # ---- Gabungkan Hasil Prediksi Emosi - Naive Bayes dan KNN ----
            combined_predictions = pd.concat([
                pd.Series(pred_nb, name="Emotion_NB"),
                pd.Series(pred_knn, name="Emotion_KNN")
            ], axis=1)

            # Hitung distribusi emosi secara keseluruhan untuk kedua model (Naive Bayes dan KNN)
            combined_predictions = combined_predictions.apply(lambda x: "-".join(x), axis=1)
            emotion_combined = combined_predictions.value_counts()

            # Plot Pie Chart untuk Distribusi Emosi Gabungan (Naive Bayes dan KNN)
            st.write("### 🍰 Distribusi Emosi Gabungan - Naive Bayes dan KNN")
            fig_combined_pie, ax_combined_pie = plt.subplots()
            ax_combined_pie.pie(emotion_combined, labels=emotion_combined.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99', '#ffcc99'])
            ax_combined_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig_combined_pie)

            # ---- Visualisasi Kata-kata Paling Sering Digunakan (Word Frequency) ----
            st.write("### 📊 Kata-kata Paling Sering Digunakan dalam Ulasan Aplikasi")
            # Gabungkan semua teks yang telah diproses menjadi satu string
            all_words = " ".join(st.session_state.df["stemmed_text"])

            # Gunakan TfidfVectorizer untuk menghitung kata-kata yang paling sering
            tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
            tfidf_matrix = tfidf_vectorizer.fit_transform([all_words])

            # Ambil kata dan skor TF-IDF yang paling tinggi
            feature_names = tfidf_vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            word_freq = pd.DataFrame(list(zip(feature_names, scores)), columns=["Word", "Frequency"])
            word_freq = word_freq.sort_values(by="Frequency", ascending=False)

            # Visualisasi Diagram Batang untuk Kata-kata Paling Sering Digunakan
            fig, ax = plt.subplots(figsize=(10, 6))
            word_freq.plot(kind="bar", x="Word", y="Frequency", ax=ax, color='#66b3ff')
            ax.set_title("Kata-kata Paling Sering Digunakan dalam Ulasan Aplikasi")
            ax.set_xlabel("Kata")
            ax.set_ylabel("Frekuensi")
            ax.bar_label(ax.containers[0])  # Menambahkan label frekuensi di atas batang
            st.pyplot(fig)

            st.success("Visualisasi selesai!")
# ================================================
# 7. WORDCLOUD
# ================================================
if menu == "WordCloud":
    st.subheader("🧠 Visualisasi WordCloud")

    if "df" not in st.session_state:
        st.warning("Lakukan preprocessing dahulu!")
    else:
        if st.button("Generate WordCloud"):
            df = st.session_state.df

            # WordCloud per kategori emosi
            emotions = ["netral", "anger", "joy", "sadness"]
            for emotion in emotions:
                st.write(f"### WordCloud untuk Emosi {emotion.capitalize()}")
                text = " ".join(df[df["emotion"] == emotion]["stemmed_text"])
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt)

            st.success("WordCloud berhasil dibuat!")

            # Menampilkan dataset yang terlabel
            st.write("### Dataset Terlabel")
            st.dataframe(df[["clean_text","emotion"]])
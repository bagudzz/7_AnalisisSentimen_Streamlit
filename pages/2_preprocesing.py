# pages/2_preprocesing.py
# =============================================================================
# HALAMAN PREPROCESSING TEKS UNTUK ANALISIS EMOSI PELANGGAN FIRSTMEDIA
# =============================================================================
# Halaman ini melakukan preprocessing teks ulasan pelanggan dalam Bahasa Indonesia
# menggunakan library Sastrawi untuk stemming dan stopword removal.
# =============================================================================

import streamlit as st
import pandas as pd
import re
import time
from collections import Counter

# Import sidebar
from components.sidebar import render as render_sidebar

# -------------------------------------------------
# CEK KETERSEDIAAN LIBRARY SASTRAWI
# -------------------------------------------------
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False

# -------------------------------------------------
# KONFIGURASI HALAMAN
# -------------------------------------------------
st.set_page_config(
    page_title="Preprocessing - Analisis Emosi",
    page_icon="🧹",
    layout="wide"
)

# Render sidebar navigasi
render_sidebar()

# -------------------------------------------------
# KAMUS SLANG/SINGKATAN BAHASA INDONESIA
# -------------------------------------------------
# Kamus ini berisi normalisasi kata slang dan singkatan umum
# yang sering digunakan dalam ulasan pelanggan Indonesia

SLANG_DICT = {
    # Singkatan umum
    "yg": "yang",
    "dgn": "dengan",
    "utk": "untuk",
    "krn": "karena",
    "karna": "karena",
    "krna": "karena",
    "spy": "supaya",
    "biar": "supaya",
    "gak": "tidak",
    "ga": "tidak",
    "gk": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "tdk": "tidak",
    "sdh": "sudah",
    "udh": "sudah",
    "udah": "sudah",
    "blm": "belum",
    "blum": "belum",
    "belom": "belum",
    "trs": "terus",
    "trus": "terus",
    "bgt": "banget",
    "bngt": "banget",
    "banget": "sekali",
    "bener": "benar",
    "bnr": "benar",
    "emg": "memang",
    "emang": "memang",
    "lg": "lagi",
    "lgi": "lagi",
    "aja": "saja",
    "aj": "saja",
    "doang": "saja",
    "doank": "saja",
    "cuma": "hanya",
    "cm": "hanya",
    "cman": "hanya",
    "bs": "bisa",
    "bsa": "bisa",
    "jg": "juga",
    "jga": "juga",
    "jd": "jadi",
    "jdi": "jadi",
    "sm": "sama",
    "sma": "sama",
    "dr": "dari",
    "dri": "dari",
    "kmn": "kemana",
    "kmana": "kemana",
    "dmn": "dimana",
    "dmna": "dimana",
    "gmn": "gimana",
    "gmna": "gimana",
    "gimana": "bagaimana",
    "bgmn": "bagaimana",
    "knp": "kenapa",
    "knpa": "kenapa",
    "kenapa": "mengapa",
    "org": "orang",
    "orng": "orang",
    "bnyk": "banyak",
    "byk": "banyak",
    "sdikit": "sedikit",
    "sdkit": "sedikit",
    "dkit": "sedikit",
    "deket": "dekat",
    "dkt": "dekat",
    
    # Kata-kata terkait emosi/sentimen
    "bgs": "bagus",
    "jlk": "jelek",
    "brk": "buruk",
    "mantap": "mantap",
    "mantep": "mantap",
    "mntap": "mantap",
    "mntep": "mantap",
    "top": "bagus",
    "oke": "oke",
    "ok": "oke",
    "okey": "oke",
    "okay": "oke",
    "sip": "oke",
    "siip": "oke",
    "sipp": "oke",
    "parah": "parah",
    "prh": "parah",
    "ancur": "hancur",
    "hncur": "hancur",
    "nyesel": "menyesal",
    "nysel": "menyesal",
    "pus": "puas",
    "kcwa": "kecewa",
    "kcw": "kecewa",
    "kesel": "kesal",
    "ksel": "kesal",
    "mrh": "marah",
    "snang": "senang",
    "sng": "senang",
    "seneng": "senang",
    "sneng": "senang",
    "tkt": "takut",
    
    # Kata-kata terkait layanan internet
    "lemot": "lambat",
    "lmot": "lambat",
    "lmbt": "lambat",
    "cepet": "cepat",
    "cpet": "cepat",
    "cpt": "cepat",
    "lma": "lama",
    "mhl": "mahal",
    "mrh": "murah",
    "gnguan": "gangguan",
    "pts": "putus",
    "mt": "mati",
    "hdp": "hidup",
    "lncr": "lancar",
    "stbl": "stabil",
    "inet": "internet",
    "net": "internet",
    "wf": "wifi",
    "spd": "kecepatan",
    "pkt": "paket",
    "tghn": "tagihan",
    "byr": "bayar",
    
    # Kata-kata layanan pelanggan
    "cs": "customer service",
    "tkns": "teknisi",
    "komplen": "komplain",
    "kmpln": "komplain",
    "rspn": "respon",
    "rmh": "ramah",
    "jtk": "jutek",
    
    # Kata-kata waktu
    "hr": "hari",
    "hri": "hari",
    "bln": "bulan",
    "blan": "bulan",
    "thn": "tahun",
    "thun": "tahun",
    "mgu": "minggu",
    "mggu": "minggu",
    "jm": "jam",
    "mnt": "menit",
    "dtk": "detik",
    
    # Kata-kata umum lainnya
    "tapi": "tetapi",
    "tp": "tetapi",
    "kalo": "kalau",
    "klo": "kalau",
    "kl": "kalau",
    "dg": "dengan",
    "pd": "pada",
    "pda": "pada",
    "dlm": "dalam",
    "dlam": "dalam",
    "lr": "luar",
    "bwh": "bawah",
    "dpn": "depan",
    "dpan": "depan",
    "blkg": "belakang",
    "blkng": "belakang",
    "plg": "paling",
    "pling": "paling",
    "sgt": "sangat",
    "sngat": "sangat",
    "amat": "sangat",
    "trlalu": "terlalu",
    "ckp": "cukup",
    "ckup": "cukup",
    "krng": "kurang",
    "lbh": "lebih",
    "lbih": "lebih",
    
    # Tertawa/ekspresi
    "wkwk": "tertawa",
    "wkwkwk": "tertawa",
    "wkwkwkwk": "tertawa",
    "haha": "tertawa",
    "hahaha": "tertawa",
    "hihi": "tertawa",
    "hehe": "tertawa",
    "lol": "tertawa",
}

# -------------------------------------------------
# INISIALISASI SASTRAWI DENGAN CACHE
# -------------------------------------------------
@st.cache_resource
def init_sastrawi():
    """
    Inisialisasi Sastrawi stemmer dan stopword.
    Menggunakan cache_resource agar tidak perlu load ulang setiap kali halaman di-refresh.
    """
    if not SASTRAWI_AVAILABLE:
        return None, set()
    
    # Inisialisasi Stemmer Factory
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()    

    # Inisialisasi Stopword Factory dan ambil daftar stopwords
    stopword_factory = StopWordRemoverFactory()
    stopwords = set(stopword_factory.get_stop_words())  
    return stemmer, stopwords

# -------------------------------------------------
# FUNGSI-FUNGSI PREPROCESSING
# -------------------------------------------------

def clean_text(text):
    """
    Membersihkan teks dari noise (URL, mention, hashtag, karakter khusus, angka).
    
    Args:
        text: String teks yang akan dibersihkan
        
    Returns:
        String teks yang sudah dibersihkan
    """
    if not isinstance(text, str):
        return ""
    
    # Hapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Hapus mention (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Hapus hashtag (#hashtag)
    text = re.sub(r'#\w+', '', text)
    
    # Hapus email
    text = re.sub(r'\S+@\S+', '', text)
    
    # Hapus nomor telepon (format Indonesia)
    text = re.sub(r'(\+62|62|0)[0-9]{9,12}', '', text)
    
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    
    # Hapus karakter khusus dan tanda baca, kecuali spasi
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Hapus underscore
    text = re.sub(r'_', ' ', text)
    
    # Hapus whitespace berlebih (ganti multiple spaces dengan single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Hapus whitespace di awal dan akhir
    text = text.strip()
    
    return text


def case_folding(text):
    """
    Mengubah semua huruf menjadi lowercase (huruf kecil).
    
    Args:
        text: String teks
        
    Returns:
        String teks dalam lowercase
    """
    if not isinstance(text, str):
        return ""
    return text.lower()


def normalize_slang(text, slang_dict):
    """
    Menormalisasi kata slang/singkatan menjadi kata baku berdasarkan kamus.
    
    Args:
        text: String teks
        slang_dict: Dictionary mapping slang -> kata baku
        
    Returns:
        String teks yang sudah dinormalisasi
    """
    if not isinstance(text, str):
        return ""
    
    words = text.split()
    normalized_words = []
    
    for word in words:
        # Cek apakah kata ada di kamus slang (case insensitive)
        normalized = slang_dict.get(word.lower(), word)
        normalized_words.append(normalized)
    
    return ' '.join(normalized_words)


def tokenize(text):
    """
    Memecah teks menjadi token (kata-kata individual).
    
    Args:
        text: String teks
        
    Returns:
        List of tokens (kata-kata)
    """
    if not isinstance(text, str):
        return []
    # Split berdasarkan whitespace dan filter kata kosong
    tokens = [word.strip() for word in text.split() if word.strip()]
    return tokens


def remove_stopwords(tokens, stopwords):
    """
    Menghapus stopwords (kata-kata umum yang tidak bermakna) dari list token.
    
    Args:
        tokens: List of tokens
        stopwords: Set of stopwords
        
    Returns:
        List of tokens tanpa stopwords
    """
    if not tokens:
        return []
    # Filter token yang bukan stopword dan panjang > 1 karakter
    filtered = [token for token in tokens if token.lower() not in stopwords and len(token) > 1]
    return filtered


def stem_tokens(tokens, stemmer):
    """
    Melakukan stemming pada setiap token (mengubah kata ke bentuk dasar).
    
    Args:
        tokens: List of tokens
        stemmer: Sastrawi stemmer object
        
    Returns:
        List of stemmed tokens
    """
    if not tokens or stemmer is None:
        return tokens
    # Stem setiap token
    stemmed = [stemmer.stem(token) for token in tokens]
    return stemmed


def preprocess_text(text, stemmer, stopwords, slang_dict,
                    do_clean=True, do_case_fold=True, do_normalize=True,
                    do_tokenize=True, do_stopword=True, do_stem=True):
    """
    Melakukan preprocessing lengkap pada satu teks.
    
    Args:
        text: String teks input
        stemmer: Sastrawi stemmer object
        stopwords: Set of stopwords
        slang_dict: Dictionary slang -> kata baku
        do_*: Boolean flags untuk setiap tahap preprocessing
        
    Returns:
        Tuple (preprocessed_text, token_list)
    """
    if not isinstance(text, str) or not text.strip():
        return "", []
    
    result = text
    
    # Tahap 1: Cleaning
    if do_clean:
        result = clean_text(result)
    
    # Tahap 2: Case Folding
    if do_case_fold:
        result = case_folding(result)
    
    # Tahap 3: Normalisasi Slang
    if do_normalize:
        result = normalize_slang(result, slang_dict)
    
    # Tahap 4: Tokenisasi
    if do_tokenize:
        tokens = tokenize(result)
    else:
        tokens = [result] if result else []
    
    # Tahap 5: Stopword Removal
    if do_stopword and stopwords:
        tokens = remove_stopwords(tokens, stopwords)
    
    # Tahap 6: Stemming
    if do_stem and stemmer:
        tokens = stem_tokens(tokens, stemmer)
    
    # Gabungkan tokens menjadi string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text, tokens


def calculate_statistics(df, original_col, preprocessed_col):
    """
    Menghitung statistik hasil preprocessing.
    
    Args:
        df: DataFrame dengan hasil preprocessing
        original_col: Nama kolom teks asli
        preprocessed_col: Nama kolom teks hasil preprocessing
        
    Returns:
        Dictionary berisi berbagai statistik
    """
    stats = {}
    
    # Total dokumen
    stats['total_documents'] = len(df)
    
    # Rata-rata panjang karakter sebelum preprocessing
    stats['avg_char_before'] = df[original_col].astype(str).str.len().mean()
    
    # Rata-rata panjang karakter setelah preprocessing
    stats['avg_char_after'] = df[preprocessed_col].astype(str).str.len().mean()
    
    # Persentase pengurangan karakter
    if stats['avg_char_before'] > 0:
        stats['char_reduction_pct'] = (
            (stats['avg_char_before'] - stats['avg_char_after']) 
            / stats['avg_char_before'] * 100
        )
    else:
        stats['char_reduction_pct'] = 0
    
    # Statistik token
    if 'token_count' in df.columns:
        stats['avg_token_count'] = df['token_count'].mean()
        stats['min_token_count'] = df['token_count'].min()
        stats['max_token_count'] = df['token_count'].max()
    
    # Hitung dokumen yang menjadi kosong setelah preprocessing
    stats['empty_docs'] = (df[preprocessed_col].astype(str).str.strip() == '').sum()
    
    # Kumpulkan semua token untuk analisis frekuensi
    if 'tokens' in df.columns:
        all_tokens = []
        for tokens in df['tokens']:
            if isinstance(tokens, list):
                all_tokens.extend(tokens)
        
        stats['total_tokens'] = len(all_tokens)
        stats['unique_tokens'] = len(set(all_tokens))
        
        # Top 10 kata paling sering muncul
        token_counts = Counter(all_tokens)
        stats['top_tokens'] = token_counts.most_common(10)
    
    return stats


# =============================================================================
# MULAI KONTEN HALAMAN
# =============================================================================

# Header halaman
st.title("🧹 Preprocessing Teks")
st.markdown("""
Halaman ini melakukan preprocessing teks ulasan pelanggan untuk mempersiapkan 
data sebelum dilakukan pelabelan emosi dan training model.
""")

st.divider()

# -------------------------------------------------
# CEK DATA DARI SESSION STATE
# -------------------------------------------------
if 'raw_df' not in st.session_state or st.session_state.raw_df is None:
    st.warning("⚠️ Data belum diunggah!")
    st.info("""
    Silakan unggah dataset terlebih dahulu di halaman **Upload Dataset**.
    
    Klik tombol di bawah untuk menuju halaman upload:
    """)
    
    if st.button("📤 Ke Halaman Upload Dataset", type="primary"):
        st.switch_page("pages/1_Upload_Dataset.py")
    
    st.stop()

# Ambil data dari session state
df = st.session_state.raw_df.copy()

st.success(f"✅ Data berhasil dimuat: **{len(df):,}** baris")

# -------------------------------------------------
# CEK KETERSEDIAAN SASTRAWI
# -------------------------------------------------
if not SASTRAWI_AVAILABLE:
    st.error("""
    ❌ **Library Sastrawi tidak tersedia!**
    
    Sastrawi diperlukan untuk stemming dan stopword removal Bahasa Indonesia.
    
    Silakan install dengan perintah:
    ```bash
    pip install Sastrawi
    ```
    
    Setelah install, restart aplikasi Streamlit.
    """)
    st.stop()

# Inisialisasi Sastrawi (menggunakan cache)
with st.spinner("⏳ Memuat Sastrawi stemmer dan stopwords..."):
    stemmer, stopwords = init_sastrawi()

st.info(f"📚 Sastrawi berhasil dimuat dengan **{len(stopwords):,}** stopwords Bahasa Indonesia")

# -------------------------------------------------
# KONFIGURASI PREPROCESSING
# -------------------------------------------------
st.subheader("⚙️ Konfigurasi Preprocessing")

col_config1, col_config2 = st.columns([1, 2])

with col_config1:
    # Pilih kolom teks yang akan dipreprocessing
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not text_columns:
        st.error("❌ Tidak ada kolom teks (string) dalam dataset!")
        st.stop()
    
    # Coba deteksi kolom teks secara otomatis berdasarkan nama kolom
    default_col_index = 0
    for i, col in enumerate(text_columns):
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['text', 'review', 'ulasan', 'komentar', 'content', 'isi', 'pesan']):
            default_col_index = i
            break
    
    selected_column = st.selectbox(
        "📝 Pilih Kolom Teks",
        options=text_columns,
        index=default_col_index,
        help="Pilih kolom yang berisi teks ulasan pelanggan yang akan dipreprocessing"
    )
    
    # Tampilkan preview sample data dari kolom yang dipilih
    st.markdown("**Preview data (5 sampel pertama):**")
    sample_data = df[selected_column].dropna().head(5).tolist()
    for i, sample in enumerate(sample_data, 1):
        sample_str = str(sample)
        # Potong teks jika terlalu panjang
        display_text = sample_str[:300] + "..." if len(sample_str) > 300 else sample_str
        with st.expander(f"Sampel {i}"):
            st.text(display_text)

with col_config2:
    st.markdown("**Pilih tahapan preprocessing yang akan dijalankan:**")
    
    col_opt1, col_opt2 = st.columns(2)
    
    with col_opt1:
        do_clean = st.checkbox(
            "🧼 Cleaning",
            value=True,
            help="Hapus URL, mention (@), hashtag (#), email, nomor telepon, angka, dan karakter khusus"
        )
        
        do_case_fold = st.checkbox(
            "🔡 Case Folding",
            value=True,
            help="Ubah semua huruf menjadi lowercase (huruf kecil)"
        )
        
        do_normalize = st.checkbox(
            "📝 Normalisasi Slang",
            value=True,
            help=f"Normalisasi {len(SLANG_DICT)} kata slang/singkatan ke bentuk baku"
        )
    
    with col_opt2:
        do_tokenize = st.checkbox(
            "✂️ Tokenisasi",
            value=True,
            help="Pecah teks menjadi kata-kata individual (token)"
        )
        
        do_stopword = st.checkbox(
            "🚫 Hapus Stopwords",
            value=True,
            help=f"Hapus {len(stopwords)} stopwords Bahasa Indonesia (kata umum tidak bermakna)"
        )
        
        do_stem = st.checkbox(
            "🌱 Stemming",
            value=True,
            help="Ubah kata ke bentuk dasar menggunakan algoritma Sastrawi"
        )
    
    # Expander untuk melihat daftar stopwords
    with st.expander("📋 Lihat contoh stopwords"):
        stopword_sample = sorted(list(stopwords))[:50]
        st.text(", ".join(stopword_sample) + ", ...")
    
    # Expander untuk melihat kamus slang
    with st.expander("📋 Lihat contoh kamus slang"):
        slang_sample = dict(list(SLANG_DICT.items())[:20])
        for slang, baku in slang_sample.items():
            st.text(f"{slang} → {baku}")

st.divider()
# -------------------------------------------------
# TOMBOL MULAI PREPROCESSING
# -------------------------------------------------
if st.button("🚀 Mulai Preprocessing", type="primary", use_container_width=True):

    # Validasi: minimal satu tahap harus dipilih
    if not any([do_clean, do_case_fold, do_normalize, do_tokenize, do_stopword, do_stem]):
        st.error("❌ Pilih minimal satu tahap preprocessing!")
        st.stop()

    # -------------------------------------------------
    # PROSES PREPROCESSING
    # -------------------------------------------------
    st.subheader("⏳ Proses Preprocessing")

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Catat waktu mulai
    start_time = time.time()

    # Siapkan list untuk menyimpan hasil
    preprocessed_texts = []
    token_lists = []

    # Pastikan index berurutan 0..n-1
    df_proc = df.reset_index(drop=True)
    total_rows = len(df_proc)

    # Loop setiap baris dan lakukan preprocessing
    for idx, row in df_proc.iterrows():

        text = row[selected_column]

        # Preprocess teks
        preprocessed, tokens = preprocess_text(
            text=text,
            stemmer=stemmer,
            stopwords=stopwords,
            slang_dict=SLANG_DICT,
            do_clean=do_clean,
            do_case_fold=do_case_fold,
            do_normalize=do_normalize,
            do_tokenize=do_tokenize,
            do_stopword=do_stopword,
            do_stem=do_stem,
        )

        preprocessed_texts.append(preprocessed)
        token_lists.append(tokens)

        # Update progress bar
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(
            f"Memproses dokumen {idx + 1:,} dari {total_rows:,} ({progress*100:.1f}%)"
        )

    # ---- SELESAI LOOP, lanjut di bawah (masih di dalam if button) ----
    end_time = time.time()
    processing_time = end_time - start_time

    progress_bar.progress(1.0)
    status_text.text(f"✅ Preprocessing selesai dalam {processing_time:.2f} detik")

    # Tambahkan kolom hasil ke dataframe
    result_df = df_proc.copy()
    result_df["text_preprocessed"] = preprocessed_texts
    result_df["tokens"] = token_lists
    result_df["token_count"] = [len(tokens) for tokens in token_lists]

    # Simpan ke session state
    st.session_state["preprocessed_df"] = result_df
    st.session_state["selected_text_column"] = selected_column

    st.divider()

    # -------------------------------------------------
    # HASIL PREPROCESSING
    # -------------------------------------------------
    st.subheader("📊 Hasil Preprocessing")

    # Hitung statistik
    stats = calculate_statistics(result_df, selected_column, "text_preprocessed")

    # Tampilkan statistik dalam metrics
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        st.metric("📄 Total Dokumen", f"{stats['total_documents']:,}")

    with col_stat2:
        st.metric(
            "📏 Rata-rata Karakter",
            f"{stats['avg_char_after']:.0f}",
            delta=f"-{stats['char_reduction_pct']:.1f}%",
            delta_color="normal",
        )

    with col_stat3:
        st.metric(
            "🔤 Rata-rata Token",
            f"{stats.get('avg_token_count', 0):.1f}",
        )

    with col_stat4:
        empty_count = stats["empty_docs"]
        if empty_count > 0:
            st.metric(
                "⚠️ Dokumen Kosong",
                f"{empty_count:,}",
                delta="perlu review",
                delta_color="inverse",
            )
        else:
            st.metric("✅ Dokumen Kosong", "0")

    col_extra1, col_extra2 = st.columns(2)
    with col_extra1:
        st.metric(
            "📊 Total Semua Token",
            f"{stats.get('total_tokens', 0):,}",
        )
    with col_extra2:
        st.metric(
            "🔠 Token Unik",
            f"{stats.get('unique_tokens', 0):,}",
        )

    st.divider()

    # -------------------------------------------------
    # TOP 10 KATA PALING SERING
    # -------------------------------------------------
    st.subheader("🏆 Top 10 Kata Paling Sering")

    if "top_tokens" in stats and stats["top_tokens"]:
        top_words_df = pd.DataFrame(stats["top_tokens"], columns=["Kata", "Frekuensi"])
        top_words_df.index = top_words_df.index + 1
        top_words_df.index.name = "Rank"

        col_top1, col_top2 = st.columns([1, 2])

        with col_top1:
            st.dataframe(top_words_df, use_container_width=True)

        with col_top2:
            st.bar_chart(top_words_df.set_index("Kata")["Frekuensi"])
    else:
        st.info("Tidak ada data token untuk ditampilkan.")

    st.divider()

    # -------------------------------------------------
    # TABEL PERBANDINGAN SEBELUM & SESUDAH
    # -------------------------------------------------
    st.subheader("📋 Perbandingan Teks Sebelum & Sesudah Preprocessing")

    comparison_df = result_df[
        [selected_column, "text_preprocessed", "token_count"]
    ].head(10)
    comparison_df.columns = ["Teks Asli", "Teks Preprocessed", "Jumlah Token"]
    comparison_df.index = comparison_df.index + 1
    comparison_df.index.name = "No"

    st.dataframe(comparison_df, use_container_width=True, height=400)

    st.divider()
# -------------------------------------------------
# TOMBOL LANJUT KE PELABELAN EMOSI
# -------------------------------------------------
st.divider()

if "preprocessed_df" in st.session_state and st.session_state.preprocessed_df is not None:
    st.info("✅ Preprocessing selesai. Kamu bisa lanjut ke tahap **Pelabelan Emosi**.")

    lanjut = st.button(
        "➡️ Lanjut ke Pelabelan Emosi",
        type="primary",
        use_container_width=True,
    )

    if lanjut:
        st.switch_page("pages/3_pelabelan.py")

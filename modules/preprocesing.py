import re
import string
import pandas as pd

# note: install dulu di venv -> pip install Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ====== STOPWORDS SEDERHANA (BISA KAMU PERLUAS) ======
stopwords_id = set([
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "tidak", "bukan",
    "adalah", "dengan", "untuk", "sebagai", "juga", "sudah", "belum",
    "nya", "agar", "supaya", "akan", "tapi", "tetapi", "kalau", "saat",
    "atau", "karena", "jadi", "pada", "oleh"
])

# slang sederhana
slang_map = {
    "bgt": "banget",
    "bgd": "banget",
    "tdk": "tidak",
    "gak": "tidak",
    "ga": "tidak",
    "jg": "juga",
    "udh": "sudah",
    "udah": "sudah",
    "bkn": "bukan"
}

# inisialisasi stemmer Sastrawi sekali saja
_factory = StemmerFactory()
_stemmer = _factory.create_stemmer()


def preprocess_text(text: str):
    """
    Melakukan:
    - cleaning (hapus URL, mention, hashtag, karakter aneh)
    - case folding (lowercase)
    - tokenizing
    - filtering (stopword + panjang < 2)
    - stemming (Sastrawi)

    Return: cleaned_text, tokens, stemmed_text
    """
    if pd.isna(text):
        return "", [], ""

    # ===== 1. CLEANING + CASE FOLDING =====  #note
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)   # hapus URL
    text = re.sub(r'@\w+', ' ', text)                     # hapus mention
    text = re.sub(r'#\w+', ' ', text)                     # hapus hashtag
    text = re.sub(r'\n', ' ', text)                       # newline -> spasi
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits))
    text = re.sub(r'\s+', ' ', text).strip()

    cleaned_text = text  # hasil cleaning + case folding

    # ===== 2. TOKENIZING =====  #note
    tokens = cleaned_text.split()

    # normalisasi slang
    tokens = [slang_map.get(tok, tok) for tok in tokens]

    # ===== 3. FILTERING (STOPWORD + PANJANG < 2) =====  #note
    tokens_filtered = [
        tok for tok in tokens
        if tok not in stopwords_id and len(tok) > 1
    ]

    # ===== 4. STEMMING (SASTRAWI) =====  #note
    # stem per token supaya jelas
    stemmed_tokens = [_stemmer.stem(tok) for tok in tokens_filtered]
    stemmed_text = " ".join(stemmed_tokens)

    return cleaned_text, tokens_filtered, stemmed_text

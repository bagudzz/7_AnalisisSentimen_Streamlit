# modules/labelling.py
# ================================================================
# Pelabelan emosi berbasis lexicon CSV (Bahasa Indonesia)
# Target label final: anger, joy, sadness, neutral
# ================================================================

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd


# ------------------------------------------------
# Mapping label pada CSV -> label final (4 kelas)
# ------------------------------------------------
DEFAULT_MAP_4 = {
    # anger
    "anger": "anger",
    "angry": "anger",
    "marah": "anger",
    "kesal": "anger",
    "benci": "anger",

    # joy
    "joy": "joy",
    "happy": "joy",
    "senang": "joy",
    "bahagia": "joy",
    "gembira": "joy",

    # sadness
    "sadness": "sadness",
    "sad": "sadness",
    "sedih": "sadness",
    "kecewa": "sadness",
}


# jika tie (seri), pilih berdasarkan prioritas ini
# (komplain layanan biasanya anger lebih dominan)
DEFAULT_TIE_PRIORITY = ["anger", "sadness", "joy"]


@dataclass
class LexiconResult:
    label: str
    scores: Dict[str, int]
    has_match: bool
    is_tie: bool


def _norm(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()


def load_emotion_lexicon_from_csv(
    csv_path: str,
    *,
    word_col: Optional[str] = None,
    emotion_col: Optional[str] = None,
    map_4: Optional[Dict[str, str]] = None,
) -> Dict[str, set]:
    """
    Membaca lexicon dari CSV -> dict set:
      {"anger": set(words), "joy": set(words), "sadness": set(words)}

    Mendukung 2 format:
    A) kolom kata + kolom emosi:
       - word, emotion (atau kata, emosi, label, kategori)
       - 1 baris = 1 (kata, emosi)

    B) one-hot:
       - word | anger | joy | sadness | ...
       - bernilai 0/1

    Returns: lex_sets untuk matching token.
    """
    if map_4 is None:
        map_4 = DEFAULT_MAP_4

    df_lex = pd.read_csv(csv_path)
    df_lex.columns = [c.strip() for c in df_lex.columns]
    cols = list(df_lex.columns)
    cols_lower = {c.lower(): c for c in cols}

    # auto detect word_col
    if word_col is None:
        for cand in ["word", "kata", "term", "token"]:
            if cand in cols_lower:
                word_col = cols_lower[cand]
                break

    # auto detect emotion_col
    if emotion_col is None:
        for cand in ["emotion", "emosi", "label", "category", "kategori"]:
            if cand in cols_lower:
                emotion_col = cols_lower[cand]
                break

    lex_sets = {"anger": set(), "joy": set(), "sadness": set()}

    # -------------------------
    # MODE A: word + emotion
    # -------------------------
    if word_col and emotion_col:
        for _, r in df_lex.iterrows():
            w = _norm(r.get(word_col))
            emo_raw = _norm(r.get(emotion_col))
            if not w or not emo_raw:
                continue
            emo_final = map_4.get(emo_raw)
            if emo_final in lex_sets:
                lex_sets[emo_final].add(w)
        return lex_sets

    # -------------------------
    # MODE B: one-hot columns
    # -------------------------
    if word_col is None:
        word_col = df_lex.columns[0]  # fallback

    # cari kolom emosi yang mungkin
    possible_cols = []
    for c in df_lex.columns:
        cl = c.lower()
        if cl in ["anger", "joy", "sadness", "marah", "senang", "sedih", "happy", "sad", "angry"]:
            possible_cols.append(c)

    if not possible_cols:
        raise ValueError(
            "Format lexicon tidak dikenali. "
            "Harus ada (word,kata,...) & (emotion,emosi,...) atau kolom one-hot anger/joy/sadness."
        )

    for _, r in df_lex.iterrows():
        w = _norm(r.get(word_col))
        if not w:
            continue

        for emo_col in possible_cols:
            val = r.get(emo_col)
            active = False
            try:
                active = int(val) == 1
            except Exception:
                active = _norm(val) in ["1", "true", "yes", "y"]

            if not active:
                continue

            emo_final = map_4.get(emo_col.lower())
            if emo_final in lex_sets:
                lex_sets[emo_final].add(w)

    return lex_sets


def stem_lexicon_sets(lex_sets: Dict[str, set], stemmer) -> Dict[str, set]:
    """
    Stemming semua kata pada lexicon agar selaras dengan token preprocessing (yang distemming).
    """
    if stemmer is None:
        return lex_sets

    out = {}
    for emo, words in lex_sets.items():
        out[emo] = set()
        for w in words:
            ww = _norm(w)
            if ww:
                out[emo].add(stemmer.stem(ww))
    return out


def analyze_emotion_lexicon(
    tokens: List[str],
    lex_sets: Dict[str, set],
    *,
    tie_priority: Optional[List[str]] = None,
) -> LexiconResult:
    """
    Hitung skor emosi dari tokens berdasarkan lexicon set.
    Rules:
    - skor tertinggi -> label itu
    - tidak ada match -> neutral
    - tie -> pilih berdasarkan tie_priority (default anger > sadness > joy)
    """
    if tie_priority is None:
        tie_priority = DEFAULT_TIE_PRIORITY

    if not tokens:
        return LexiconResult("neutral", {}, has_match=False, is_tie=False)

    scores = Counter()
    has_match = False

    for t in tokens:
        w = _norm(t)
        if not w:
            continue
        for emo, s in lex_sets.items():
            if w in s:
                scores[emo] += 1
                has_match = True

    if not scores:
        return LexiconResult("neutral", {}, has_match=False, is_tie=False)

    max_score = max(scores.values())
    top = [emo for emo, sc in scores.items() if sc == max_score]

    if len(top) == 1:
        return LexiconResult(top[0], dict(scores), has_match=True, is_tie=False)

    # tie
    for p in tie_priority:
        if p in top:
            return LexiconResult(p, dict(scores), has_match=True, is_tie=True)

    return LexiconResult("neutral", dict(scores), has_match=True, is_tie=True)


def label_dataframe_lexicon(
    df: pd.DataFrame,
    tokens_col: str = "tokens",
    lex_sets: Optional[Dict[str, set]] = None,
    *,
    tie_priority: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Label semua baris df berdasarkan tokens.
    Output kolom:
      - label_emosi
      - score_anger, score_joy, score_sadness
      - debug_has_match (bool)
      - debug_is_tie (bool)
    """
    if lex_sets is None:
        raise ValueError("lex_sets belum diisi. Load lexicon dulu.")

    labels: List[str] = []
    score_anger: List[int] = []
    score_joy: List[int] = []
    score_sad: List[int] = []
    debug_has_match: List[bool] = []
    debug_is_tie: List[bool] = []

    for _, row in df.iterrows():
        tokens = row.get(tokens_col, [])
        if not isinstance(tokens, list):
            tokens = str(tokens).split()

        res = analyze_emotion_lexicon(tokens, lex_sets, tie_priority=tie_priority)

        labels.append(res.label)
        score_anger.append(int(res.scores.get("anger", 0)))
        score_joy.append(int(res.scores.get("joy", 0)))
        score_sad.append(int(res.scores.get("sadness", 0)))

        debug_has_match.append(bool(res.has_match))
        debug_is_tie.append(bool(res.is_tie))

    out = df.copy()
    out["label_emosi"] = labels
    out["score_anger"] = score_anger
    out["score_joy"] = score_joy
    out["score_sadness"] = score_sad
    out["debug_has_match"] = debug_has_match
    out["debug_is_tie"] = debug_is_tie
    return out


def emotion_distribution(df: pd.DataFrame, label_col: str = "label_emosi") -> pd.Series:
    if label_col not in df.columns:
        raise ValueError(f"Kolom '{label_col}' tidak ditemukan di DataFrame.")
    return df[label_col].value_counts().sort_index()


def debug_neutral_breakdown(df: pd.DataFrame) -> Dict[str, int]:
    """
    Breakdown neutral:
    - neutral_no_match: label neutral karena tidak ada match (skor 0)
    - neutral_tie: label neutral karena tie (seri) (kalau kamu masih pakai aturan neutral)
      (di versi ini tie tidak jadi neutral, tapi tetap kita laporkan)
    """
    if "label_emosi" not in df.columns:
        return {}
    if "debug_has_match" not in df.columns:
        return {}

    neutral = df[df["label_emosi"] == "neutral"]
    neutral_no_match = int((neutral["debug_has_match"] == False).sum())

    # tie count across all rows
    tie_all = int(df.get("debug_is_tie", pd.Series([False] * len(df))).sum())

    return {
        "neutral_total": int(len(neutral)),
        "neutral_no_match": neutral_no_match,
        "tie_all_rows": tie_all,
    }

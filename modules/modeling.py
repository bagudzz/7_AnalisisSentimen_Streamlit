# modules/modeling.py
# ================================================================
# Training & evaluasi SVM sederhana (TF-IDF + SVC)
# Label: anger / joy / sadness / neutral
# ================================================================

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

DEFAULT_TFIDF_PARAMS = {
    "max_features": 20000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}

DEFAULT_SVM_PARAMS = {
    "C": 1.0,
    "gamma": "scale",
    "degree": 3,
}


@dataclass
class TrainResult:
    model: Any
    vectorizer: TfidfVectorizer
    label_encoder: LabelEncoder
    metrics: Dict[str, Any]


def prepare_data(
    df: pd.DataFrame,
    text_col: str = "text_preprocessed",
    label_col: str = "label_emosi",
    min_text_len: int = 1,
) -> Tuple[pd.Series, pd.Series]:
    if text_col not in df.columns:
        raise ValueError(f"Kolom '{text_col}' tidak ditemukan.")
    if label_col not in df.columns:
        raise ValueError(f"Kolom '{label_col}' tidak ditemukan.")

    tmp = df[[text_col, label_col]].copy()
    tmp[text_col] = tmp[text_col].astype(str).fillna("").str.strip()
    tmp[label_col] = tmp[label_col].astype(str).fillna("").str.strip()

    tmp = tmp[tmp[text_col].str.len() >= min_text_len]
    tmp = tmp[tmp[label_col].str.len() > 0]

    return tmp[text_col], tmp[label_col]


def train_svm_simple(
    texts: pd.Series,
    labels: pd.Series,
    kernel: str = "linear",
    test_size: float = 0.2,
    random_state: int = 42,
    use_balanced_weight: bool = True,
    tfidf_params: Optional[Dict[str, Any]] = None,
    svm_params: Optional[Dict[str, Any]] = None,
    multiclass_strategy: str = "ovo",
) -> TrainResult:
    # --- merge params ---
    tfidf_cfg = dict(DEFAULT_TFIDF_PARAMS)
    if tfidf_params:
        tfidf_cfg.update(tfidf_params)

    svm_cfg = dict(DEFAULT_SVM_PARAMS)
    if svm_params:
        svm_cfg.update(svm_params)

    class_weight = "balanced" if use_balanced_weight else None

    # --- encode labels ---
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # --- split ---
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    # --- TF-IDF ---
    vectorizer = TfidfVectorizer(**tfidf_cfg)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # --- SVM ---
    base_svc = SVC(
        kernel=kernel,
        C=svm_cfg["C"],
        gamma=svm_cfg["gamma"],
        degree=svm_cfg["degree"],
        class_weight=class_weight,
    )

    if multiclass_strategy == "ovr":
        model = OneVsRestClassifier(base_svc)
    else:
        model = base_svc

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=le.classes_, zero_division=0
    )

    metrics = {
        "kernel": kernel,
        "test_size": test_size,
        "random_state": random_state,
        "class_weight": class_weight,
        "tfidf_params": tfidf_cfg,
        "svm_params": svm_cfg,
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "confusion_matrix": cm,
        "labels": le.classes_.tolist(),
        "classification_report": report,
        "multiclass_strategy" : multiclass_strategy,
    }

    return TrainResult(
        model=model,
        vectorizer=vectorizer,
        label_encoder=le,
        metrics=metrics,
    )

    
def undersample_dataframe(
    df: pd.DataFrame, label_col: str = "label_emosi", random_state: int = 42,
) -> pd.DataFrame:
    """
    Undersampling semua kelas ke ukuran kelas minoritas.
    """
    classes = df[label_col].unique()
    min_count = df[label_col].value_counts().min()

    dfs = []
    for c in classes:
        df_c = df[df[label_col] == c]
        df_c_down = resample(
            df_c,
            replace=False,
            n_samples=min_count,
            random_state=random_state,
        )
        dfs.append(df_c_down)

    return pd.concat(dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)

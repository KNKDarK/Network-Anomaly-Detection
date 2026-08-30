"""
Data loading, cleaning, and feature engineering utilities.

Handles raw CICIDS2017 CSVs from ``data/raw`` and produces clean,
encoded, scaled arrays saved to ``data/processed``.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Columns in the raw CICIDS2017 CSVs that are not useful flow features.
DROP_COLUMNS = ["Flow ID", "Source IP", "Destination IP", "Timestamp", "Fwd Header Length.1", "SimillarHTTP"]


def load_raw_csvs(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and concatenate all raw CICIDS2017 CSV files in ``raw_dir``."""
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Download CICIDS2017 and place the "
            "raw CSVs in data/raw/ (see README)."
        )

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        frames.append(df)
        print(f"Loaded {f.name}: {df.shape}")

    data = pd.concat(frames, ignore_index=True)
    return data


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-feature columns, convert types, and replace infinite values."""
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # Label column is the last one
    y = df.iloc[:, -1].astype(str)

    X = df.iloc[:, :-1]
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are entirely missing
    X = X.dropna(axis=1, how="all")

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Scale all numeric features with StandardScaler."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols)],
        remainder="drop",
    )
    return preprocessor


def prepare_data(sample_limit: int | None = None) -> tuple:
    """
    End-to-end preprocessing pipeline.

    Returns a tuple:
        (X_train, X_test, y_train, y_test, le, preprocessor, feature_names)
    """
    raw = load_raw_csvs()
    X, y = clean_features(raw)

    if sample_limit:
        X = X.iloc[:sample_limit]
        y = y.iloc[:sample_limit]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    preprocessor = build_preprocessor(X)
    X_scaled = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )

    # Cache for later reuse
    np.savez_compressed(
        PROCESSED_DIR / "processed_arrays.npz",
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    )
    joblib.dump(le, PROCESSED_DIR / "label_encoder.joblib")
    joblib.dump(preprocessor, PROCESSED_DIR / "preprocessor.joblib")

    return X_train, X_test, y_train, y_test, le, preprocessor, feature_names


def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """Balance classes using SMOTE."""
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 10) -> tuple:
    """
    Reshape tabular data into time-windowed sequences for the LSTM.
    Returns (X_seq, y_seq) where each sample is a window of ``seq_len`` rows.
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len : i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


if __name__ == "__main__":
    prepare_data()
    print("Preprocessing complete. Processed arrays saved to", PROCESSED_DIR)

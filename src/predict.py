"""
Prediction helpers shared between the demo app and notebooks.

Loads a trained model, preprocessor, and label encoder, then exposes a
``predict(df)`` function that returns predicted labels and SHAP explanations.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


class Predictor:
    """Encapsulate a trained model + its preprocessing pipeline."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        preprocessor_path: Path | str | None = None,
        encoder_path: Path | str | None = None,
    ):
        model_path = Path(model_path) if model_path else _default_model_path()
        preprocessor_path = Path(preprocessor_path) if preprocessor_path else MODELS_DIR / "preprocessor.joblib"
        encoder_path = Path(encoder_path) if encoder_path else MODELS_DIR / "label_encoder.joblib"

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.encoder = joblib.load(encoder_path)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return class labels (strings) for the rows of ``df``."""
        X = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        # Keep only columns the preprocessor understands
        X = X.reindex(columns=_preprocessor_columns(self.preprocessor), fill_value=np.nan)
        X_scaled = self.preprocessor.transform(X)
        y_encoded = self.model.predict(X_scaled)
        return self.encoder.inverse_transform(y_encoded)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X = X.reindex(columns=_preprocessor_columns(self.preprocessor), fill_value=np.nan)
        X_scaled = self.preprocessor.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        raise AttributeError("Model does not implement predict_proba.")


def _default_model_path() -> Path:
    """Prefer Random Forest, else the first available model file."""
    for name in ["random_forest.pkl", "xgboost_model.pkl", "lstm_model.h5"]:
        p = MODELS_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "No trained model found. Train models first with: python -m src.train_models"
    )


def _preprocessor_columns(preprocessor) -> list[str]:
    """Derive the raw column names (without the num__ prefix) the preprocessor expects."""
    cols = []
    for name, _transformer, _columns in preprocessor.transformers_:
        cols.extend(_columns)
    return cols

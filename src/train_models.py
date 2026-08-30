"""
Train, evaluate, and save the three anomaly-detection models.

Models:
  1. Random Forest Classifier
  2. XGBoost Classifier
  3. LSTM (Keras) on time-windowed sequences

Run:  python -m src.train_models
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest...")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")
    return model, elapsed


def train_xgboost(X_train, y_train):
    print("\nTraining XGBoost...")
    t0 = time.time()
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")
    return model, elapsed


def train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, n_classes):
    """Train a small LSTM on time-windowed sequences."""
    import tensorflow as tf
    from tensorflow.keras import layers, models  # noqa: F401

    print("\nTraining LSTM...")
    t0 = time.time()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=10,
        batch_size=256,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")
    return model, elapsed


def evaluate_model(model, X_test, y_test, le):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        pass  # classification_report below
    print("\n" + "=" * 60)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"Macro F1: {macro_f1:.4f}")
    print("=" * 60)
    return macro_f1


def save_models(models_dict: dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models_dict.items():
        if name.endswith(".h5") or isinstance(model, np.ndarray):
            continue
        joblib.dump(model, MODELS_DIR / name)
        print(f"Saved {MODELS_DIR / name}")


def main():
    from .data_utils import apply_smote, build_sequences, prepare_data

    # Full tabular pipeline (no LSTM sequences yet)
    X_train, X_test, y_train, y_test, le, preprocessor, _features = prepare_data()

    # Rebalance using SMOTE for tree models
    X_bal, y_bal = apply_smote(X_train, y_train)
    print(f"\nAfter SMOTE: {X_bal.shape[0]} samples")

    # --- Random Forest ---
    rf, t_rf = train_random_forest(X_bal, y_bal)
    evaluate_model(rf, X_test, y_test, le)

    # --- XGBoost ---
    xgb, t_xgb = train_xgboost(X_bal, y_bal)
    evaluate_model(xgb, X_test, y_test, le)

    # --- LSTM (sequences from the non-SMOTE train set, note: for speed we sample) ---
    seq_len = 10
    X_seq, y_seq = build_sequences(X_train, y_train, seq_len)
    split = int(0.85 * len(X_seq))
    X_train_seq, X_val_seq = X_seq[:split], X_seq[split:]
    y_train_seq, y_val_seq = y_seq[:split], y_seq[split:]

    lstm, t_lstm = train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, n_classes=len(le.classes_))
    # Evaluate LSTM on sequence-shaped test set
    X_test_seq, y_test_seq = build_sequences(X_test, y_test, seq_len)
    lstm.evaluate(X_test_seq, y_test_seq, verbose=0)

    # Save everything
    save_models({"random_forest.pkl": rf, "xgboost_model.pkl": xgb})
    lstm.save(MODELS_DIR / "lstm_model.h5")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    print("\nAll models saved to", MODELS_DIR)

    # Summary table
    print("\n=== Results Summary ===")
    for name, t in [("Random Forest", t_rf), ("XGBoost", t_xgb), ("LSTM", t_lstm)]:
        print(f"{name}: trained in {t:.1f}s")


if __name__ == "__main__":
    main()

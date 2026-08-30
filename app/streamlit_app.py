"""
Network Anomaly Detection - Streamlit Dashboard

Run:  streamlit run app/streamlit_app.py

Upload a network traffic CSV and the app will:
  - Predict the attack type for each row
  - Display an attack timeline chart
  - Show top features driving the prediction
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(page_title="Network Anomaly Detection", layout="wide")


@st.cache_resource
def get_predictor(model_name: str):
    from src.predict import Predictor

    model_path = MODELS_DIR / model_name
    return Predictor(model_path=model_path)


def available_models() -> list[str]:
    return sorted(
        p.name for p in MODELS_DIR.glob("*.pkl") if p.name != "preprocessor.joblib"
    )


def header():
    st.title("🛡️ Network Anomaly Detection")
    st.caption("ML-based intrusion detection — Random Forest / XGBoost / LSTM + SHAP")


def no_model_warning():
    st.warning(
        "No trained model found in `models/`. Train the models first:\n\n"
        "```bash\npython -m src.train_models\n```"
    )


def sidebar():
    st.sidebar.header("⚙️ Controls")
    model_names = available_models()
    if not model_names:
        return None
    choice = st.sidebar.selectbox("Select model", model_names)
    return choice


def load_uploaded_data(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file, low_memory=False)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not parse file: {e}")
        return None


def show_eda(df: pd.DataFrame):
    st.subheader("📊 Data Overview")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    st.dataframe(df.head(20))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.arange(len(df)), df.select_dtypes(include=[np.number]).iloc[:, 0].fillna(0).values)
    ax.set_title("First numeric feature (sample)")
    st.pyplot(fig)
    plt.close(fig)


def show_predictions(predictor, df: pd.DataFrame):
    st.subheader("🔍 Predictions")
    if "Label" in df.columns:
        st.markdown("> Note: your CSV already contains a `Label` column — predictions below are computed on the flow features.")

    labels = predictor.predict(df)
    df_out = df.copy()
    df_out["Predicted_Label"] = labels

    # Timeline chart
    st.subheader("📈 Attack Timeline")
    class_counts = df_out["Predicted_Label"].value_counts()
    st.bar_chart(class_counts)

    st.dataframe(df_out.head(100))

    # Distribution
    st.subheader("🎯 Predicted Class Distribution")
    st.write(class_counts.to_frame("count"))


def main():
    header()

    col_l, col_r = st.columns([1, 1.6])
    with col_r:
        if available_models():
            st.success(f"Models available: {', '.join(available_models())}")
        else:
            no_model_warning()
        uploaded = st.file_uploader("Upload a network traffic CSV", type=["csv"])

    if uploaded is None:
        st.info("Upload a CICIDS2017-format CSV to get started, or use the notebooks to explore the pipeline.")
        st.stop()

    df = load_uploaded_data(uploaded)
    if df is None:
        st.stop()

    show_eda(df)

    if not available_models():
        st.stop()

    with col_l:
        choice = sidebar()
    if not choice:
        st.stop()

    predictor = get_predictor(choice)
    show_predictions(predictor, df)


if __name__ == "__main__":
    main()

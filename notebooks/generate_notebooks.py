"""Generate the four roadmap notebooks with guidance markdown cells."""

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent


def make_notebook(title, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1)


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}


# ---- 01_eda.ipynb ----
eda = make_notebook(
    "Exploratory Data Analysis",
    [
        md("# 01 - Exploratory Data Analysis (EDA)\n\nCICIDS2017 dataset."),
        md("## Load data\nUse the shared preprocessing utilities."),
        code("import sys\nsys.path.insert(0, '..')\n\nfrom src.data_utils import load_raw_csvs, clean_features\n\nraw = load_raw_csvs()\nX, y = clean_features(raw)\nX.head()"),
        md("## Class distribution\nNormal vs Attack imbalance."),
        code("import matplotlib.pyplot as plt\nimport seaborn as sns\n\ny.value_counts().plot(kind='bar', figsize=(10,4))\nplt.title('Class distribution')\nplt.show()"),
        md("## Missing values & data types"),
        code("print('Nulls:')\nprint(X.isnull().sum().sum())\nprint('\\nData types:')\nprint(X.dtypes.value_counts())"),
        md("## Correlation heatmap\nTop correlated numeric features."),
        code("corr = X.select_dtypes('number').corr().abs()\nplt.figure(figsize=(10,8))\nsns.heatmap(corr, cmap='viridis')\nplt.title('Correlation heatmap')\nplt.show()"),
        md("## Takeaways\nDocument which features best separate normal from malicious traffic here."),
    ],
)

# ---- 02_preprocessing.ipynb ----
preproc = make_notebook(
    "Preprocessing & Feature Engineering",
    [
        md("# 02 - Preprocessing & Feature Engineering"),
        md("## Handle missing values, encode labels, scale features, balance with SMOTE"),
        code("import sys\nsys.path.insert(0, '..')\n\nfrom src.data_utils import prepare_data, apply_smote\n\nX_train, X_test, y_train, y_test, le, pre, feats = prepare_data()\nprint('Train shape:', X_train.shape)\nprint('Classes:', le.classes_)"),
        code("X_bal, y_bal = apply_smote(X_train, y_train)\nprint('After SMOTE:', X_bal.shape)"),
        md("Stratified 70/15/15 split is handled inside `prepare_data`."),
    ],
)

# ---- 03_modeling.ipynb ----
modeling = make_notebook(
    "Model Building & Training",
    [
        md("# 03 - Model Building & Training\n\nCompare Random Forest, XGBoost, and LSTM."),
        code("import sys\nsys.path.insert(0, '..')\n\nfrom src.data_utils import prepare_data, apply_smote, build_sequences\nfrom src.train_models import train_random_forest, train_xgboost, evaluate_model\n\nX_train, X_test, y_train, y_test, le, pre, feats = prepare_data()\nX_bal, y_bal = apply_smote(X_train, y_train)"),
        code("rf, t = train_random_forest(X_bal, y_bal)\nevaluate_model(rf, X_test, y_test, le)"),
        code("xgb, t = train_xgboost(X_bal, y_bal)\nevaluate_model(xgb, X_test, y_test, le)"),
        md("## LSTM on time-windowed sequences"),
        code("from src.data_utils import build_sequences\nfrom src.train_models import train_lstm\n\nseq_len = 10\nX_seq, y_seq = build_sequences(X_train, y_train, seq_len)\nsplit = int(0.85 * len(X_seq))\n\nlstm, t = train_lstm(X_seq[:split], y_seq[:split], X_seq[split:], y_seq[split:], len(le.classes_))"),
        md("Save the best model to `models/`.", ),
    ],
)

# ---- 04_evaluation.ipynb ----
evaluation = make_notebook(
    "Evaluation & Explainability",
    [
        md("# 04 - Evaluation & Explainability\n\nConfusion matrices, classification reports, ROC-AUC, and SHAP."),
        md("## Confusion matrix"),
        code("import joblib\nimport matplotlib.pyplot as plt\nfrom sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n\nmodel = joblib.load('../models/random_forest.pkl')"),
        code("# Compute and plot confusion matrix\ny_pred = model.predict(X_test)"),
        md("## SHAP feature importance"),
        code("# import shap\n# explainer = shap.TreeExplainer(model)\n# shap_values = explainer.shap_values(X_test[:100])\n# shap.summary_plot(shap_values, X_test[:100], feature_names=feats)"),
        md("## Model card\nRecord capabilities, limitations, and false-positive rates."),
    ],
)

files = {
    "01_eda.ipynb": eda,
    "02_preprocessing.ipynb": preproc,
    "03_modeling.ipynb": modeling,
    "04_evaluation.ipynb": evaluation,
}

for name, content in files.items():
    (NB_DIR / name).write_text(content)
    print("Wrote", NB_DIR / name)

# 🛡️ Network Anomaly Detection using Machine Learning

> A beginner-to-intermediate ML project built to demonstrate AI-driven network intrusion detection — aligned with Cisco's security product line (Hypershield, SecureX).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-red?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff69b4?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Why This Project Matters for Cisco](#-why-this-project-matters-for-cisco)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Models Used](#-models-used)
- [How to Run](#-how-to-run)
- [Demo App](#-demo-app)
- [Tech Stack](#-tech-stack)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Problem Statement

Modern networks face constant threats — DDoS attacks, brute-force logins, port scans, and more. Traditional rule-based intrusion detection systems (IDS) fail to catch novel or evolving attack patterns.

This project builds a **machine learning-based anomaly detector** that:

- Classifies network traffic as **Normal** or one of several **Attack types**
- Uses three different ML models to compare accuracy and speed
- Provides **explainable predictions** via SHAP values
- Wraps everything in an **interactive Streamlit dashboard**

---

## 🏢 Why This Project Matters for Cisco

Cisco's security product line — including **Hypershield** and **SecureX** — relies on AI-driven threat detection at network scale. This project directly maps to those products by:

| Project Component | Cisco Product Relevance |
|---|---|
| Multi-class attack detection | Cisco Hypershield (autonomous threat segmentation) |
| Network traffic classification | Cisco SecureX (unified threat visibility) |
| SHAP explainability | Enterprise security audit & compliance needs |
| LSTM time-series modeling | Real-time flow-based anomaly detection |
| Streamlit dashboard | SOC analyst tooling prototype |

---

## 📦 Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017

| Property | Details |
|---|---|
| Source | University of New Brunswick (UNB) |
| Download | https://www.unb.ca/cic/datasets/ids-2017.html |
| Size | ~2.8 GB (multiple CSV files) |
| Features | 80+ network flow features |
| Labels | BENIGN, DDoS, PortScan, BruteForce, Bot, Infiltration, Web Attacks |

**Attack types covered:**

```
✅ DDoS
✅ PortScan
✅ Brute Force (SSH, FTP)
✅ Web Attacks (SQL Injection, XSS)
✅ Bot Traffic
✅ Infiltration
```

> ⚠️ Note: Place raw CSV files in `data/raw/` after downloading. They are excluded from this repo via `.gitignore` due to size.

---

## 📁 Project Structure

```
network-anomaly-detection/
│
├── data/
│   ├── raw/                    # Original CICIDS2017 CSVs (gitignored)
│   └── processed/              # Cleaned, encoded, scaled arrays
│
├── src/
│   ├── data_utils.py           # Load, clean, preprocess, SMOTE, sequences
│   ├── train_models.py         # Train & save RF / XGBoost / LSTM
│   └── predict.py              # Prediction + SHAP helper (Predictor)
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Cleaning, encoding, SMOTE
│   ├── 03_modeling.ipynb       # RF, XGBoost, LSTM training
│   └── 04_evaluation.ipynb     # Metrics, SHAP, ROC-AUC
│
├── models/
│   ├── random_forest.pkl       # Saved Random Forest model (gitignored)
│   ├── xgboost_model.pkl       # Saved XGBoost model (gitignored)
│   └── lstm_model.h5           # Saved LSTM model (gitignored)
│
├── app/
│   └── streamlit_app.py        # Interactive Streamlit dashboard
│
├── reports/
│   └── figures/                # SHAP plots, confusion matrices, ROC curves
│
├── requirements.txt            # All dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🤖 Models Used

### 1. Random Forest Classifier
- Fast, interpretable, handles high-dimensional data well
- Good baseline for tabular network flow data
- Feature importance available natively

### 2. XGBoost Classifier
- Gradient boosting — higher accuracy than Random Forest in most cases
- Handles class imbalance via `scale_pos_weight`
- Industry standard for tabular classification

### 3. LSTM (Long Short-Term Memory)
- Captures **temporal patterns** in sequential network traffic
- Built with TensorFlow/Keras using time-windowed flow sequences
- Most complex model — best suited for catching evolving attack patterns over time

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/KNKDarK/Network-Anomaly-Detection.git
cd Network-Anomaly-Detection
```

### 2. Create and activate virtual environment
```bash
python3 -m venv anomaly-env

# On Windows
anomaly-env\Scripts\activate

# On Mac/Linux
source anomaly-env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Go to https://www.unb.ca/cic/datasets/ids-2017.html and place the CSV files inside `data/raw/`.

### 5. Train the models
```bash
python -m src.train_models
```

### 6. Explore the notebooks
```bash
jupyter notebook
```
Open and run (in order):
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_modeling.ipynb`
4. `notebooks/04_evaluation.ipynb`

---

## 🖥️ Demo App

```bash
streamlit run app/streamlit_app.py
```

Upload a network traffic CSV and the app will:
- Predict attack type for each row
- Display an attack timeline chart
- Show predicted class distribution and a row-level data table

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML Models | scikit-learn, XGBoost |
| Deep Learning | TensorFlow / Keras |
| Explainability | SHAP |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| App / Dashboard | Streamlit |
| Model Saving | joblib, pickle |
| Version Control | Git + GitHub |

Install all at once:

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
tensorflow>=2.12
shap>=0.43
imbalanced-learn>=0.11
matplotlib>=3.7
seaborn>=0.12
streamlit>=1.28
joblib>=1.3
jupyter>=1.0
```

---

## 💡 Key Learnings

- How to handle **severely imbalanced datasets** (99% normal vs 1% attack) using SMOTE and class weights
- How **SHAP values** make black-box models explainable for enterprise security use cases
- Difference between **static classifiers** (RF, XGBoost) and **sequential models** (LSTM) for network data
- How to build an **end-to-end ML pipeline**: raw data → preprocessing → training → evaluation → deployment

---

## 🚀 Future Improvements

- [ ] Add real-time packet capture using Scapy + live prediction pipeline
- [ ] Experiment with Isolation Forest for unsupervised anomaly detection
- [ ] Add a REST API using FastAPI for integration with network monitoring tools
- [ ] Containerize the app with Docker for easy deployment
- [ ] Benchmark against Cisco's open network telemetry datasets

---

## 👨‍💻 Author

**Sk. Shafi Masthan Koushik**
- Email: knkssmk@gmail.com
- GitHub: [@KNKDarK](https://github.com/KNKDarK)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> ⭐ If you found this project useful, please consider giving it a star on GitHub!

# 🔍 MLOps Pipeline — Fraud Detection

![CI/CD](https://github.com/NaimMG/mlops-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![MLflow](https://img.shields.io/badge/MLflow-2.11-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![DVC](https://img.shields.io/badge/DVC-3.43-purple)

End-to-end MLOps pipeline for credit card fraud detection — from raw data to monitored API deployment with automated alerting.

---

## 🎯 Project Overview

This project implements a production-grade MLOps pipeline that detects fraudulent bank transactions. It covers the full ML lifecycle: data versioning, experiment tracking, model optimization, API deployment, CI/CD automation, data drift monitoring and automated email alerts.

**Dataset:** 284,807 real transactions — 492 frauds (0.17%) from OpenML  
**Best model:** Random Forest — ROC AUC: 98.68% | F1: 80.61%

---

## 🏗️ Architecture
```
Raw Data (OpenML)
      ↓
DVC Data Versioning (DagsHub Storage)
      ↓
Preprocessing (pandas/scikit-learn + SMOTE)
      ↓
Experiment Tracking → MLflow (hosted on DagsHub)
      ↓
Hyperparameter Optimization → Optuna (15 trials)
      ↓
Model Comparison → Random Forest vs XGBoost
      ↓
Model Registry → MLflow (Production stage)
      ↓
Best Model → FastAPI REST endpoint
      ↓
Containerization → Docker
      ↓
CI/CD → GitHub Actions (tests + Docker build)
      ↓
Monitoring → Evidently AI (data drift detection)
      ↓
Alerting → Automated email if drift detected
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| MLflow + DagsHub | Experiment tracking & model registry |
| Optuna | Bayesian hyperparameter optimization |
| FastAPI | REST API for model serving |
| Docker | Containerization |
| GitHub Actions | CI/CD automation |
| Evidently AI | Data drift monitoring |
| DVC + DagsHub Storage | Data versioning |
| imbalanced-learn | SMOTE class rebalancing |
| scikit-learn | Random Forest model |
| XGBoost | Alternative model for comparison |

---

## 📊 Model Comparison Results

| Model | Recall | Precision | F1 | ROC AUC |
|-------|--------|-----------|-----|---------|
| Random Forest (baseline) | 80.61% | 80.61% | 80.61% | 98.68% ✅ |
| XGBoost (baseline) | 84.69% | 71.55% | 77.57% | 97.47% |
| Random Forest + SMOTE | 85.71% | 63.64% | 73.04% | 98.39% |
| XGBoost + SMOTE | 87.76% | 32.82% | 47.78% | 97.39% |

**Conclusion:** Random Forest without SMOTE offers the best overall balance with highest ROC AUC.

All experiments tracked on [DagsHub](https://dagshub.com/NaimMG/mlops-pipeline)

---

## ⚙️ Configuration

All parameters are configurable via environment variables in `.env` :
```env
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=models/best_model.pkl
MODEL_NAME=fraud-detection-model
FRAUD_THRESHOLD=0.5
DRIFT_THRESHOLD=0.3
ALERT_EMAIL_SENDER=your@gmail.com
ALERT_EMAIL_PASSWORD=your_app_password
ALERT_EMAIL_RECEIVER=your@gmail.com
```

---

## 🚀 Quick Start

### 1. Clone & setup
```bash
git clone https://github.com/NaimMG/mlops-pipeline.git
cd mlops-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull data with DVC
```bash
dvc pull
```

### 3. Train & optimize model
```bash
python src/data/load_data.py
python src/data/preprocess.py
python src/models/optimize.py
```

### 4. Run API
```bash
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs
```

### 5. Run with Docker
```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

### 6. Run monitoring & alerts
```bash
python src/monitoring/monitor.py
python src/monitoring/alerts.py
open reports/drift_report.html
```

### 7. Run tests
```bash
pytest tests/ -v
```

---

## 📁 Project Structure
```
mlops-pipeline/
├── src/
│   ├── data/
│   │   ├── load_data.py           # Dataset loading from OpenML
│   │   ├── preprocess.py          # Feature engineering & train/test split
│   │   └── preprocess_smote.py    # SMOTE rebalancing
│   ├── models/
│   │   ├── train.py               # Training + MLflow tracking
│   │   ├── optimize.py            # Optuna hyperparameter search
│   │   ├── train_xgboost.py       # XGBoost training & comparison
│   │   ├── train_with_smote.py    # Training with SMOTE data
│   │   └── register_model.py      # MLflow Model Registry
│   └── monitoring/
│       ├── monitor.py             # Evidently drift detection
│       └── alerts.py              # Automated email alerts
├── api/
│   └── main.py                    # FastAPI prediction endpoint
├── tests/
│   ├── test_api.py                # API endpoint tests
│   ├── test_model.py              # Model unit tests
│   └── test_preprocessing.py     # Preprocessing unit tests
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI/CD
├── config.py                      # Centralized configuration
├── Dockerfile
├── docker-compose.yml
├── data.dvc                       # DVC data tracking
└── requirements.txt
```

---

## 🔔 Monitoring & Alerting

The pipeline includes automated drift detection and email alerting:

- **Evidently AI** computes drift scores between training and production data
- If drift exceeds the configured threshold (`DRIFT_THRESHOLD=0.3`), an **email alert** is sent automatically
- Logs are structured with timestamps and severity levels, saved to `logs/pipeline.log`

---

## 🔗 Links

- 📈 **Experiments on DagsHub:** https://dagshub.com/NaimMG/mlops-pipeline
- 🐙 **GitHub:** https://github.com/NaimMG/mlops-pipeline
- 📖 **API Docs:** http://localhost:8000/docs (after running)
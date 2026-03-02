# 🔍 MLOps Pipeline — Fraud Detection

![CI/CD](https://github.com/NaimMG/mlops-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![MLflow](https://img.shields.io/badge/MLflow-2.11-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

End-to-end MLOps pipeline for credit card fraud detection — from raw data to monitored API deployment.

---

## 🎯 Project Overview

This project implements a production-grade MLOps pipeline that detects fraudulent bank transactions. It covers the full ML lifecycle: data versioning, experiment tracking, model optimization, API deployment, CI/CD automation, and data drift monitoring.

**Dataset:** 284,807 real transactions — 492 frauds (0.17%) from OpenML  
**Best model:** Random Forest — ROC AUC: 98.68% | F1: 80.61%

---

## 🏗️ Architecture
```
Raw Data (OpenML)
      ↓
Preprocessing (pandas/scikit-learn)
      ↓
Experiment Tracking → MLflow (hosted on DagsHub)
      ↓
Hyperparameter Optimization → Optuna (15 trials)
      ↓
Best Model → FastAPI REST endpoint
      ↓
Containerization → Docker
      ↓
CI/CD → GitHub Actions (tests + Docker build)
      ↓
Monitoring → Evidently AI (data drift detection)
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
| scikit-learn | Random Forest model |
| DVC | Data versioning (planned) |

---

## 📊 Results

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Accuracy | 99.93% | 99.93% |
| ROC AUC | 97.98% | **98.68%** |
| F1 Score | 80.61% | 80.61% |
| Precision | 80.61% | 80.61% |

All experiments tracked on [DagsHub](https://dagshub.com/NaimMG/mlops-pipeline)

---

## 🚀 Quick Start

### 1. Clone & setup
```bash
git clone https://github.com/NaimMG/mlops-pipeline.git
cd mlops-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare data
```bash
python src/data/load_data.py
python src/data/preprocess.py
```

### 3. Train model
```bash
python src/models/train.py
```

### 4. Optimize hyperparameters
```bash
python src/models/optimize.py
```

### 5. Run API
```bash
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs
```

### 6. Run with Docker
```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

### 7. Generate monitoring report
```bash
python src/monitoring/monitor.py
open reports/drift_report.html
```

---

## 📁 Project Structure
```
mlops-pipeline/
├── src/
│   ├── data/
│   │   ├── load_data.py        # Dataset loading from OpenML
│   │   └── preprocess.py       # Feature engineering & train/test split
│   ├── models/
│   │   ├── train.py            # Training + MLflow tracking
│   │   └── optimize.py         # Optuna hyperparameter search
│   └── monitoring/
│       └── monitor.py          # Evidently drift detection
├── api/
│   └── main.py                 # FastAPI prediction endpoint
├── tests/
│   └── test_api.py             # Automated API tests
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🔗 Links

- 📈 **Experiments on DagsHub:** https://dagshub.com/NaimMG/mlops-pipeline
- 🐙 **GitHub:** https://github.com/NaimMG/mlops-pipeline
- 📖 **API Docs:** http://localhost:8000/docs (after running)
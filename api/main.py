from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, MODEL_NAME, FRAUD_THRESHOLD, MLFLOW_TRACKING_URI

app = FastAPI(
    title="Fraud Detection API",
    description="API de détection de fraude bancaire — MLOps Pipeline",
    version="2.0.0"
)

model = None

def load_model():
    global model

    if MLFLOW_TRACKING_URI:
        try:
            import dagshub
            import mlflow.sklearn
            dagshub.init(repo_owner='NaimMG', repo_name='mlops-pipeline', mlflow=True)
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
            print("✅ Modèle chargé depuis MLflow Registry")
            return
        except Exception as e:
            print(f"⚠️ Registry indisponible : {e}")

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Modèle chargé depuis {MODEL_PATH}")
    else:
        print("❌ Aucun modèle disponible")

load_model()


class Transaction(BaseModel):
    features: list[float]
    model_config = {
        "json_schema_extra": {
            "examples": [{"features": [0.1] * 29}]
        }
    }


class Prediction(BaseModel):
    is_fraud: bool
    probability: float
    risk_level: str


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "fraud_threshold": FRAUD_THRESHOLD
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    if len(transaction.features) != model.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"{model.n_features_in_} features attendues, {len(transaction.features)} reçues"
        )

    features = np.array(transaction.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    # Utilise le seuil configurable
    is_fraud = probability >= FRAUD_THRESHOLD

    if probability < 0.3:
        risk = "LOW"
    elif probability < 0.7:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return Prediction(
        is_fraud=bool(is_fraud),
        probability=round(float(probability), 4),
        risk_level=risk
    )
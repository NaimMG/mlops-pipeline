from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import dagshub
import mlflow.sklearn

app = FastAPI(
    title="Fraud Detection API",
    description="API de détection de fraude bancaire — MLOps Pipeline",
    version="2.0.0"
)

# Connexion DagsHub + chargement depuis le Model Registry
dagshub.init(repo_owner='NaimMG', repo_name='mlops-pipeline', mlflow=True)

MODEL_NAME = "fraud-detection-model"
MODEL_ALIAS = "Production"

try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_ALIAS}")
    print(f"✅ Modèle chargé depuis MLflow Registry ({MODEL_NAME}/{MODEL_ALIAS})")
    print(f"   Features attendues : {model.n_features_in_}")
except Exception as e:
    print(f"⚠️ Impossible de charger depuis Registry : {e}")
    # Fallback sur le fichier local
    import joblib
    if os.path.exists("models/best_model.pkl"):
        model = joblib.load("models/best_model.pkl")
        print("✅ Modèle chargé depuis fichier local (fallback)")
    else:
        model = None
        print("❌ Aucun modèle disponible")


class Transaction(BaseModel):
    features: list[float]

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "features": [0.1] * 29
            }]
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
        "model_source": "MLflow Registry"
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

    if probability < 0.3:
        risk = "LOW"
    elif probability < 0.7:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return Prediction(
        is_fraud=bool(prediction),
        probability=round(float(probability), 4),
        risk_level=risk
    )
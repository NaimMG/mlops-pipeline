from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="Fraud Detection API",
    description="API de détection de fraude bancaire — MLOps Pipeline",
    version="1.0.0"
)

# Chargement du modèle au démarrage
MODEL_PATH = "models/best_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé avec succès")
else:
    model = None
    print("⚠️ Modèle non trouvé")


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
        "model_loaded": model is not None
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

print("Model expects:", model.n_features_in_)
if hasattr(model, 'feature_names_in_'):
    print("Feature names:", model.feature_names_in_)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI(
    title="Fraud Detection API",
    description="API de détection de fraude bancaire — MLOps Pipeline",
    version="2.0.0"
)

model = None

def load_model():
    global model

    if os.getenv("MLFLOW_TRACKING_URI"):
        try:
            import dagshub
            import mlflow.sklearn
            dagshub.init(repo_owner='NaimMG', repo_name='mlops-pipeline', mlflow=True)
            model = mlflow.sklearn.load_model("models:/fraud-detection-model/Production")
            print("✅ Modèle chargé depuis MLflow Registry")
            return
        except Exception as e:
            print(f"⚠️ Registry indisponible : {e}")

    if os.path.exists("models/best_model.pkl"):
        model = joblib.load("models/best_model.pkl")
        print("✅ Modèle chargé depuis fichier local")
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

    risk = "LOW" if probability < 0.3 else "MEDIUM" if probability < 0.7 else "HIGH"

    return Prediction(
        is_fraud=bool(prediction),
        probability=round(float(probability), 4),
        risk_level=risk
    )
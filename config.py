import os
from dotenv import load_dotenv

load_dotenv()

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Modèle
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud-detection-model")
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

# MLflow / DagsHub
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME", "NaimMG")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "mlops-pipeline")

# Monitoring
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))

# Email alertes
ALERT_EMAIL_SENDER = os.getenv("ALERT_EMAIL_SENDER", "")
ALERT_EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")
ALERT_EMAIL_RECEIVER = os.getenv("ALERT_EMAIL_RECEIVER", "")
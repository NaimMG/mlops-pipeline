import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub
import joblib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import get_logger

logger = get_logger("train")

def train_model(n_estimators=100, max_depth=10, random_state=42):
    
    dagshub.init(repo_owner='NaimMG', repo_name='mlops-pipeline', mlflow=True)
    
    logger.info("Chargement des données...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test  = pd.read_csv("data/processed/y_test.csv").values.ravel()

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run():
        logger.info(f"Entraînement : n_estimators={n_estimators}, max_depth={max_depth}")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1_score":  f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_pred_proba)
        }
        
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metrics(metrics)
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        logger.info(f"Accuracy  : {metrics['accuracy']:.4f}")
        logger.info(f"Precision : {metrics['precision']:.4f}")
        logger.info(f"Recall    : {metrics['recall']:.4f}")
        logger.info(f"F1 Score  : {metrics['f1_score']:.4f}")
        logger.info(f"ROC AUC   : {metrics['roc_auc']:.4f}")
        logger.info("Run loggé sur DagsHub avec succès")

        return model, metrics

if __name__ == "__main__":
    train_model()
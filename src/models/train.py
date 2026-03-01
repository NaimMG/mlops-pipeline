import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score)
import mlflow
import mlflow.sklearn
import dagshub
import joblib
import os

def train_model(n_estimators=100, max_depth=10, random_state=42):
    
    # Connexion DagsHub
    dagshub.init(repo_owner='NaimMG', repo_name='mlops-pipeline', mlflow=True)
    
    # Chargement des données
    print("📂 Chargement des données...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test  = pd.read_csv("data/processed/y_test.csv").values.ravel()

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run():

        print(f"🚀 Entraînement : n_estimators={n_estimators}, max_depth={max_depth}")
        
        # Entraînement
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'  # Important car dataset déséquilibré
        )
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred      = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métriques
        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1_score":  f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log paramètres et métriques dans MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metrics(metrics)
        
        # Sauvegarde du modèle
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Affichage des résultats
        print("\n📊 RÉSULTATS :")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1 Score  : {metrics['f1_score']:.4f}")
        print(f"  ROC AUC   : {metrics['roc_auc']:.4f}")
        print("\n✅ Run loggé sur DagsHub !")

        return model, metrics

if __name__ == "__main__":
    train_model()
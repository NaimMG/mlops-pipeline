import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub
import joblib
import os

def train_xgboost(n_estimators=100, max_depth=6, learning_rate=0.1):
    
    dagshub.init(repo_owner='NaimMG', repo_name='mlops-pipeline', mlflow=True)

    print("📂 Chargement des données...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test  = pd.read_csv("data/processed/y_test.csv").values.ravel()

    # Calcul du ratio pour gérer le déséquilibre
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    mlflow.set_experiment("model-comparison")

    with mlflow.start_run(run_name="xgboost-baseline"):

        print(f"🚀 Entraînement XGBoost...")

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,  # Gère le déséquilibre
            random_state=42,
            eval_metric='logloss',
            verbosity=0
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

        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 2))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "xgboost_model")

        print("\n📊 RÉSULTATS XGBoost :")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1 Score  : {metrics['f1_score']:.4f}")
        print(f"  ROC AUC   : {metrics['roc_auc']:.4f}")
        print("\n✅ Run loggé sur DagsHub !")

        joblib.dump(model, "models/xgboost_model.pkl")

        return model, metrics

if __name__ == "__main__":
    train_xgboost()
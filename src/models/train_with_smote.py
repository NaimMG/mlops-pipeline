import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub
import joblib
import os

def train_with_smote():
    dagshub.init(repo_owner='NaimMG', repo_name='mlops-pipeline', mlflow=True)

    print("📂 Chargement des données SMOTE...")
    X_train = pd.read_csv("data/processed_smote/X_train.csv")
    X_test  = pd.read_csv("data/processed_smote/X_test.csv")
    y_train = pd.read_csv("data/processed_smote/y_train.csv").values.ravel()
    y_test  = pd.read_csv("data/processed_smote/y_test.csv").values.ravel()

    mlflow.set_experiment("model-comparison")

    models = {
        "RandomForest-SMOTE": RandomForestClassifier(
            n_estimators=226, max_depth=14,
            min_samples_split=2, min_samples_leaf=2,
            max_features="log2", random_state=42, n_jobs=-1
        ),
        "XGBoost-SMOTE": XGBClassifier(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, random_state=42,
            eval_metric='logloss', verbosity=0
        )
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n🚀 Entraînement {model_name}...")

        with mlflow.start_run(run_name=model_name):
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

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("smote", True)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, model_name)

            results[model_name] = metrics

            print(f"  Recall    : {metrics['recall']:.4f}")
            print(f"  Precision : {metrics['precision']:.4f}")
            print(f"  F1 Score  : {metrics['f1_score']:.4f}")
            print(f"  ROC AUC   : {metrics['roc_auc']:.4f}")

    # Comparaison finale
    print("\n" + "="*55)
    print("📊 COMPARAISON FINALE")
    print("="*55)
    print(f"{'Modèle':<25} {'Recall':>8} {'Precision':>10} {'F1':>8} {'AUC':>8}")
    print("-"*55)
    for name, m in results.items():
        print(f"{name:<25} {m['recall']:>8.4f} {m['precision']:>10.4f} {m['f1_score']:>8.4f} {m['roc_auc']:>8.4f}")

    print("\n✅ Tous les runs loggés sur DagsHub !")

if __name__ == "__main__":
    train_with_smote()
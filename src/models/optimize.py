import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
import joblib
import os

def objective(trial, X_train, X_test, y_train, y_test):

    # Hyperparamètres à optimiser
    params = {
        "n_estimators":  trial.suggest_int("n_estimators", 50, 300),
        "max_depth":     trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":  trial.suggest_categorical("max_features", ["sqrt", "log2"])
    }

    with mlflow.start_run(nested=True):
        model = RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        mlflow.log_params(params)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

    return auc  # Optuna optimise le ROC AUC


def optimize():
    dagshub.init(repo_owner='NaimMG', repo_name='mlops-pipeline', mlflow=True)

    print("📂 Chargement des données...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test  = pd.read_csv("data/processed/y_test.csv").values.ravel()

    mlflow.set_experiment("fraud-detection-optuna")

    with mlflow.start_run(run_name="optuna-optimization"):

        study = optuna.create_study(direction="maximize")

        print("🔍 Lancement de 15 trials Optuna...")
        study.optimize(
            lambda trial: objective(trial, X_train, X_test, y_train, y_test),
            n_trials=15,
            show_progress_bar=True
        )

        # Meilleurs résultats
        best = study.best_params
        print(f"\n🏆 Meilleurs hyperparamètres :")
        for k, v in best.items():
            print(f"  {k}: {v}")
        print(f"\n  ROC AUC : {study.best_value:.4f}")

        # Log des meilleurs params
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_roc_auc", study.best_value)

        # Réentraînement avec les meilleurs params
        best_model = RandomForestClassifier(
            **best,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        best_model.fit(X_train, y_train)

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.pkl")
        mlflow.sklearn.log_model(best_model, "best_model")

        print("\n✅ Meilleur modèle sauvegardé !")


if __name__ == "__main__":
    optimize()
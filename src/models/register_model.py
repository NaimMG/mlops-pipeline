import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, f1_score
from mlflow import MlflowClient

def register_best_model():
    dagshub.init(repo_owner='NaimMG', repo_name='mlops-pipeline', mlflow=True)
    
    client = MlflowClient()
    
    # Récupère le meilleur run de l'expérience fraud-detection-optuna
    experiment = client.get_experiment_by_name("fraud-detection-optuna")
    
    if experiment is None:
        print("❌ Expérience fraud-detection-optuna non trouvée")
        return
    
    # Récupère tous les runs et trouve le meilleur ROC AUC
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1
    )
    
    if not runs:
        print("❌ Aucun run trouvé")
        return
    
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_auc = best_run.data.metrics.get("roc_auc", 0)
    
    print(f"🏆 Meilleur run trouvé : {best_run_id}")
    print(f"   ROC AUC : {best_auc:.4f}")
    
    # Enregistre le modèle dans le Registry
    model_uri = f"runs:/{best_run_id}/random_forest_model"
    
    registered = mlflow.register_model(
        model_uri=model_uri,
        name="fraud-detection-model"
    )
    
    print(f"✅ Modèle enregistré : version {registered.version}")
    
    # Passe le modèle en stage "Production"
    client.transition_model_version_stage(
        name="fraud-detection-model",
        version=registered.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"🚀 Modèle passé en Production !")
    print(f"\nPour charger ce modèle :")
    print(f"  mlflow.sklearn.load_model('models:/fraud-detection-model/Production')")

if __name__ == "__main__":
    register_best_model()
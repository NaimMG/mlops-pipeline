import mlflow
import dagshub

dagshub.init(repo_owner='NaimMG', repo_name='mlops-pipeline', mlflow=True)

mlflow.set_experiment("test-connection")

with mlflow.start_run():
    mlflow.log_param("test_param", 42)
    mlflow.log_metric("test_metric", 0.99)
    print("✅ MLflow connecté à DagsHub avec succès !")
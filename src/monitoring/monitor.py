import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DataDriftTable, DatasetDriftMetric
import os

def generate_monitoring_report():
    print("📊 Génération du rapport de monitoring...")

    # Chargement des données
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    y_test  = pd.read_csv("data/processed/y_test.csv")

    # On simule de la "production data" avec un léger drift
    np.random.seed(42)
    X_production = X_test.copy()
    # On ajoute du bruit sur quelques colonnes pour simuler un drift
    for col in ['V1', 'V2', 'V3', 'Amount']:
        X_production[col] = X_production[col] + np.random.normal(0, 0.5, len(X_production))

    # Données de référence (train) et de production (test avec drift)
    reference_data = X_train.copy()
    reference_data['target'] = y_train.values

    current_data = X_production.copy()
    current_data['target'] = y_test.values

    # Rapport Data Drift
    drift_report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])

    drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    # Sauvegarde du rapport HTML
    os.makedirs("reports", exist_ok=True)
    drift_report.save_html("reports/drift_report.html")

    print("✅ Rapport sauvegardé : reports/drift_report.html")
    print("🌐 Ouvre ce fichier dans ton navigateur pour visualiser le drift !")

if __name__ == "__main__":
    generate_monitoring_report()
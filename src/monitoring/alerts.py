import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import ALERT_EMAIL_SENDER, ALERT_EMAIL_PASSWORD, ALERT_EMAIL_RECEIVER, DRIFT_THRESHOLD
from src.logger import get_logger

logger = get_logger("alerts")

def send_alert_email(subject: str, body: str):
    """Envoie un email d'alerte via Gmail SMTP"""
    
    if not ALERT_EMAIL_SENDER or not ALERT_EMAIL_PASSWORD:
        logger.warning("Credentials email non configurés — alerte non envoyée")
        return False

    try:
        msg = MIMEMultipart()
        msg['From']    = ALERT_EMAIL_SENDER
        msg['To']      = ALERT_EMAIL_RECEIVER
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(ALERT_EMAIL_SENDER, ALERT_EMAIL_PASSWORD)
            server.sendmail(ALERT_EMAIL_SENDER, ALERT_EMAIL_RECEIVER, msg.as_string())

        logger.info(f"Alerte envoyée à {ALERT_EMAIL_RECEIVER}")
        return True

    except Exception as e:
        logger.error(f"Erreur envoi email : {e}")
        return False


def check_and_alert_drift(drift_report_path: str = "reports/drift_report.html"):
    """Vérifie le rapport Evidently et envoie une alerte si drift détecté"""
    
    logger.info("Vérification du drift...")

    # Charge les données pour recalculer le drift
    import pandas as pd
    import numpy as np
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import DatasetDriftMetric
    
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    y_test  = pd.read_csv("data/processed/y_test.csv")

    reference_data = X_train.copy()
    reference_data['target'] = y_train.values

    # Simule du drift en production
    np.random.seed(42)
    current_data = X_test.copy()
    for col in ['V1', 'V2', 'V3', 'Amount']:
        current_data[col] = current_data[col] + np.random.normal(0, 0.5, len(current_data))
    current_data['target'] = y_test.values

    report = Report(metrics=[DatasetDriftMetric()])
    report.run(reference_data=reference_data, current_data=current_data)

    result      = report.as_dict()
    drift_score = result['metrics'][0]['result']['share_of_drifted_columns']
    drift_detected = result['metrics'][0]['result']['dataset_drift']

    logger.info(f"Score de drift : {drift_score:.4f} (seuil : {DRIFT_THRESHOLD})")

    if drift_detected or drift_score >= DRIFT_THRESHOLD:
        logger.warning(f"⚠️ DRIFT DÉTECTÉ ! Score : {drift_score:.4f}")
        
        subject = "🚨 [MLOps Alert] Data Drift Détecté — Fraud Detection"
        body = f"""
        <html>
        <body>
            <h2 style="color: red;">⚠️ Data Drift Détecté</h2>
            <p>Le monitoring Evidently a détecté un drift significatif dans les données de production.</p>
            <table border="1" cellpadding="8">
                <tr><td><b>Score de drift</b></td><td>{drift_score:.4f}</td></tr>
                <tr><td><b>Seuil configuré</b></td><td>{DRIFT_THRESHOLD}</td></tr>
                <tr><td><b>Drift détecté</b></td><td>{'OUI ⚠️' if drift_detected else 'NON ✅'}</td></tr>
            </table>
            <p>👉 <b>Action recommandée :</b> Vérifier les données de production et envisager un réentraînement du modèle.</p>
            <p><i>— MLOps Pipeline Fraud Detection</i></p>
        </body>
        </html>
        """
        send_alert_email(subject, body)
    else:
        logger.info("✅ Pas de drift significatif détecté")

    return drift_score, drift_detected


if __name__ == "__main__":
    score, detected = check_and_alert_drift()
    print(f"\nDrift score : {score:.4f}")
    print(f"Drift détecté : {detected}")
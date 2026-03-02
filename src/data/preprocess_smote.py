import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

def preprocess_with_smote():
    print("⚙️ Preprocessing avec SMOTE...")

    df = pd.read_csv("data/raw/creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"].astype(int)

    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])
    if "Time" in X.columns:
        X = X.drop("Time", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📊 Avant SMOTE — Fraudes train : {y_train.sum()} / {len(y_train)}")

    # Application du SMOTE uniquement sur le train
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"📊 Après SMOTE — Fraudes train : {y_train_resampled.sum()} / {len(y_train_resampled)}")

    # Sauvegarde
    os.makedirs("data/processed_smote", exist_ok=True)
    X_train_resampled.to_csv("data/processed_smote/X_train.csv", index=False)
    X_test.to_csv("data/processed_smote/X_test.csv", index=False)
    pd.Series(y_train_resampled, name="Class").to_csv("data/processed_smote/y_train.csv", index=False)
    y_test.to_csv("data/processed_smote/y_test.csv", index=False)

    print("✅ Données SMOTE sauvegardées dans data/processed_smote/")
    return X_train_resampled, X_test, y_train_resampled, y_test

if __name__ == "__main__":
    preprocess_with_smote()
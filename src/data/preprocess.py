import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess_data():
    print("⚙️ Preprocessing en cours...")
    
    # Chargement
    df = pd.read_csv("data/raw/creditcard.csv")
    
    # La colonne Class est notre cible (0=normal, 1=fraude)
    X = df.drop("Class", axis=1)
    y = df["Class"].astype(int)
    
    # Normalisation de la colonne Amount (les autres colonnes V1-V28 sont déjà normalisées)
    scaler = StandardScaler()
    X["Amount"] = scaler.fit_transform(X[["Amount"]])
    # On supprime Time si elle existe
    if "Time" in X.columns:
        X = X.drop("Time", axis=1)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Sauvegarde
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    
    print(f"✅ Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")
    print(f"📊 Fraudes train : {y_train.sum()} | Fraudes test : {y_test.sum()}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()
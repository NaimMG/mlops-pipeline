import pandas as pd
from sklearn.datasets import fetch_openml
import os

def load_data():
    print("📥 Téléchargement du dataset...")
    
    # Téléchargement depuis OpenML
    data = fetch_openml(name='creditcard', version=1, as_frame=True, parser='auto')
    
    df = data.frame
    
    # Sauvegarde en local
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/creditcard.csv", index=False)
    
    print(f"✅ Dataset sauvegardé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"📊 Fraudes : {df['Class'].value_counts()[1]} ({df['Class'].value_counts(normalize=True)[1]*100:.2f}%)")
    
    return df

if __name__ == "__main__":
    load_data()
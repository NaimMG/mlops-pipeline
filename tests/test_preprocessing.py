import pytest
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_shape():
    """Vérifie que les données preprocessées ont la bonne forme"""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    y_test  = pd.read_csv("data/processed/y_test.csv")

    assert X_train.shape[1] == X_test.shape[1], "Train et test doivent avoir le même nombre de colonnes"
    assert len(X_train) == len(y_train), "X_train et y_train doivent avoir le même nombre de lignes"
    assert len(X_test) == len(y_test), "X_test et y_test doivent avoir le même nombre de lignes"
    assert X_train.shape[0] > X_test.shape[0], "Train doit être plus grand que test"

def test_no_missing_values():
    """Vérifie qu'il n'y a pas de valeurs manquantes"""
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test  = pd.read_csv("data/processed/X_test.csv")

    assert X_train.isnull().sum().sum() == 0, "X_train contient des valeurs manquantes"
    assert X_test.isnull().sum().sum() == 0, "X_test contient des valeurs manquantes"

def test_target_binary():
    """Vérifie que la cible est bien binaire (0 ou 1)"""
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test  = pd.read_csv("data/processed/y_test.csv").values.ravel()

    assert set(np.unique(y_train)).issubset({0, 1}), "y_train doit contenir uniquement 0 et 1"
    assert set(np.unique(y_test)).issubset({0, 1}), "y_test doit contenir uniquement 0 et 1"

def test_fraud_ratio():
    """Vérifie que le ratio de fraudes est cohérent"""
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    fraud_ratio = y_test.sum() / len(y_test)

    assert fraud_ratio < 0.01, f"Ratio de fraudes anormal : {fraud_ratio:.4f}"
    assert fraud_ratio > 0.0, "Aucune fraude dans le test set"
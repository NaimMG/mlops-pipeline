import pytest
import numpy as np
import joblib
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_exists():
    """Vérifie que le modèle existe"""
    assert os.path.exists("models/best_model.pkl"), "Le modèle best_model.pkl n'existe pas"

def test_model_predict():
    """Vérifie que le modèle prédit correctement"""
    model = joblib.load("models/best_model.pkl")
    X = np.random.rand(5, model.n_features_in_)
    predictions = model.predict(X)
    assert len(predictions) == 5
    assert set(predictions).issubset({0, 1})

def test_model_predict_proba():
    """Vérifie que les probabilités sont entre 0 et 1"""
    model = joblib.load("models/best_model.pkl")
    X = np.random.rand(5, model.n_features_in_)
    probas = model.predict_proba(X)
    assert probas.shape == (5, 2)
    assert np.all(probas >= 0) and np.all(probas <= 1)

def test_model_features():
    """Vérifie que le modèle attend le bon nombre de features"""
    model = joblib.load("models/best_model.pkl")
    assert model.n_features_in_ == 29, f"Attendu 29 features, got {model.n_features_in_}"
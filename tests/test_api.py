from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict():
    response = client.post("/predict", json={
        "features": [0.1] * 29
    })
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
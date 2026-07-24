import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi.testclient import TestClient
from api.main import app

def test_pred_pos():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text":"The movie is great!"})
        assert response.status_code==200
        assert "label" in response.json()

def test_emptystring():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text":""})
        assert response.status_code==422

def test_pred_neg():
    with TestClient(app) as client:
        response=client.post("/predict", json={"text":"This was such a horrible movie!"})
        assert response.status_code==200
        assert "label" in response.json()


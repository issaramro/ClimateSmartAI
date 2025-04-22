from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_valid_historical_date():
    response = client.post("/get_features_values_at_date", json={"date": "01-01-2000"})
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

def test_invalid_date_format():
    response = client.post("/get_features_values_at_date", json={"date": "2000/01/01"})
    assert response.status_code == 400

def test_missing_date():
    response = client.post("/get_features_values_at_date", json={})
    assert response.status_code == 422  # missing field
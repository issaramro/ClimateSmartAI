from fastapi.testclient import TestClient
from EEP_interface.app import app

client = TestClient(app)

def test_invalid_date_format():
    response = client.post("/get_agricultural_variables_and_factors", json={"date": "2025/01/01"})
    assert response.status_code == 400
    assert "Date format must be DD-MM-YYYY" in response.text

def test_missing_date():
    response = client.post("/get_agricultural_variables_and_factors", json={})
    assert response.status_code == 422

# Optional: you can add more integration tests once IEPs are running
# def test_valid_historical_date():
#     response = client.post("/get_agricultural_variables_and_factors", json={"date": "01-01-2022"})
#     assert response.status_code == 200
#     assert "Drought condition" in response.json()

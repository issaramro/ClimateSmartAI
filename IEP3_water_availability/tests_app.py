from fastapi.testclient import TestClient
from IEP3_water_availability.app import app

client = TestClient(app)

def test_valid_input():
    response = client.post("/irrigation_need", json={"values": [1.0]*14})
    assert response.status_code == 200
    assert "irrigation" in response.json()
    assert response.json()["irrigation"] in ["No irrigation needed", "Irrigation needed"]

def test_invalid_input_length():
    response = client.post("/irrigation_need", json={"values": [1.0]*10})
    assert response.status_code == 422

def test_invalid_data_type():
    response = client.post("/irrigation_need", json={"values": ["x"]*14})
    assert response.status_code == 422

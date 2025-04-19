from fastapi.testclient import TestClient
from IEP2_drought_assessment.app import app

client = TestClient(app)

def test_valid_input():
    response = client.post("/assess_drought", json={"values": [1.0]*14})
    assert response.status_code == 200
    assert "drought_class" in response.json()
    assert "drought_label" in response.json()

def test_invalid_input_length():
    response = client.post("/assess_drought", json={"values": [1.0]*10})
    assert response.status_code == 422  # Pydantic will catch this

def test_invalid_data_type():
    response = client.post("/assess_drought", json={"values": ["a"]*14})
    assert response.status_code == 422

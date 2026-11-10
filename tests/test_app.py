
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# Test 1: Health API
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# Test 2: Homepage
def test_homepage():
    response = client.get("/")
    assert response.status_code == 200
    assert "Phishing Website Detection" in response.text


# Test 3: Valid prediction
def test_valid_prediction():
    values = [1] * 28

    response = client.post(
        "/predict",
        json={"values": values}
    )

    assert response.status_code == 200
    assert "result" in response.json()


# Test 4: Invalid input
def test_invalid_input():
    values = [1, 0]

    response = client.post(
        "/predict",
        json={"values": values}
    )

    assert response.status_code == 422

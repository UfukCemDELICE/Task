"""Test cases for the FastAPI sentiment analysis application."""
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "IMDB Sentiment API is running" in response.text

def test_predict_positive():
    """Test the predict endpoint with a positive review."""
    response = client.post("/predict", json={"text": "This movie was fantastic! I loved it."})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "positive"
    assert data["confidence"] > 0.5

def test_predict_negative():
    """Test the predict endpoint with a negative review."""
    response = client.post("/predict", json={"text": "This movie was terrible. I hated it."})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "negative"
    assert data["confidence"] > 0.5

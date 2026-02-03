import os
from fastapi.testclient import TestClient
from src.api.server import app

os.environ["SKIP_MODEL_LOAD"] = "true"

client = TestClient(app)

def test_health_check_no_model():
    """Test health check when model isn't loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "model_not_loaded"}

def test_generate_requires_model():
    """Test that generation fails gracefully if model isn't loaded."""
    response = client.post("/generate", json={"prompt": "Hello"})
    # Expect 503 because SKIP_MODEL_LOAD is true
    assert response.status_code == 503
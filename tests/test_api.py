
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch):
    """Create test client with mocked dependencies."""
    # Mock the config loading and service initialization
    mock_config = type("Config", (), {})()
    mock_service = type("Service", (), {})()
    mock_service.analyze_question = lambda question: {
        "question": question,
        "answer": "Test answer",
        "confidence_score": 0.85,
        "timestamp": "2023-01-01T00:00:00",
    }

    monkeypatch.setattr("nopin.api.load_config", lambda x: mock_config)
    monkeypatch.setattr(
        "nopin.api.NoPinocchio.from_config", lambda config: mock_service
    )

    # Import after mocking to avoid real initialization
    from nopin.api import app

    return TestClient(app)


def test_health_endpoint_returns_healthy_status(api_client):
    """Test health check endpoint returns healthy status."""
    response = api_client.get("/health")

    assert response.status_code == 200


def test_health_endpoint_contains_timestamp(api_client):
    """Test health endpoint includes timestamp."""
    response = api_client.get("/health")
    data = response.json()

    assert "timestamp" in data


def test_analyze_endpoint_returns_analysis(api_client):
    """Test analyze endpoint returns confidence analysis."""
    response = api_client.post("/analyze", json={"question": "Test question"})

    assert response.status_code == 200


def test_analyze_endpoint_contains_confidence_score(api_client):
    """Test analyze response includes confidence score."""
    response = api_client.post("/analyze", json={"question": "Test question"})
    data = response.json()

    assert "confidence_score" in data

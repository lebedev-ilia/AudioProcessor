"""
Basic tests for AudioProcessor.
"""
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["service"] == "AudioProcessor"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"
    assert "uptime" in data


def test_process_endpoint_invalid_uri():
    """Test process endpoint with invalid URI."""
    request_data = {
        "video_id": "test_video",
        "audio_uri": "invalid_uri",
        "dataset": "test"
    }
    
    response = client.post("/process", json=request_data)
    assert response.status_code == 400
    
    data = response.json()
    assert "Invalid audio_uri" in data["error"]


def test_process_endpoint_valid_request():
    """Test process endpoint with valid request."""
    request_data = {
        "video_id": "test_video",
        "audio_uri": "s3://bucket/audio.wav",
        "dataset": "test"
    }
    
    response = client.post("/process", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["accepted"] is True
    assert "celery_task_id" in data


def test_task_status_endpoint():
    """Test task status endpoint."""
    task_id = "test_task_123"
    response = client.get(f"/task/{task_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["task_id"] == task_id
    assert "status" in data


def test_list_extractors_endpoint():
    """Test list extractors endpoint."""
    response = client.get("/extractors")
    assert response.status_code == 200
    
    data = response.json()
    assert "extractors" in data
    assert isinstance(data["extractors"], list)
    assert len(data["extractors"]) > 0
    
    # Check that all extractors have required fields
    for extractor in data["extractors"]:
        assert "name" in extractor
        assert "version" in extractor
        assert "description" in extractor
        assert "status" in extractor


def test_docs_endpoint():
    """Test that docs endpoint is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_redoc_endpoint():
    """Test that redoc endpoint is accessible."""
    response = client.get("/redoc")
    assert response.status_code == 200

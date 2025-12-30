"""Tests for the FastAPI inference API.

These tests verify the REST API endpoints work correctly.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_predictor():
    """Create a mock predictor for testing without loading the model."""
    from src.inference.predictor import PredictionResult

    mock = MagicMock()
    mock.device = "cpu"
    mock.class_mapping = {f"gloss_{i}": i for i in range(90)}
    mock.get_glosses.return_value = [f"gloss_{i}" for i in range(90)]

    # Mock prediction result
    mock_result = PredictionResult(
        gloss="test_gloss",
        confidence=0.95,
        class_id=0,
        top_k=[
            ("test_gloss", 0.95, 0),
            ("gloss_1", 0.03, 1),
            ("gloss_2", 0.02, 2),
        ],
        inference_time_ms=5.0,
        landmark_extraction_time_ms=100.0,
    )
    mock.predict_from_video.return_value = mock_result
    mock.predict_from_landmarks.return_value = mock_result
    mock.predict_batch_from_landmarks.return_value = [mock_result, mock_result]

    return mock


@pytest.fixture
def client(mock_predictor):
    """Create a test client with mocked predictor."""
    import src.inference.api as api_module

    # Patch the global predictor
    with patch.object(api_module, "predictor", mock_predictor):
        with TestClient(api_module.app) as client:
            yield client


@pytest.fixture
def real_landmarks() -> np.ndarray:
    """Get real landmarks from the test dataset if available."""
    test_dir = Path("data/test_landmarks")
    if test_dir.exists():
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                for f in class_dir.iterdir():
                    if f.suffix == ".npy":
                        return np.load(f)
    # Return random landmarks if no test data
    return np.random.randn(30, 258).astype(np.float32)


# ============================================================================
# Health Endpoint Tests
# ============================================================================


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_returns_healthy(self, client):
        """Test that health check returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "device" in data
        assert data["num_classes"] == 90

    def test_health_check_without_model(self):
        """Test health check when model is not loaded returns degraded status."""
        import src.inference.api as api_module

        # Save original and set to None
        original = api_module.predictor
        api_module.predictor = None

        # Create client without triggering lifespan (which would reload model)
        client = TestClient(api_module.app, raise_server_exceptions=False)
        try:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            # When predictor is None, status should be degraded
            assert data["model_loaded"] is False
            assert data["status"] == "degraded"
        finally:
            api_module.predictor = original
            client.close()


# ============================================================================
# Glosses Endpoint Tests
# ============================================================================


class TestGlossesEndpoint:
    """Tests for the /glosses endpoint."""

    def test_get_glosses(self, client):
        """Test getting list of glosses."""
        response = client.get("/glosses")
        assert response.status_code == 200
        data = response.json()
        assert "glosses" in data
        assert "count" in data
        assert data["count"] == 90
        assert len(data["glosses"]) == 90

    def test_glosses_are_sorted(self, client):
        """Test that glosses are returned in sorted order."""
        response = client.get("/glosses")
        data = response.json()
        glosses = data["glosses"]
        assert glosses == sorted(glosses)


# ============================================================================
# Predict from Landmarks Tests
# ============================================================================


class TestPredictFromLandmarks:
    """Tests for the /predict/landmarks endpoint."""

    def test_predict_from_landmarks_success(self, client, real_landmarks):
        """Test successful prediction from landmarks."""
        payload = {
            "landmarks": real_landmarks.tolist(),
            "top_k": 5,
        }
        response = client.post("/predict/landmarks", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "gloss" in data
        assert "confidence" in data
        assert "class_id" in data
        assert "top_k" in data
        assert "inference_time_ms" in data
        assert "total_time_ms" in data

    def test_predict_from_landmarks_invalid_shape(self, client):
        """Test error handling for invalid landmark shape."""
        # Wrong number of features (should be 258)
        payload = {
            "landmarks": np.random.randn(30, 100).tolist(),
            "top_k": 5,
        }
        response = client.post("/predict/landmarks", json=payload)
        # Can be 400 or 500 depending on where validation happens
        assert response.status_code in [400, 500]
        assert "258" in response.json()["detail"]

    def test_predict_from_landmarks_invalid_dimensions(self, client):
        """Test error handling for invalid dimensions."""
        # 1D array instead of 2D - pydantic will convert to 2D with 1 row
        payload = {
            "landmarks": np.random.randn(258).tolist(),
            "top_k": 5,
        }
        response = client.post("/predict/landmarks", json=payload)
        # 422 is validation error from pydantic
        assert response.status_code in [400, 422]

    def test_predict_confidence_in_range(self, client, real_landmarks):
        """Test that confidence scores are in valid range."""
        payload = {"landmarks": real_landmarks.tolist()}
        response = client.post("/predict/landmarks", json=payload)
        data = response.json()
        assert 0 <= data["confidence"] <= 1
        for pred in data["top_k"]:
            assert 0 <= pred["confidence"] <= 1


# ============================================================================
# Batch Predict Tests
# ============================================================================


class TestBatchPredict:
    """Tests for the /predict/batch endpoint."""

    def test_batch_predict_success(self, client, real_landmarks):
        """Test successful batch prediction."""
        # Create batch of 3 samples
        batch = np.stack([real_landmarks, real_landmarks, real_landmarks])
        payload = {
            "landmarks_batch": batch.tolist(),
            "top_k": 3,
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        # Note: mock returns 2 results, but we check the structure
        data = response.json()
        assert isinstance(data, list)

    def test_batch_predict_invalid_shape(self, client):
        """Test error handling for invalid batch shape."""
        # 2D instead of 3D - pydantic validation error
        payload = {
            "landmarks_batch": np.random.randn(30, 258).tolist(),
            "top_k": 5,
        }
        response = client.post("/predict/batch", json=payload)
        # 422 is pydantic validation error
        assert response.status_code in [400, 422]


# ============================================================================
# Predict from Video Tests
# ============================================================================


class TestPredictFromVideo:
    """Tests for the /predict endpoint (video upload)."""

    def test_predict_from_video_success(self, client):
        """Test successful prediction from video upload with mock."""
        # The mock predictor handles the actual prediction
        # We just need to test the endpoint accepts video uploads
        # Find a real test video if available
        test_video = Path("data/test")
        video_path = None

        if test_video.exists():
            for class_dir in test_video.iterdir():
                if class_dir.is_dir():
                    for f in class_dir.iterdir():
                        if f.suffix == ".mp4":
                            video_path = f
                            break
                if video_path:
                    break

        if video_path and video_path.exists():
            with open(video_path, "rb") as f:
                response = client.post(
                    "/predict",
                    files={"video": (video_path.name, f, "video/mp4")},
                    params={"top_k": 5},
                )
            assert response.status_code == 200
            data = response.json()
            assert "gloss" in data
            assert "confidence" in data
        else:
            pytest.skip("No test video found")

    def test_predict_invalid_file_type(self, client):
        """Test error handling for invalid file type."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is not a video file")
            tmp_path = f.name

        try:
            with open(tmp_path, "rb") as f:
                response = client.post(
                    "/predict",
                    files={"video": ("test.txt", f, "text/plain")},
                )
            assert response.status_code == 400
            assert "Invalid file type" in response.json()["detail"]
        finally:
            os.unlink(tmp_path)


# ============================================================================
# Root Endpoint Tests
# ============================================================================


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_info(self, client):
        """Test that root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_model_not_loaded_error(self):
        """Test error when model is not loaded."""
        import src.inference.api as api_module

        # Save original and set to None
        original = api_module.predictor
        api_module.predictor = None

        client = TestClient(api_module.app, raise_server_exceptions=False)
        try:
            payload = {"landmarks": np.random.randn(30, 258).tolist()}
            response = client.post("/predict/landmarks", json=payload)
            assert response.status_code == 503
            assert "not loaded" in response.json()["detail"]
        finally:
            api_module.predictor = original
            client.close()

    def test_missing_required_fields(self, client):
        """Test error when required fields are missing."""
        response = client.post("/predict/landmarks", json={})
        assert response.status_code == 422  # Validation error


# ============================================================================
# Integration Tests (require model)
# ============================================================================


@pytest.mark.skipif(
    not Path("models/best.pt").exists(),
    reason="Model checkpoint not found",
)
class TestIntegration:
    """Integration tests that require the actual model."""

    @pytest.fixture
    def real_client(self):
        """Create a test client with real predictor."""
        from src.inference.api import app

        # This will load the actual model
        with TestClient(app) as client:
            yield client

    def test_real_prediction_from_landmarks(self, real_client, real_landmarks):
        """Test prediction with real model and landmarks."""
        payload = {"landmarks": real_landmarks.tolist(), "top_k": 5}
        response = real_client.post("/predict/landmarks", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["confidence"] > 0
        assert len(data["top_k"]) == 5

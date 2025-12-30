"""Inference pipeline for Malaysian Sign Language translation.

This module provides the SignLanguagePredictor class for running
inference on video files or raw landmark data.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.data.preprocessing import (
    LandmarkExtractor,
    extract_landmarks_from_video,
    TOTAL_FEATURES,
)
from src.data.transforms import Normalize
from src.models.lstm import BiLSTMWithAttention, LSTMClassifier, LSTMWithPooling
from src.utils.device import get_device

logger = logging.getLogger(__name__)


# Model registry for loading different architectures
MODEL_REGISTRY = {
    "lstm": LSTMClassifier,
    "bilstm_attention": BiLSTMWithAttention,
    "lstm_pooling": LSTMWithPooling,
}


@dataclass
class PredictionResult:
    """Result of a sign language prediction.

    Attributes:
        gloss: Predicted sign language gloss (word/phrase).
        confidence: Confidence score for the prediction (0-1).
        class_id: Integer class ID.
        top_k: List of top-k predictions with (gloss, confidence, class_id).
        inference_time_ms: Inference time in milliseconds.
        landmark_extraction_time_ms: Landmark extraction time in milliseconds.
    """

    gloss: str
    confidence: float
    class_id: int
    top_k: list[tuple[str, float, int]] = field(default_factory=list)
    inference_time_ms: float = 0.0
    landmark_extraction_time_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "gloss": self.gloss,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "top_k": [
                {"gloss": g, "confidence": c, "class_id": i}
                for g, c, i in self.top_k
            ],
            "inference_time_ms": round(self.inference_time_ms, 2),
            "landmark_extraction_time_ms": round(self.landmark_extraction_time_ms, 2),
            "total_time_ms": round(
                self.inference_time_ms + self.landmark_extraction_time_ms, 2
            ),
        }


class SignLanguagePredictor:
    """Inference pipeline for Malaysian Sign Language translation.

    This class handles loading the model, preprocessing input videos,
    and running inference to predict sign language glosses.
    """

    def __init__(
        self,
        model_path: str | Path,
        class_mapping_path: Optional[str | Path] = None,
        device: Optional[torch.device] = None,
        num_frames: int = 30,
        confidence_threshold: float = 0.5,
    ):
        """Initialize the predictor.

        Args:
            model_path: Path to the trained model checkpoint.
            class_mapping_path: Path to class mapping JSON file.
                If None, will look for class_mapping.json in same directory.
            device: Device to run inference on. If None, auto-detects.
            num_frames: Number of frames to extract from video.
            confidence_threshold: Minimum confidence to return prediction.
        """
        self.model_path = Path(model_path)
        self.device = device or get_device()
        self.num_frames = num_frames
        self.confidence_threshold = confidence_threshold

        # Load class mapping
        if class_mapping_path is None:
            class_mapping_path = self.model_path.parent / "class_mapping.json"
        self.class_mapping = self._load_class_mapping(class_mapping_path)
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Initialize landmark extractor (lazy loading)
        self._landmark_extractor: Optional[LandmarkExtractor] = None

        # Initialize normalizer (same as used during training)
        self._normalize = Normalize()

        logger.info(f"Predictor initialized on device: {self.device}")
        logger.info(f"Model loaded from: {self.model_path}")
        logger.info(f"Number of classes: {len(self.class_mapping)}")

    def _load_class_mapping(self, path: str | Path) -> dict[str, int]:
        """Load class mapping from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Class mapping not found: {path}")

        with open(path) as f:
            return json.load(f)

    def _load_model(self) -> torch.nn.Module:
        """Load the trained model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False,
        )

        # Get model config
        config = checkpoint.get("model_config", {})
        model_type = config.pop("model_type", "bilstm_attention")
        config.pop("num_parameters", None)  # Remove non-constructor args

        # Get model class
        model_cls = MODEL_REGISTRY.get(model_type, BiLSTMWithAttention)

        # Create and load model
        model = model_cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)

        return model

    @property
    def landmark_extractor(self) -> LandmarkExtractor:
        """Lazy-load the landmark extractor."""
        if self._landmark_extractor is None:
            self._landmark_extractor = LandmarkExtractor(
                num_frames=self.num_frames,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return self._landmark_extractor

    def predict_from_video(
        self,
        video_path: str | Path,
        top_k: int = 5,
    ) -> PredictionResult:
        """Predict sign language gloss from a video file.

        Args:
            video_path: Path to the video file.
            top_k: Number of top predictions to return.

        Returns:
            PredictionResult with predicted gloss and confidence.

        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If video cannot be processed.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Extract landmarks
        start_time = time.perf_counter()
        landmarks = self.landmark_extractor.extract(video_path)
        extraction_time = (time.perf_counter() - start_time) * 1000

        # Run inference
        result = self.predict_from_landmarks(landmarks, top_k=top_k)
        result.landmark_extraction_time_ms = extraction_time

        return result

    def predict_from_landmarks(
        self,
        landmarks: np.ndarray,
        top_k: int = 5,
    ) -> PredictionResult:
        """Predict sign language gloss from pre-extracted landmarks.

        Args:
            landmarks: Numpy array of shape (num_frames, 258) or (batch, num_frames, 258).
            top_k: Number of top predictions to return.

        Returns:
            PredictionResult with predicted gloss and confidence.

        Raises:
            ValueError: If landmarks have invalid shape.
        """
        # Validate input shape
        if landmarks.ndim == 2:
            # Single sample: (num_frames, features) -> (1, num_frames, features)
            landmarks = landmarks[np.newaxis, ...]
        elif landmarks.ndim != 3:
            raise ValueError(
                f"Expected landmarks with 2 or 3 dimensions, got {landmarks.ndim}"
            )

        # Validate feature dimension
        if landmarks.shape[-1] != TOTAL_FEATURES:
            raise ValueError(
                f"Expected {TOTAL_FEATURES} features, got {landmarks.shape[-1]}"
            )

        # Apply normalization (same as during training)
        # Normalize each sample independently
        normalized = np.array([self._normalize(sample) for sample in landmarks])

        # Convert to tensor and move to device
        tensor = torch.from_numpy(normalized).float().to(self.device)

        # Run inference
        start_time = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)

        inference_time = (time.perf_counter() - start_time) * 1000

        # Get predictions (take first sample if batch)
        probs = probs[0].cpu().numpy()

        # Get top-k predictions
        top_k_indices = np.argsort(probs)[::-1][:top_k]
        top_k_predictions = [
            (self.idx_to_class[idx], float(probs[idx]), int(idx))
            for idx in top_k_indices
        ]

        # Get best prediction
        best_idx = int(top_k_indices[0])
        best_conf = float(probs[best_idx])
        best_gloss = self.idx_to_class[best_idx]

        # Apply confidence threshold
        if best_conf < self.confidence_threshold:
            best_gloss = "unknown"

        return PredictionResult(
            gloss=best_gloss,
            confidence=best_conf,
            class_id=best_idx,
            top_k=top_k_predictions,
            inference_time_ms=inference_time,
        )

    def predict_batch_from_landmarks(
        self,
        landmarks_batch: np.ndarray,
        top_k: int = 5,
    ) -> list[PredictionResult]:
        """Predict sign language glosses for a batch of landmarks.

        Args:
            landmarks_batch: Numpy array of shape (batch, num_frames, 258).
            top_k: Number of top predictions per sample.

        Returns:
            List of PredictionResults.
        """
        if landmarks_batch.ndim != 3:
            raise ValueError(
                f"Expected 3D batch array, got {landmarks_batch.ndim}D"
            )

        # Apply normalization (same as during training)
        normalized = np.array([self._normalize(sample) for sample in landmarks_batch])

        # Convert to tensor
        tensor = torch.from_numpy(normalized).float().to(self.device)

        # Run inference
        start_time = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)

        inference_time = (time.perf_counter() - start_time) * 1000
        per_sample_time = inference_time / len(landmarks_batch)

        # Process each sample
        results = []
        probs_np = probs.cpu().numpy()

        for i, sample_probs in enumerate(probs_np):
            top_k_indices = np.argsort(sample_probs)[::-1][:top_k]
            top_k_predictions = [
                (self.idx_to_class[idx], float(sample_probs[idx]), int(idx))
                for idx in top_k_indices
            ]

            best_idx = int(top_k_indices[0])
            best_conf = float(sample_probs[best_idx])
            best_gloss = self.idx_to_class[best_idx]

            if best_conf < self.confidence_threshold:
                best_gloss = "unknown"

            results.append(
                PredictionResult(
                    gloss=best_gloss,
                    confidence=best_conf,
                    class_id=best_idx,
                    top_k=top_k_predictions,
                    inference_time_ms=per_sample_time,
                )
            )

        return results

    def get_glosses(self) -> list[str]:
        """Get list of all supported glosses (class names).

        Returns:
            List of gloss names sorted alphabetically.
        """
        return sorted(self.class_mapping.keys())

    def close(self) -> None:
        """Release resources."""
        if self._landmark_extractor is not None:
            self._landmark_extractor.close()
            self._landmark_extractor = None

    def __enter__(self) -> "SignLanguagePredictor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


def load_predictor(
    model_path: str | Path = "models/best.pt",
    device: Optional[torch.device] = None,
) -> SignLanguagePredictor:
    """Convenience function to load a predictor.

    Args:
        model_path: Path to the model checkpoint.
        device: Device to run inference on.

    Returns:
        Initialized SignLanguagePredictor.
    """
    return SignLanguagePredictor(
        model_path=model_path,
        device=device,
    )

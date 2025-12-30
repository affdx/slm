"""Inference module for Malaysian Sign Language translation.

This module provides:
- SignLanguagePredictor: Main inference pipeline
- FastAPI REST endpoints
"""

from src.inference.predictor import (
    SignLanguagePredictor,
    PredictionResult,
    load_predictor,
)

__all__ = [
    "SignLanguagePredictor",
    "PredictionResult",
    "load_predictor",
]

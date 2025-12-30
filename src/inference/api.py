"""FastAPI REST API for Malaysian Sign Language translation.

This module provides REST endpoints for:
- Video-based sign language prediction
- Landmark-based prediction
- Gloss listing
- Health checks

Usage:
    uvicorn src.inference.api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference.predictor import SignLanguagePredictor, PredictionResult

logger = logging.getLogger(__name__)

# Global predictor instance
predictor: Optional[SignLanguagePredictor] = None


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    gloss: str = Field(..., description="Predicted sign language gloss")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    class_id: int = Field(..., ge=0, description="Class ID")
    top_k: list[dict] = Field(
        default_factory=list,
        description="Top-k predictions with gloss, confidence, class_id",
    )
    inference_time_ms: float = Field(..., description="Model inference time in ms")
    landmark_extraction_time_ms: float = Field(
        default=0, description="Landmark extraction time in ms"
    )
    total_time_ms: float = Field(..., description="Total processing time in ms")


class LandmarkPredictionRequest(BaseModel):
    """Request model for landmark-based prediction."""

    landmarks: list[list[float]] = Field(
        ...,
        description="2D array of landmarks: (num_frames, 258 features)",
    )
    top_k: int = Field(default=5, ge=1, le=90, description="Number of top predictions")


class BatchLandmarkPredictionRequest(BaseModel):
    """Request model for batch landmark-based prediction."""

    landmarks_batch: list[list[list[float]]] = Field(
        ...,
        description="3D array: (batch_size, num_frames, 258 features)",
    )
    top_k: int = Field(default=5, ge=1, le=90, description="Number of top predictions")


class GlossesResponse(BaseModel):
    """Response model for glosses list."""

    glosses: list[str] = Field(..., description="List of supported glosses")
    count: int = Field(..., description="Total number of glosses")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Compute device being used")
    num_classes: int = Field(..., description="Number of supported classes")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    detail: str = Field(..., description="Error message")


# ============================================================================
# Lifespan Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load/unload model."""
    global predictor

    # Startup: Load the model
    model_path = os.getenv("MODEL_PATH", "models/best.pt")
    class_mapping_path = os.getenv("CLASS_MAPPING_PATH", "models/class_mapping.json")

    logger.info(f"Loading model from: {model_path}")

    try:
        predictor = SignLanguagePredictor(
            model_path=model_path,
            class_mapping_path=class_mapping_path,
            num_frames=30,
            confidence_threshold=0.5,
        )
        logger.info("Model loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        predictor = None
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None

    yield

    # Shutdown: Clean up resources
    if predictor is not None:
        predictor.close()
        logger.info("Predictor resources released")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Malaysian Sign Language Translation API",
    description=(
        "AI-powered API for translating Malaysian Sign Language (MSL) gestures "
        "to text. Upload a video or provide pre-extracted landmarks to get "
        "predictions with confidence scores."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================


def get_predictor() -> SignLanguagePredictor:
    """Get the global predictor instance."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs.",
        )
    return predictor


def result_to_response(result: PredictionResult) -> PredictionResponse:
    """Convert PredictionResult to API response."""
    return PredictionResponse(
        gloss=result.gloss,
        confidence=result.confidence,
        class_id=result.class_id,
        top_k=result.to_dict()["top_k"],
        inference_time_ms=result.inference_time_ms,
        landmark_extraction_time_ms=result.landmark_extraction_time_ms,
        total_time_ms=result.inference_time_ms + result.landmark_extraction_time_ms,
    )


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs."""
    return {"message": "MSL Translation API", "docs": "/docs"}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
async def health_check():
    """Check if the service is healthy and model is loaded."""
    model_loaded = predictor is not None
    device = str(predictor.device) if predictor else "N/A"
    num_classes = len(predictor.class_mapping) if predictor else 0

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        device=device,
        num_classes=num_classes,
    )


@app.get(
    "/glosses",
    response_model=GlossesResponse,
    tags=["Glosses"],
    summary="Get list of supported glosses",
)
async def get_glosses():
    """Get the list of all supported MSL glosses (sign language words)."""
    pred = get_predictor()
    glosses = pred.get_glosses()
    return GlossesResponse(glosses=glosses, count=len(glosses))


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict sign from video upload",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid video file"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict_from_video(
    video: UploadFile = File(..., description="Video file (MP4, AVI, MOV, etc.)"),
    top_k: int = Query(default=5, ge=1, le=90, description="Number of top predictions"),
):
    """Predict Malaysian Sign Language gloss from an uploaded video.

    The video will be processed through:
    1. Frame extraction (30 frames evenly sampled)
    2. MediaPipe landmark detection
    3. LSTM model inference

    Returns the predicted gloss with confidence score and top-k alternatives.
    """
    pred = get_predictor()

    # Validate file type
    allowed_types = {
        "video/mp4",
        "video/avi",
        "video/quicktime",
        "video/x-msvideo",
        "video/webm",
    }
    content_type = video.content_type or ""
    if content_type and content_type not in allowed_types:
        # Also allow by extension
        ext = Path(video.filename or "").suffix.lower()
        allowed_ext = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
        if ext not in allowed_ext:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {content_type}. Allowed: MP4, AVI, MOV, WebM, MKV",
            )

    # Save uploaded file to temp location
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=Path(video.filename or ".mp4").suffix,
            delete=False,
        ) as tmp:
            content = await video.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Run prediction
        result = pred.predict_from_video(tmp_path, top_k=top_k)
        return result_to_response(result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Clean up temp file
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post(
    "/predict/landmarks",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict sign from pre-extracted landmarks",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid landmarks"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict_from_landmarks(request: LandmarkPredictionRequest):
    """Predict Malaysian Sign Language gloss from pre-extracted landmarks.

    Use this endpoint when you've already extracted MediaPipe landmarks
    on the client side. Expects a 2D array of shape (num_frames, 258).

    Features per frame:
    - Pose landmarks: 33 x 4 (x, y, z, visibility) = 132 features
    - Left hand: 21 x 3 (x, y, z) = 63 features
    - Right hand: 21 x 3 (x, y, z) = 63 features
    - Total: 258 features
    """
    pred = get_predictor()

    try:
        landmarks = np.array(request.landmarks, dtype=np.float32)

        # Validate shape
        if landmarks.ndim != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 2D array, got {landmarks.ndim}D",
            )
        if landmarks.shape[1] != 258:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 258 features per frame, got {landmarks.shape[1]}",
            )

        result = pred.predict_from_landmarks(landmarks, top_k=request.top_k)
        return result_to_response(result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
    tags=["Prediction"],
    summary="Batch predict from multiple landmark sequences",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid landmarks"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
async def predict_batch(request: BatchLandmarkPredictionRequest):
    """Predict MSL glosses for a batch of landmark sequences.

    Use this for efficient batch inference when processing multiple
    sign language clips. Expects a 3D array of shape
    (batch_size, num_frames, 258).
    """
    pred = get_predictor()

    try:
        landmarks_batch = np.array(request.landmarks_batch, dtype=np.float32)

        # Validate shape
        if landmarks_batch.ndim != 3:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 3D array, got {landmarks_batch.ndim}D",
            )
        if landmarks_batch.shape[2] != 258:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 258 features per frame, got {landmarks_batch.shape[2]}",
            )

        results = pred.predict_batch_from_landmarks(
            landmarks_batch, top_k=request.top_k
        )
        return [result_to_response(r) for r in results]

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the server
    uvicorn.run(
        "src.inference.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

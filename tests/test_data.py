"""Unit tests for data loading and preprocessing."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import (
    Compose,
    Normalize,
    TemporalJitter,
    TemporalCrop,
    SpatialNoise,
    SpatialScale,
    RandomFlip,
    DropFrames,
    get_train_transforms,
    get_eval_transforms,
)
from src.data.preprocessing import TOTAL_FEATURES


class TestNormalize:
    """Tests for Normalize transform."""

    def test_normalize_basic(self):
        """Test basic normalization."""
        transform = Normalize()
        landmarks = np.random.randn(30, 258).astype(np.float32)
        landmarks = landmarks * 10 + 5  # Add offset and scale

        normalized = transform(landmarks)

        assert normalized.shape == landmarks.shape
        assert np.abs(normalized.mean()) < 0.1
        assert np.abs(normalized.std() - 1.0) < 0.1

    def test_normalize_zero_variance(self):
        """Test normalization with zero variance input."""
        transform = Normalize()
        landmarks = np.ones((30, 258), dtype=np.float32)

        normalized = transform(landmarks)

        assert normalized.shape == landmarks.shape
        assert not np.isnan(normalized).any()


class TestTemporalJitter:
    """Tests for TemporalJitter transform."""

    def test_jitter_shape(self):
        """Test that jitter preserves shape."""
        transform = TemporalJitter(max_shift=2, p=1.0)
        landmarks = np.random.randn(30, 258).astype(np.float32)

        jittered = transform(landmarks)

        assert jittered.shape == landmarks.shape

    def test_jitter_probability(self):
        """Test that jitter respects probability."""
        transform = TemporalJitter(max_shift=2, p=0.0)
        landmarks = np.random.randn(30, 258).astype(np.float32)

        jittered = transform(landmarks)

        np.testing.assert_array_equal(jittered, landmarks)


class TestTemporalCrop:
    """Tests for TemporalCrop transform."""

    def test_crop_shape(self):
        """Test that crop preserves shape after resizing."""
        transform = TemporalCrop(min_ratio=0.8, p=1.0)
        landmarks = np.random.randn(30, 258).astype(np.float32)

        cropped = transform(landmarks)

        assert cropped.shape == landmarks.shape

    def test_crop_probability(self):
        """Test that crop respects probability."""
        transform = TemporalCrop(min_ratio=0.5, p=0.0)
        landmarks = np.random.randn(30, 258).astype(np.float32)

        cropped = transform(landmarks)

        np.testing.assert_array_equal(cropped, landmarks)


class TestSpatialNoise:
    """Tests for SpatialNoise transform."""

    def test_noise_shape(self):
        """Test that noise preserves shape."""
        transform = SpatialNoise(noise_std=0.01, p=1.0)
        landmarks = np.random.randn(30, 258).astype(np.float32)

        noisy = transform(landmarks)

        assert noisy.shape == landmarks.shape

    def test_noise_is_added(self):
        """Test that noise is actually added."""
        transform = SpatialNoise(noise_std=0.1, p=1.0)
        landmarks = np.zeros((30, 258), dtype=np.float32)

        noisy = transform(landmarks)

        assert not np.allclose(noisy, landmarks)


class TestSpatialScale:
    """Tests for SpatialScale transform."""

    def test_scale_shape(self):
        """Test that scale preserves shape."""
        transform = SpatialScale(scale_range=(0.9, 1.1), p=1.0)
        landmarks = np.random.randn(30, 258).astype(np.float32)

        scaled = transform(landmarks)

        assert scaled.shape == landmarks.shape


class TestRandomFlip:
    """Tests for RandomFlip transform."""

    def test_flip_shape(self):
        """Test that flip preserves shape."""
        transform = RandomFlip(p=1.0)
        landmarks = np.random.randn(30, 258).astype(np.float32)

        flipped = transform(landmarks)

        assert flipped.shape == landmarks.shape

    def test_flip_swaps_hands(self):
        """Test that flip swaps left and right hands."""
        transform = RandomFlip(p=1.0)
        landmarks = np.zeros((30, 258), dtype=np.float32)

        # Set distinct values for left and right hands (y and z coordinates only)
        # x coordinates get flipped, so use y index (every 3rd starting at offset 1)
        for i in range(132, 195, 3):
            landmarks[:, i + 1] = 1.0  # Left hand y
        for i in range(195, 258, 3):
            landmarks[:, i + 1] = 2.0  # Right hand y

        flipped = transform(landmarks)

        # After flip, y values should be swapped
        for i in range(132, 195, 3):
            assert np.allclose(flipped[:, i + 1], 2.0)  # Was right hand y
        for i in range(195, 258, 3):
            assert np.allclose(flipped[:, i + 1], 1.0)  # Was left hand y


class TestDropFrames:
    """Tests for DropFrames transform."""

    def test_drop_shape(self):
        """Test that drop preserves shape after interpolation."""
        transform = DropFrames(drop_ratio=0.1, p=1.0)
        landmarks = np.random.randn(30, 258).astype(np.float32)

        dropped = transform(landmarks)

        assert dropped.shape == landmarks.shape


class TestCompose:
    """Tests for Compose transform."""

    def test_compose_multiple(self):
        """Test composing multiple transforms."""
        transforms = [
            SpatialNoise(noise_std=0.01, p=1.0),
            Normalize(),
        ]
        compose = Compose(transforms)

        landmarks = np.random.randn(30, 258).astype(np.float32) * 10

        result = compose(landmarks)

        assert result.shape == landmarks.shape
        # Should be normalized
        assert np.abs(result.mean()) < 0.5

    def test_compose_empty(self):
        """Test composing empty list."""
        compose = Compose([])
        landmarks = np.random.randn(30, 258).astype(np.float32)

        result = compose(landmarks)

        np.testing.assert_array_equal(result, landmarks)


class TestTransformFactories:
    """Tests for transform factory functions."""

    def test_train_transforms(self):
        """Test get_train_transforms factory."""
        transform = get_train_transforms()
        landmarks = np.random.randn(30, 258).astype(np.float32)

        result = transform(landmarks)

        assert result.shape == landmarks.shape

    def test_eval_transforms(self):
        """Test get_eval_transforms factory."""
        transform = get_eval_transforms()
        landmarks = np.random.randn(30, 258).astype(np.float32)

        result = transform(landmarks)

        assert result.shape == landmarks.shape

    def test_eval_is_deterministic(self):
        """Test that eval transforms are deterministic."""
        transform = get_eval_transforms()
        landmarks = np.random.randn(30, 258).astype(np.float32)

        result1 = transform(landmarks.copy())
        result2 = transform(landmarks.copy())

        np.testing.assert_array_equal(result1, result2)


class TestPreprocessingConstants:
    """Tests for preprocessing constants."""

    def test_total_features(self):
        """Test that TOTAL_FEATURES is correct."""
        # 33 pose * 4 + 21 left * 3 + 21 right * 3 = 132 + 63 + 63 = 258
        assert TOTAL_FEATURES == 258

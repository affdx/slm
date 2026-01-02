# AGENTS.md - Malaysian Sign Language (MSL) Translation System

## Project Overview
AI-powered Malaysian Sign Language translation system using deep learning for gesture recognition with a Next.js web application frontend.

---

## Build/Lint/Test Commands

### Python Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install -r requirements-dev.txt
```

### Training Commands
```bash
# Train model (auto-detects MPS/CUDA/CPU)
python src/training/train.py --epochs 50 --batch-size 32

# Train with specific device
python src/training/train.py --device mps    # Apple Silicon
python src/training/train.py --device cuda   # NVIDIA GPU
python src/training/train.py --device cpu    # CPU fallback

# Resume training from checkpoint
python src/training/train.py --resume models/checkpoint.pt
```

### Testing Commands
```bash
# Run all tests
pytest

# Run single test file
pytest tests/test_model.py

# Run single test function
pytest tests/test_model.py::test_forward_pass -v

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run tests matching pattern
pytest -k "test_inference" -v
```

### Linting Commands
```bash
# Run all linters
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/

# Type checking
mypy src/ --ignore-missing-imports
```

### Web Application (Next.js)
```bash
cd web

# Install dependencies
npm install

# Development server
npm run dev

# Build for production
npm run build

# Run production build
npm start

# Lint frontend
npm run lint

# Type check
npm run type-check
```

---

## Hardware Detection Pattern

**IMPORTANT**: This project runs on MacBook M1 Pro. Always prioritize MPS (Metal Performance Shaders) for PyTorch operations. Use this device detection pattern:

```python
import torch

def get_device() -> torch.device:
    """Get the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Usage
device = get_device()
model = model.to(device)
tensor = tensor.to(device)
```

---

## Code Style Guidelines

### Python Standards
- **Version**: Python 3.10+
- **Formatter**: Ruff
- **Linter**: Ruff
- **Type Checker**: mypy

### Import Order
```python
# 1. Standard library
import os
from pathlib import Path

# 2. Third-party packages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# 3. Local modules
from src.models.classifier import SignClassifier
from src.utils.device import get_device
```

### Type Annotations
Always use type hints for function signatures:
```python
def load_model(path: Path, device: torch.device) -> nn.Module:
    """Load a trained model from checkpoint."""
    ...

def predict(video_frames: torch.Tensor) -> tuple[str, float]:
    """Predict sign language gloss from video frames."""
    ...
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Variables | snake_case | `batch_size`, `learning_rate` |
| Functions | snake_case | `load_dataset()`, `train_epoch()` |
| Classes | PascalCase | `SignClassifier`, `VideoDataset` |
| Constants | UPPER_SNAKE | `NUM_CLASSES`, `DEFAULT_LR` |
| Private | _prefix | `_internal_method()` |

### Error Handling
```python
# Use specific exceptions with context
class ModelNotFoundError(Exception):
    """Raised when model checkpoint is not found."""
    pass

def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise ModelNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=get_device())

# Use logging, not print statements
import logging
logger = logging.getLogger(__name__)

logger.info(f"Training on device: {device}")
logger.warning("Dataset size is small, consider augmentation")
logger.error(f"Failed to load video: {video_path}")
```

### Docstrings
Use Google-style docstrings:
```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 50,
) -> dict[str, list[float]]:
    """Train the sign language classification model.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        epochs: Number of training epochs.

    Returns:
        Dictionary containing training history with loss and accuracy.

    Raises:
        ValueError: If epochs is less than 1.
    """
    ...
```

---

## Project Structure
```
slm/
├── data/                    # 90 gloss subfolders (dataset)
├── models/                  # Saved model weights
├── src/
│   ├── training/           # train.py, losses.py
│   ├── inference/          # predict.py, api.py
│   ├── models/             # model architectures
│   ├── data/               # dataset classes, transforms
│   └── utils/              # device.py, logging.py
├── web/                     # Next.js frontend
├── tests/                   # pytest test suites
├── notebooks/              # Jupyter experiments
└── requirements.txt
```

---

## Quick Reference

```bash
# Train model
python src/training/train.py

# Run inference
python src/inference/predict.py --video input.mp4

# Start API server
uvicorn src.inference.api:app --reload

# Run specific test
pytest tests/test_model.py::test_forward_pass -v
```

## IMPORTANT
- refer `CHECKLIST.md` for Item you need to work on
- refer `PRD.md` to have more insight of item you are working on

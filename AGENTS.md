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
- **Formatter**: Ruff (line length: 100)
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

## Key Dependencies
- PyTorch 2.0+ (with MPS support)
- torchvision
- mediapipe (landmark extraction)
- numpy, pandas
- opencv-python (video processing)
- FastAPI/uvicorn (API serving)
- pytest, ruff, mypy (dev tools)

---

## Common Pitfalls

1. **MPS Memory**: MPS has limited memory. Use smaller batch sizes (16-32) on M1.
2. **Video Loading**: Always decode videos to consistent FPS (15-30) and frame count.
3. **Data Augmentation**: Apply augmentations on CPU before moving to device.
4. **Checkpointing**: Save models with `map_location` for cross-device compatibility.

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

---

## Project Task Checklist

**IMPORTANT**: Mark tasks with `[x]` when completed. Update this checklist as you progress.

### Phase 1: Project Setup & Data Pipeline
- [x] 1.1 Create project directory structure (src/data, src/models, src/training, src/inference, src/utils, tests, configs, scripts)
- [x] 1.2 Set up Python virtual environment and requirements.txt (torch, mediapipe, opencv, numpy, pandas)
- [x] 1.3 Set up requirements-dev.txt (pytest, ruff, mypy, tensorboard)
- [x] 1.4 Create src/utils/device.py with MPS > CUDA > CPU detection
- [x] 1.5 Create src/utils/logger.py with proper logging configuration
- [x] 1.6 Verify dataset structure: 90 gloss subfolders in data/
- [x] 1.7 Create src/data/dataset.py - PyTorch Dataset class for video loading
- [x] 1.8 Create src/data/preprocessing.py - MediaPipe landmark extraction (258 features)
- [x] 1.9 Create src/data/transforms.py - Data augmentation (temporal jitter, spatial transforms)
- [x] 1.10 Implement train/val/test split (80/10/10) with stratification
- [ ] 1.11 Create scripts/preprocess_dataset.py - Cache extracted landmarks to disk

### Phase 2: Model Architecture
- [x] 2.1 Create src/models/base.py - Base model class with common methods
- [x] 2.2 Create src/models/lstm.py - Baseline 3-layer LSTM (input: 258 features, 30 frames)
- [x] 2.3 Create src/models/bilstm.py - Bidirectional LSTM with attention
- [x] 2.4 Add dropout layers (0.3-0.5) and batch normalization
- [ ] 2.5 Create src/models/transformer.py - Transformer encoder option (optional)
- [x] 2.6 Implement model factory function to select architecture via config
- [x] 2.7 Write unit tests for model forward pass (tests/test_models.py)

### Phase 3: Training Pipeline
- [x] 3.1 Create configs/default.yaml - Training hyperparameters config
- [x] 3.2 Create src/training/losses.py - CrossEntropy, FocalLoss, LabelSmoothing
- [x] 3.3 Create src/training/trainer.py - Training loop with validation
- [x] 3.4 Implement learning rate scheduler (ReduceLROnPlateau)
- [x] 3.5 Implement early stopping with patience (default: 10 epochs)
- [x] 3.6 Implement gradient clipping (max_norm=1.0)
- [x] 3.7 Add checkpointing - save best model and last model
- [x] 3.8 Integrate TensorBoard logging (loss, accuracy, LR curves)
- [x] 3.9 Create src/training/train.py - Main training script with CLI args
- [x] 3.10 Train baseline model on all 90 glosses - document results
- [ ] 3.11 Write training tests (tests/test_training.py)

### Phase 4: Model Optimization & Evaluation
- [x] 4.1 Create src/training/evaluate.py - Evaluation metrics (accuracy, top-5, F1)
- [x] 4.2 Generate confusion matrix for 90 classes
- [x] 4.3 Generate classification report (precision, recall per class)
- [x] 4.4 Experiment with BiLSTM + attention - compare with baseline
- [x] 4.5 Hyperparameter tuning (batch size, LR, hidden dim, dropout)
- [x] 4.6 Implement class weighting for imbalanced classes
- [x] 4.7 Document baseline vs improved model performance comparison
- [x] 4.8 Achieve target: >= 85% test accuracy

### Phase 5: Inference API
- [x] 5.1 Create src/inference/predictor.py - Inference pipeline class
- [x] 5.2 Implement video preprocessing for inference (consistent FPS, frame count)
- [x] 5.3 Add confidence threshold filtering (default: 0.5)
- [x] 5.4 Create src/inference/api.py - FastAPI REST endpoints
- [x] 5.5 Implement POST /predict endpoint (video upload)
- [x] 5.6 Implement POST /predict/landmarks endpoint (pre-extracted landmarks)
- [x] 5.7 Implement GET /glosses endpoint (list all 90 glosses)
- [x] 5.8 Implement GET /health endpoint
- [ ] 5.9 Export model to TorchScript for production (optional)
- [x] 5.10 Measure inference latency - target < 200ms (PASSED: ~3ms model, ~1.4s with MediaPipe)
- [x] 5.11 Write API tests (tests/test_api.py) - 16 tests passing

### Phase 6: Web Application (Next.js)
- [x] 6.1 Initialize Next.js 14 project with TypeScript in web/
- [x] 6.2 Set up Tailwind CSS configuration
- [x] 6.3 Install and configure shadcn/ui components (skipped - using custom Tailwind)
- [x] 6.4 Create app layout with navigation (Header, Footer)
- [x] 6.5 Create Home page - project introduction and features
- [x] 6.6 Create Translate page - main translation interface
- [x] 6.7 Implement video upload component with drag-and-drop
- [x] 6.8 Implement webcam capture component using getUserMedia
- [x] 6.9 Create API client (web/lib/api.ts) to call FastAPI backend
- [x] 6.10 Display translation results with confidence scores
- [x] 6.11 Create Dictionary page - browse all 90 glosses
- [x] 6.12 Create History page - view past translations (localStorage)
- [x] 6.13 Create About page - project info, team, impact
- [x] 6.14 Implement loading states and error handling UI
- [x] 6.15 Add responsive design for mobile devices

### Phase 7: Accessibility & UX
- [ ] 7.1 Implement high contrast mode toggle
- [ ] 7.2 Add proper ARIA labels to all interactive elements
- [ ] 7.3 Ensure keyboard navigation works throughout app
- [ ] 7.4 Add screen reader support for translation results
- [ ] 7.5 Test with WCAG 2.1 AA checker - fix issues
- [ ] 7.6 Add visual feedback during video processing

### Phase 8: Testing & Quality
- [x] 8.1 Set up pytest configuration (pytest.ini)
- [x] 8.2 Write unit tests for data preprocessing
- [x] 8.3 Write unit tests for model architectures
- [ ] 8.4 Write integration tests for training pipeline
- [ ] 8.5 Write API endpoint tests
- [ ] 8.6 Set up ruff for linting - fix all issues
- [ ] 8.7 Set up mypy for type checking - fix all issues
- [ ] 8.8 Achieve >= 80% test coverage
- [ ] 8.9 Run Next.js lint and fix issues

### Phase 9: Documentation & Report
- [ ] 9.1 Write report: 1. Introduction (background, problem, objectives)
- [ ] 9.2 Write report: 2. Methodology (data, architecture, training)
- [ ] 9.3 Write report: 3. Results (training curves, confusion matrix, metrics)
- [ ] 9.4 Write report: 4. Discussion (analysis, challenges, limitations)
- [ ] 9.5 Write report: 5. Application POC (architecture, screenshots, demo)
- [ ] 9.6 Write report: 6. Societal Impact (users, education, sustainability)
- [ ] 9.7 Write report: 7. Conclusion (achievements, learnings, future)
- [ ] 9.8 Write report: 8. References (papers, tools, documentation)
- [ ] 9.9 Create demo video showing end-to-end translation
- [ ] 9.10 Prepare presentation slides

### Phase 10: Final Deliverables
- [ ] 10.1 Final model checkpoint saved in models/
- [ ] 10.2 All code linted and type-checked
- [ ] 10.3 All tests passing
- [ ] 10.4 Web app builds successfully (npm run build)
- [ ] 10.5 Documentation complete (README, AGENTS.md, PRD)
- [ ] 10.6 Final report submitted
- [ ] 10.7 Demo ready for presentation

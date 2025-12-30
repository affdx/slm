# MSL Translator

AI-powered Malaysian Sign Language (Bahasa Isyarat Malaysia) translation system that converts sign language gestures into text.

## Features

- **90 Sign Glosses** - Comprehensive vocabulary covering common Malaysian Sign Language words
- **Real-time Translation** - Upload videos or use webcam to translate signs
- **High Accuracy** - LSTM-based deep learning model with MediaPipe landmark extraction
- **REST API** - FastAPI backend for easy integration
- **Modern Web UI** - Next.js 14 frontend with responsive design

## Demo

![MSL Translator Demo](docs/demo.gif)

## Project Structure

```
slm/
├── src/
│   ├── inference/      # API and prediction pipeline
│   ├── models/         # LSTM model architecture
│   ├── training/       # Training scripts and utilities
│   └── utils/          # Device detection, logging
├── web/                # Next.js frontend
├── models/             # Trained model weights
├── tests/              # Pytest test suites
└── configs/            # Training configuration
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Webcam (optional, for live translation)

### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn src.inference.api:app --reload
```

API will be available at http://localhost:8000

### Frontend Setup

```bash
cd web

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at http://localhost:3000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict sign from video upload |
| `/predict/landmarks` | POST | Predict from pre-extracted landmarks |
| `/glosses` | GET | List all 90 supported glosses |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |

### Example Usage

```bash
# Predict from video file
curl -X POST "http://localhost:8000/predict" \
  -F "video=@sign_video.mp4"

# Response
{
  "gloss": "terima_kasih",
  "confidence": 0.95,
  "top_k": [...]
}
```

## Model Architecture

- **Input**: 30 frames of MediaPipe landmarks (258 features per frame)
  - Pose: 33 landmarks × 4 values (x, y, z, visibility)
  - Hands: 21 landmarks × 3 values (x, y, z) × 2 hands
- **Model**: 3-layer Bidirectional LSTM with attention
- **Output**: 90 sign language glosses

## Training

```bash
# Train model (auto-detects MPS/CUDA/CPU)
python src/training/train.py --epochs 50 --batch-size 32

# Train on specific device
python src/training/train.py --device mps  # Apple Silicon
```

Training logs are saved to `runs/` for TensorBoard visualization:

```bash
tensorboard --logdir runs
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## Supported Glosses

The model recognizes 90 Malaysian Sign Language glosses including:

| Category | Examples |
|----------|----------|
| Greetings | assalamualaikum, hi, apa_khabar |
| Family | ayah, emak, abang, kakak, anak_lelaki |
| Questions | apa, siapa, bila, mana, bagaimana |
| Actions | makan, minum, pergi, baca, tidur |
| Objects | bola, pen, payung, kereta, bas |

See full list at `/glosses` endpoint or Dictionary page.

## Tech Stack

**Backend**
- Python 3.10+
- PyTorch (with MPS/CUDA support)
- MediaPipe (landmark extraction)
- FastAPI (REST API)

**Frontend**
- Next.js 14
- TypeScript
- Tailwind CSS

## Hardware Support

The system automatically detects and uses the best available hardware:

| Device | Priority | Use Case |
|--------|----------|----------|
| MPS | 1st | Apple Silicon (M1/M2/M3) |
| CUDA | 2nd | NVIDIA GPUs |
| CPU | 3rd | Fallback |

## License

MIT License

## Acknowledgments

- Malaysian Federation of the Deaf (MFD)
- MediaPipe team at Google
- PyTorch team

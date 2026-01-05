# Isyarat

> **isyarat.tech** — AI-powered Malaysian Sign Language (Bahasa Isyarat Malaysia) translation system that converts sign language gestures into text.

## Features

- **90 Sign Glosses** - Comprehensive vocabulary covering common Malaysian Sign Language words
- **Real-time Translation** - Live webcam detection with continuous inference
- **Fully Client-side** - All processing runs in the browser (no server required)
- **High Accuracy** - BiLSTM model with attention achieving 92.75% accuracy
- **Modern Web UI** - Next.js 14 frontend with responsive design

## Architecture

The system uses **client-side inference** for privacy and low latency:

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   Webcam    │───▶│  MediaPipe  │───▶│   ONNX Runtime  │  │
│  │   Video     │    │  Landmarks  │    │   (BiLSTM)      │  │
│  └─────────────┘    └─────────────┘    └────────┬────────┘  │
│                                                  │           │
│                                         ┌────────▼────────┐  │
│                                         │   Prediction    │  │
│                                         │   (90 glosses)  │  │
│                                         └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

- **MediaPipe Tasks Vision** - Extracts 258 landmarks per frame (pose + hands)
- **ONNX Runtime Web** - Runs BiLSTM model inference in WebAssembly
- **No Backend Required** - Models are downloaded once and cached

## Demo

Visit the [live demo](https://isyarat.tech) or run locally.

## Quick Start

### Prerequisites

- Node.js 18+
- Modern browser (Chrome, Firefox, Edge, Safari)
- Webcam (for real-time translation)

### Running Locally

```bash
cd web

# Install dependencies
npm install

# Start development server
npm run dev
```

Open http://localhost:3000 in your browser.

## Translation Modes

### 1. Upload Video
Upload a pre-recorded video file for translation.

### 2. Record Webcam
Record a short video clip, then submit for translation.

### 3. Real-time Detection
Continuous live detection with:
- **Movement Detection** - Only processes when hands are moving
- **Hand Detection** - Requires visible hands before processing
- **Prediction Stability** - Shows result only after 3 consistent predictions
- **High Confidence Threshold** - 70% minimum confidence required

## Project Structure

```
slm/
├── web/                    # Next.js frontend (main application)
│   ├── src/
│   │   ├── app/           # Next.js app router pages
│   │   ├── components/    # React components
│   │   ├── hooks/         # Custom React hooks
│   │   └── lib/           # Inference and utility libraries
│   └── public/            # ONNX models and static assets
├── src/                    # Python training code
│   ├── inference/         # Prediction pipeline
│   ├── models/            # LSTM model architecture
│   ├── training/          # Training scripts
│   └── utils/             # Device detection, logging
├── models/                 # Trained PyTorch weights
├── scripts/               # Export and utility scripts
└── tests/                 # Pytest test suites
```

## Model Architecture

- **Input**: 30 frames of MediaPipe landmarks (258 features per frame)
  - Pose: 33 landmarks × 4 values (x, y, z, visibility) = 132
  - Left Hand: 21 landmarks × 3 values (x, y, z) = 63
  - Right Hand: 21 landmarks × 3 values (x, y, z) = 63
- **Model**: 2-layer Bidirectional LSTM with scalar attention
- **Output**: 90 sign language glosses
- **Parameters**: ~969K

### Available Models

The app includes two models you can switch between:

| Model | Description | Accuracy | Best For |
|-------|-------------|----------|----------|
| **Baseline BiLSTM** | Original training with standard normalization | 92.75% | General use |
| **Improved BiLSTM** | Optimized training pipeline with per-feature normalization | 92.75% | Real-time detection |

Both models use the same architecture (BetterLSTM) but differ in training:

**Baseline Model:**
- Standard z-score normalization across all features
- Trained on original NPY dataset
- Good for general video translation

**Improved Model:**
- Per-feature normalization (each of 258 features normalized independently)
- Gaussian noise augmentation during training
- Better handles variations in webcam input
- Recommended for real-time webcam detection

## Training (Optional)

If you want to train your own model:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train model (auto-detects MPS/CUDA/CPU)
python src/training/train.py --epochs 50 --batch-size 32

# Export to ONNX for web deployment
python scripts/export_onnx.py
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

See full list on the Dictionary page.

## Tech Stack

**Frontend**
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- ONNX Runtime Web
- MediaPipe Tasks Vision

**Training**
- Python 3.10+
- PyTorch (with MPS/CUDA support)
- MediaPipe

## Browser Compatibility

| Browser | Status | Notes |
|---------|--------|-------|
| Chrome | Fully supported | Best performance with WebGL |
| Firefox | Fully supported | Uses WebAssembly backend |
| Edge | Fully supported | Same as Chrome |
| Safari | Supported | May need CPU fallback |

## Hardware Support (Training)

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
- ONNX Runtime team at Microsoft
- PyTorch team

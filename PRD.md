# Product Requirements Document (PRD)
# Malaysian Sign Language (MSL) Translation System

## 1. Overview

### 1.1 Product Vision
An AI-powered Malaysian Sign Language translation system that enables seamless communication between the deaf/hard-of-hearing community and hearing individuals through real-time sign language recognition and translation.

### 1.2 Problem Statement
Malaysian Sign Language (MSL) translation research and applications remain in early stages due to:
- Limited datasets and linguistic resources
- Low-resource neural translation models
- Lack of accessible translation tools for the Malaysian deaf community

### 1.3 Target Users
- Deaf and hard-of-hearing individuals in Malaysia
- Family members and friends of deaf individuals
- Healthcare providers and public service workers
- Educational institutions
- Businesses seeking inclusive communication

---

## 2. Technical Requirements

### 2.1 Deep Learning Model

#### Baseline Approach (from reference implementation)
The reference implementation uses:
- **Feature Extraction**: MediaPipe Holistic for landmark detection
  - Pose landmarks: 33 points x 4 values (x, y, z, visibility) = 132 features
  - Left hand: 21 points x 3 values = 63 features
  - Right hand: 21 points x 3 values = 63 features
  - Total: 258 features per frame
- **Sequence Length**: 30 frames per video
- **Architecture**: 3-layer stacked LSTM + Dense layers
- **Training**: Basic cross-entropy loss with Adam optimizer

#### Modern Improvements to Implement
1. **Architecture Enhancements**:
   - Replace stacked LSTM with Bidirectional LSTM or Transformer encoder
   - Add attention mechanisms for temporal modeling
   - Implement dropout and batch normalization
   - Consider CNN-LSTM hybrid for spatial-temporal features

2. **Training Best Practices**:
   - Learning rate scheduling (ReduceLROnPlateau, CosineAnnealing)
   - Early stopping with patience
   - Gradient clipping for stability
   - Mixed precision training (where supported)
   - Proper train/val/test splits (80/10/10)

3. **Data Pipeline**:
   - Data augmentation (temporal jittering, spatial transforms)
   - Balanced sampling for class imbalance
   - Caching preprocessed landmarks for faster training

4. **Experiment Tracking**:
   - Use Weights & Biases or TensorBoard
   - Log metrics, hyperparameters, and model artifacts
   - Version control for datasets and models

#### Dataset
- 90 glosses (sign language vocabulary classes)
- Video-based dataset organized in 90 subfolders
- Training/validation/test split: 80/10/10

#### Performance Targets
| Metric | Target |
|--------|--------|
| Test Accuracy | >= 85% |
| Top-5 Accuracy | >= 95% |
| Inference Latency | < 200ms per sequence |
| Model Size | < 50MB (for web deployment) |

#### Hardware Requirements
- **Primary**: Apple Silicon (M1/M2/M3) with MPS acceleration
- **Secondary**: NVIDIA GPU with CUDA support
- **Fallback**: CPU inference

### 2.2 Web Application (POC)

#### Technology Stack
- **Frontend**: Next.js 14+, TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: Next.js API routes or FastAPI
- **ML Serving**: PyTorch model via ONNX or TorchScript

#### Core Features
1. **Video Upload Translation**: Upload pre-recorded sign language videos
2. **Real-time Camera Translation**: Live webcam-based translation
3. **Translation History**: View past translations
4. **Gloss Dictionary**: Browse all 90 supported signs

#### Accessibility Requirements
- WCAG 2.1 AA compliance
- Mobile-responsive layout
- High contrast mode support
- Screen reader compatibility

---

## 3. Project Structure

```
slm/
├── data/                    # Dataset (90 gloss subfolders)
├── models/                  # Trained model weights & checkpoints
├── src/
│   ├── data/               # Dataset classes, transforms, augmentation
│   ├── models/             # Model architectures (LSTM, Transformer)
│   ├── training/           # Training scripts, losses, schedulers
│   ├── inference/          # Prediction pipeline, API endpoints
│   └── utils/              # Device detection, logging, metrics
├── web/                     # Next.js application
│   ├── app/                # App router pages
│   ├── components/         # React components
│   └── lib/                # API clients, utilities
├── tests/                   # pytest test suites
├── configs/                # Training configs (YAML)
└── scripts/                # Data preprocessing, evaluation scripts
```

---

## 4. Milestones

### Phase 1: Data Pipeline & Model Development (Week 1-2)
- [ ] Implement data loading with proper splits
- [ ] Build landmark extraction pipeline
- [ ] Implement baseline LSTM model with modern training loop
- [ ] Add experiment tracking (W&B or TensorBoard)
- [ ] Train and evaluate on all 90 glosses

### Phase 2: Model Optimization (Week 3)
- [ ] Implement data augmentation
- [ ] Experiment with architecture improvements
- [ ] Hyperparameter tuning
- [ ] Model quantization for deployment

### Phase 3: Inference API (Week 4)
- [ ] Create inference pipeline with preprocessing
- [ ] Build REST API with FastAPI
- [ ] Export model to ONNX/TorchScript
- [ ] Add confidence thresholds and error handling

### Phase 4: Web Application (Week 5)
- [ ] Set up Next.js project with shadcn/ui
- [ ] Implement video upload and camera capture
- [ ] Integrate with inference API
- [ ] Add gloss dictionary and history features

### Phase 5: Testing & Documentation (Week 6)
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Documentation and demo preparation
- [ ] Accessibility audit

---

## 5. Sustainability & Impact

### Revenue Model Options
- B2B licensing to healthcare/education sectors
- Government partnerships for public services
- Freemium model with premium features

### Social Impact
- Improved accessibility for 50,000+ deaf Malaysians
- Educational tool for sign language learning
- Promotes inclusive communication in public spaces

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Limited dataset size | Data augmentation, transfer learning, synthetic data |
| Class imbalance | Weighted loss, oversampling, focal loss |
| Model overfitting | Dropout, early stopping, cross-validation |
| Hardware compatibility | MPS/CUDA/CPU auto-detection |
| Real-time performance | Model quantization, ONNX optimization |
| Poor generalization | Diverse test set, cross-validation |

---

## 7. Report Structure

The final report should follow this structure to meet documentation requirements:

1. **Introduction**
   - Background on MSL and accessibility challenges
   - Problem statement and project objectives
   - Scope and limitations

2. **Methodology**
   - Data collection and preprocessing pipeline
   - Model architecture and design decisions
   - Training procedure and hyperparameters
   - Evaluation metrics

3. **Results**
   - Training curves (loss, accuracy)
   - Confusion matrix and classification report
   - Performance comparison (baseline vs improved)
   - Inference demonstrations

4. **Discussion**
   - Analysis of model performance
   - Challenges encountered and solutions
   - Comparison with existing approaches
   - Limitations and future work

5. **Application POC**
   - System architecture and user flow
   - Screenshots and demo walkthrough
   - Accessibility features implemented
   - User feedback (if available)

6. **Societal Impact**
   - Target user analysis
   - Educational applications
   - Sustainability/revenue model
   - Broader implications for accessibility

7. **Conclusion**
   - Summary of achievements
   - Key learnings
   - Future directions

8. **References**
   - Academic papers, documentation, tools used

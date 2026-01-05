# Adding a New Model to the MSL Translator Frontend

This guide documents how to add a new trained model to the Next.js frontend for browser-based inference.

## Prerequisites

- Trained PyTorch model (`.pt` file with weights)
- Normalization statistics (`.npz` file with `mean` and `std` arrays)
- Model metadata (architecture configuration)
- Same feature extraction as existing models (258-dim MediaPipe landmarks)

## Overview

The frontend uses ONNX Runtime Web to run inference in the browser. To add a new model:

1. Export PyTorch model to ONNX format
2. Convert normalization stats from `.npz` to `.json`
3. Add model configuration to `inference.ts`
4. Verify the model works in the browser

## Step-by-Step Guide

### Step 1: Export Model to ONNX

Create or use an export script that converts your PyTorch model to ONNX.

**For BiLSTM models:**
```bash
python scripts/export_onnx.py \
  --checkpoint models/your_model.pt \
  --norm-stats models/norm_stats.npz \
  --output web/public/model_yourname.onnx
```

**For TCN models:**
```bash
python scripts/export_tcn_onnx.py \
  --weights path/to/weights.pt \
  --meta path/to/meta.json \
  --output web/public/model_tcn.onnx
```

#### Export Script Requirements

Your export script must:

1. **Load the model architecture** - Instantiate the exact same model class used during training
2. **Load trained weights** - Use `model.load_state_dict(torch.load(...))`
3. **Set to eval mode** - Call `model.eval()` before export
4. **Export with correct input shape** - Use `(1, 30, 258)` for batch=1, frames=30, features=258

Example ONNX export:
```python
import torch

model.eval()
dummy_input = torch.randn(1, 30, 258)  # (batch, frames, features)

torch.onnx.export(
    model,
    (dummy_input,),
    "web/public/model_yourname.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["landmarks"],
    output_names=["logits"],
    dynamic_axes={
        "landmarks": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
)
```

### Step 2: Convert Normalization Stats

The frontend expects normalization stats in JSON format with this structure:

```json
{
  "mean": [0.497, 0.386, ...],  // 258 values
  "std": [0.123, 0.456, ...],   // 258 values
  "shape": [1, 1, 258]
}
```

**Conversion script:**
```python
import numpy as np
import json

# Load npz file
norm = np.load("path/to/norm_stats.npz")
mean = norm["mean"].flatten().tolist()
std = norm["std"].flatten().tolist()

# Save as JSON
stats = {
    "mean": mean,
    "std": std,
    "shape": [1, 1, len(mean)],
}

with open("web/public/model_yourname_norm_stats.json", "w") as f:
    json.dump(stats, f)
```

### Step 3: Update Frontend Configuration

Edit `web/src/lib/inference.ts`:

1. **Add to ModelType union:**
```typescript
export type ModelType = "baseline" | "improved" | "tcn" | "yourmodel";
```

2. **Add to MODEL_CONFIGS:**
```typescript
export const MODEL_CONFIGS: Record<ModelType, ModelConfig> = {
  // ... existing models ...
  yourmodel: {
    name: "Your Model Name",
    description: "Brief description of the model",
    modelPath: "/model_yourname.onnx",
    normStatsPath: "/model_yourname_norm_stats.json",
  },
};
```

### Step 4: Verify the Model

1. **Check file placement:**
```bash
ls -la web/public/model_yourname.onnx
ls -la web/public/model_yourname_norm_stats.json
```

2. **Build and test:**
```bash
cd web
npm run build
npm run dev
```

3. **Test in browser:**
   - Open http://localhost:3000
   - Switch to your new model in the model selector
   - Check browser console for errors
   - Test with webcam or video upload

## Model Compatibility Requirements

| Requirement | Value | Notes |
|-------------|-------|-------|
| Input shape | `(batch, 30, 258)` | 30 frames, 258 features per frame |
| Output shape | `(batch, num_classes)` | Logits (not softmax) |
| Feature order | pose + left_hand + right_hand | 132 + 63 + 63 = 258 |
| ONNX opset | 14+ | For WASM compatibility |

### Feature Structure (258 dimensions)

```
Features [0:131]   = Pose landmarks (33 x 4: x, y, z, visibility)
Features [132:194] = Left hand landmarks (21 x 3: x, y, z)
Features [195:257] = Right hand landmarks (21 x 3: x, y, z)
```

This must match the MediaPipe Holistic extraction used in training.

## File Checklist

After adding a new model, you should have:

- [ ] `web/public/model_yourname.onnx` - ONNX model file
- [ ] `web/public/model_yourname_norm_stats.json` - Normalization stats
- [ ] Updated `web/src/lib/inference.ts` - Model configuration
- [ ] Tested in browser - Model loads and runs inference

## Troubleshooting

### Model fails to load
- Check browser console for ONNX errors
- Verify ONNX file is valid: `python -c "import onnx; onnx.checker.check_model('model.onnx')"`
- Ensure opset version is compatible (14 recommended)

### Inference produces wrong results
- Verify normalization stats match training
- Check feature extraction order matches
- Compare PyTorch vs ONNX outputs on same input

### Model is too large
- LSTM models: ~4MB
- TCN models: ~10MB
- Consider quantization for larger models

## Example: Adding TCN Model

The TCN model was added following these exact steps:

```bash
# 1. Export to ONNX
python scripts/export_tcn_onnx.py

# 2. Convert norm stats (done by export script or manually)
python -c "
import numpy as np
import json
norm = np.load('webcam-optimized/norm_stats.npz')
stats = {'mean': norm['mean'].flatten().tolist(), 'std': norm['std'].flatten().tolist(), 'shape': [1,1,258]}
json.dump(stats, open('web/public/model_tcn_norm_stats.json', 'w'))
"

# 3. Update inference.ts (add 'tcn' to ModelType and MODEL_CONFIGS)

# 4. Test
cd web && npm run dev
```

## Architecture-Specific Notes

### BiLSTM (BetterLSTM)
- Uses bidirectional LSTM with attention
- Requires full 30-frame sequence
- ~969K parameters, ~4MB ONNX

### Causal TCN (CausalTCNClassifier)
- Uses 1D temporal convolutions with causal padding
- Strictly causal - only uses past frames (real-time friendly)
- ~2.5M parameters, ~10MB ONNX
- 6 dilation levels (1, 2, 4, 8, 16, 32) cover 30-frame receptive field

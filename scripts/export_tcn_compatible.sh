#!/bin/bash
# Export TCN model with compatible opset version for ONNX Runtime Web 1.14.0

echo "Exporting TCN model with opset version 13 (IR version 8) for ONNX Runtime Web compatibility..."

python scripts/export_onnx.py \
  --checkpoint CV_Live_Model_TCN/best_tcn_causal.pt \
  --meta CV_Live_Model_TCN/best_tcn_causal_meta.json \
  --norm-stats CV_Live_Model_TCN/norm_stats.npz \
  --output web/public/model_tcn.onnx \
  --opset-version 13

echo ""
echo "Converting to single file (no external data)..."

python scripts/fix_onnx_model.py web/public/model_tcn.onnx web/public/model_tcn.onnx

echo ""
echo "✅ Done! TCN model should now be compatible with ONNX Runtime Web 1.14.0"


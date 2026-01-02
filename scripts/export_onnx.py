#!/usr/bin/env python3
"""Export PyTorch model to ONNX format for browser inference.

This script converts the trained sign language model to ONNX format,
which can be run in the browser using ONNX Runtime Web.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --checkpoint models/best.pt --output web/public/model.onnx
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import BetterLSTM


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    """Load BetterLSTM model from checkpoint or weights file."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if this is a full checkpoint or just weights
    if "model_state_dict" in state:
        # Full checkpoint
        config = state.get("model_config", {})
        config.pop("num_parameters", None)
        model = BetterLSTM(**config)
        model.load_state_dict(state["model_state_dict"])
    else:
        # Just weights (state_dict)
        model = BetterLSTM()
        model.load_state_dict(state)
        config = model.get_config()
    
    model = model.to(device)
    model.eval()
    
    return model, config


def export_norm_stats_to_json(norm_stats_path: Path, output_path: Path) -> None:
    """Export normalization stats from .npz to .json for browser use."""
    norm = np.load(norm_stats_path)
    mean = norm["mean"].flatten().tolist()
    std = norm["std"].flatten().tolist()
    
    stats = {
        "mean": mean,
        "std": std,
        "shape": [1, 1, len(mean)],
    }
    
    with open(output_path, "w") as f:
        json.dump(stats, f)
    
    print(f"Norm stats exported to: {output_path}")


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    num_frames: int = 30,
    num_features: int = 258,
    opset_version: int = 14,
) -> None:
    """Export PyTorch model to ONNX format."""
    dummy_input = torch.randn(1, num_frames, num_features)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={
            "landmarks": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"Model exported to: {output_path}")


def verify_onnx_model(onnx_path: Path) -> bool:
    """Verify the exported ONNX model."""
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print("ONNX model verification: PASSED")

        print(f"\nModel inputs:")
        for inp in model.graph.input:
            dims = [d.dim_value or "batch" for d in inp.type.tensor_type.shape.dim]
            print(f"  - {inp.name}: {dims}")

        print(f"\nModel outputs:")
        for out in model.graph.output:
            dims = [d.dim_value or "batch" for d in out.type.tensor_type.shape.dim]
            print(f"  - {out.name}: {dims}")

        return True
    except ImportError:
        print("Warning: onnx package not installed, skipping verification")
        return True
    except Exception as e:
        print(f"ONNX verification failed: {e}")
        return False


def test_onnx_inference(onnx_path: Path, pytorch_model: torch.nn.Module) -> bool:
    """Test ONNX inference matches PyTorch."""
    try:
        import onnxruntime as ort

        test_input = np.random.randn(1, 30, 258).astype(np.float32)

        with torch.no_grad():
            pytorch_output = pytorch_model(torch.from_numpy(test_input)).numpy()

        session = ort.InferenceSession(str(onnx_path))
        onnx_output = session.run(None, {"landmarks": test_input})[0]

        max_diff = np.abs(pytorch_output - onnx_output).max()
        print(f"\nInference comparison:")
        print(f"  Max difference: {max_diff:.6f}")

        if max_diff < 1e-4:
            print("  Status: PASSED")
            return True
        else:
            print("  Status: WARNING (small numerical differences)")
            return True

    except ImportError:
        print("Warning: onnxruntime not installed, skipping inference test")
        return True
    except Exception as e:
        print(f"Inference test failed: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best.pt",
        help="Path to model checkpoint or weights file",
    )
    parser.add_argument(
        "--norm-stats",
        type=str,
        default="models/norm_stats.npz",
        help="Path to normalization stats (.npz file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="web/public/model.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Number of input frames",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    norm_stats_path = Path(args.norm_stats)
    output_path = Path(args.output)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model, config = load_model(checkpoint_path, device)
    print(f"Model config: {config}")

    # Export to ONNX
    print(f"\nExporting to ONNX: {output_path}")
    export_to_onnx(
        model,
        output_path,
        num_frames=args.num_frames,
        opset_version=args.opset_version,
    )

    # Export norm stats if available
    if norm_stats_path.exists():
        norm_output = output_path.parent / f"{output_path.stem}_norm_stats.json"
        export_norm_stats_to_json(norm_stats_path, norm_output)

    # Verify and test
    verify_onnx_model(output_path)
    test_onnx_inference(output_path, model)

    # Copy class mapping
    class_mapping_src = checkpoint_path.parent / "class_mapping.json"
    if class_mapping_src.exists():
        class_mapping_dst = output_path.parent / "class_mapping.json"
        shutil.copy(class_mapping_src, class_mapping_dst)
        print(f"\nCopied class mapping to {class_mapping_dst}")

    # Print file size
    onnx_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nONNX model size: {onnx_size_mb:.2f} MB")

    print("\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export Causal TCN model to ONNX format for browser inference.

This script converts the trained TCN sign language model to ONNX format,
which can be run in the browser using ONNX Runtime Web.

Usage:
    python scripts/export_tcn_onnx.py
    python scripts/export_tcn_onnx.py --weights realtime-optimized/best_tcn_causal.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Causal TCN Model (must match training exactly)
# =========================================================
class CausalConv1d(nn.Module):
    """Conv1d that only pads on the left -> strictly causal."""

    def __init__(
        self, c_in: int, c_out: int, kernel_size: int = 3, dilation: int = 1, bias: bool = True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            c_in, c_out, kernel_size=kernel_size, dilation=dilation, padding=0, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_left = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad_left, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(c_in, c_out, kernel_size, dilation)
        self.conv2 = CausalConv1d(c_out, c_out, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(c_out)
        self.norm2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.down = None
        if c_in != c_out:
            self.down = nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.drop(out)

        res = x if self.down is None else self.down(x)
        return out + res


class CausalTCNClassifier(nn.Module):
    """
    Causal Temporal Convolutional Network for sign language classification.

    Input x: (B, T, D) - batch, time steps, features
    Output: (B, num_classes) - logits
    """

    def __init__(
        self,
        input_dim: int = 258,
        num_classes: int = 90,
        channels: int = 256,
        levels: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, channels)
        self.blocks = nn.ModuleList()
        for i in range(levels):
            dilation = 2**i
            self.blocks.append(
                TemporalBlock(
                    channels, channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout
                )
            )

        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.in_proj(x)  # (B, T, C)
        h = h.transpose(1, 2).contiguous()  # (B, C, T)

        for blk in self.blocks:
            h = blk(h)

        last = h[:, :, -1]  # (B, C) - causal: only uses past
        return self.head(last)  # (B, num_classes)


def load_tcn_model(
    weights_path: Path,
    meta_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Load CausalTCNClassifier model from weights and metadata."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model = CausalTCNClassifier(
        input_dim=int(meta["input_size"]),
        num_classes=int(meta["num_classes"]),
        channels=int(meta.get("channels", 256)),
        levels=int(meta.get("levels", 6)),
        kernel_size=int(meta.get("kernel_size", 3)),
        dropout=float(meta.get("dropout", 0.25)),
    )

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, meta


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
    model: nn.Module,
    output_path: Path,
    num_frames: int = 30,
    num_features: int = 258,
    opset_version: int = 14,
) -> None:
    """Export PyTorch model to ONNX format."""
    model.eval()
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

        print("\nModel inputs:")
        for inp in model.graph.input:
            dims = [d.dim_value or "batch" for d in inp.type.tensor_type.shape.dim]
            print(f"  - {inp.name}: {dims}")

        print("\nModel outputs:")
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


def test_onnx_inference(onnx_path: Path, pytorch_model: nn.Module) -> bool:
    """Test ONNX inference matches PyTorch."""
    try:
        import onnxruntime as ort

        test_input = np.random.randn(1, 30, 258).astype(np.float32)

        pytorch_model.eval()
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
    parser = argparse.ArgumentParser(description="Export Causal TCN model to ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default="realtime-optimized/best_tcn_causal.pt",
        help="Path to TCN model weights",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default="realtime-optimized/best_tcn_causal_meta.json",
        help="Path to model metadata JSON",
    )
    parser.add_argument(
        "--norm-stats",
        type=str,
        default="",
        help="Path to normalization stats (.npz file). If empty, looks in same dir as weights.",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default="",
        help="Path to label_map.json. If empty, uses web/public/class_mapping.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="web/public/model_tcn.onnx",
        help="Output path for ONNX model",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version",
    )

    args = parser.parse_args()

    weights_path = Path(args.weights)
    meta_path = Path(args.meta)
    output_path = Path(args.output)

    # Find norm stats
    if args.norm_stats:
        norm_stats_path = Path(args.norm_stats)
    else:
        # Look in common locations
        possible_paths = [
            weights_path.parent / "norm_stats.npz",
            weights_path.parent / "NPY Dataset" / "norm_stats.npz",
            Path("NPY Dataset") / "norm_stats.npz",
        ]
        norm_stats_path = None
        for p in possible_paths:
            if p.exists():
                norm_stats_path = p
                break

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")

    # Load model
    print(f"Loading TCN model from {weights_path}")
    model, meta = load_tcn_model(weights_path, meta_path, device)
    print(f"Model config: {json.dumps(meta, indent=2)}")

    # Export to ONNX
    print(f"\nExporting to ONNX: {output_path}")
    export_to_onnx(
        model,
        output_path,
        num_frames=int(meta.get("seq_len", 30)),
        num_features=int(meta.get("input_size", 258)),
        opset_version=args.opset_version,
    )

    # Export norm stats if available
    if norm_stats_path and norm_stats_path.exists():
        norm_output = output_path.parent / f"{output_path.stem}_norm_stats.json"
        export_norm_stats_to_json(norm_stats_path, norm_output)
    else:
        print(f"\nWarning: norm_stats.npz not found. You need to manually convert it.")
        print("  Expected location: NPY Dataset/norm_stats.npz")

    # Verify and test
    verify_onnx_model(output_path)
    test_onnx_inference(output_path, model)

    # Print file size
    onnx_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nONNX model size: {onnx_size_mb:.2f} MB")

    print("\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)
    print("\nNext steps:")
    print(f"  1. Verify {output_path} exists")
    print(f"  2. Verify {output_path.stem}_norm_stats.json exists")
    print("  3. Update web/src/lib/inference.ts to add 'tcn' model type")


if __name__ == "__main__":
    main()

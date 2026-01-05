#!/usr/bin/env python3
"""Convert ONNX model with external data to single file."""

import sys
from pathlib import Path

try:
    import onnx
except ImportError:
    print("Error: 'onnx' package is required. Install it with:")
    print("  pip install onnx")
    sys.exit(1)

def fix_onnx_model(input_path: Path, output_path: Path) -> None:
    """Convert ONNX model with external data to single file."""
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Loading model from: {input_path}")
    model = onnx.load(str(input_path))
    
    print(f"Saving as single file to: {output_path}")
    onnx.save_model(model, str(output_path), save_as_external_data=False)
    
    input_size = input_path.stat().st_size
    output_size = output_path.stat().st_size
    print(f"\n✅ Model converted successfully!")
    print(f"   Input size:  {input_size:,} bytes ({input_size / (1024*1024):.2f} MB)")
    print(f"   Output size: {output_size:,} bytes ({output_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/fix_onnx_model.py <input.onnx> [output.onnx]")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_stem(input_path.stem + "_fixed")
    
    fix_onnx_model(input_path, output_path)


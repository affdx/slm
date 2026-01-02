#!/usr/bin/env python3
"""Consolidate individual landmark files into training-ready NPY arrays.

This script takes the individual .npy landmark files (created by preprocess_dataset.py)
and consolidates them into single arrays for efficient training.

Usage:
    python scripts/consolidate_dataset.py
    python scripts/consolidate_dataset.py --data-dir data --output-dir data/npy

Input structure (from preprocess_dataset.py):
    data/
    ├── train_landmarks/
    │   ├── apa_khabar/
    │   │   ├── video1.npy
    │   │   └── video2.npy
    │   └── terima_kasih/
    │       └── ...
    ├── val_landmarks/
    │   └── ...
    └── test_landmarks/
        └── ...

Output structure:
    data/npy/
    ├── X_train.npy      # Shape: (num_samples, 30, 258)
    ├── y_train.npy      # Shape: (num_samples,)
    ├── X_val.npy
    ├── y_val.npy
    ├── X_test.npy
    ├── y_test.npy
    ├── label_map.json   # {"gloss_name": index, ...}
    └── dataset_info.json # Metadata about the dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm


def load_landmarks_from_split(
    landmarks_dir: Path,
    label_map: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    """Load all landmarks from a split directory.
    
    Args:
        landmarks_dir: Path to landmarks directory (e.g., data/train_landmarks)
        label_map: Existing label map to use (for val/test consistency)
        
    Returns:
        Tuple of (X, y, label_map, class_counts)
    """
    X_list = []
    y_list = []
    class_counts: dict[str, int] = {}
    
    # Get all gloss directories
    gloss_dirs = sorted([d for d in landmarks_dir.iterdir() if d.is_dir()])
    
    # Build label map from training data if not provided
    if label_map is None:
        label_map = {d.name: i for i, d in enumerate(gloss_dirs)}
    
    print(f"  Found {len(gloss_dirs)} gloss directories")
    
    for gloss_dir in tqdm(gloss_dirs, desc="Loading"):
        gloss_name = gloss_dir.name
        
        if gloss_name not in label_map:
            print(f"  Warning: {gloss_name} not in label_map, skipping")
            continue
            
        label_idx = label_map[gloss_name]
        
        # Load all .npy files in this gloss directory
        npy_files = sorted(gloss_dir.glob("*.npy"))
        class_counts[gloss_name] = len(npy_files)
        
        for npy_file in npy_files:
            try:
                landmarks = np.load(npy_file)
                
                # Validate shape
                if landmarks.shape != (30, 258):
                    print(f"  Warning: {npy_file} has shape {landmarks.shape}, expected (30, 258)")
                    continue
                    
                X_list.append(landmarks)
                y_list.append(label_idx)
                
            except Exception as e:
                print(f"  Error loading {npy_file}: {e}")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    return X, y, label_map, class_counts


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate landmark files into training-ready NPY arrays"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory containing *_landmarks folders (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for consolidated NPY files (default: data/npy)",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Training landmarks directory (default: {data-dir}/train_landmarks)",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Validation landmarks directory (default: {data-dir}/val_landmarks)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Test landmarks directory (default: {data-dir}/test_landmarks)",
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "npy"
    
    train_dir = Path(args.train_dir) if args.train_dir else data_dir / "train_landmarks"
    val_dir = Path(args.val_dir) if args.val_dir else data_dir / "val_landmarks"
    test_dir = Path(args.test_dir) if args.test_dir else data_dir / "test_landmarks"
    
    print("=" * 60)
    print("Dataset Consolidation")
    print("=" * 60)
    print(f"Train directory: {train_dir}")
    print(f"Val directory:   {val_dir}")
    print(f"Test directory:  {test_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data (this defines the label map)
    print("Loading training data...")
    if not train_dir.exists():
        print(f"Error: Training directory {train_dir} does not exist")
        sys.exit(1)
        
    X_train, y_train, label_map, train_counts = load_landmarks_from_split(train_dir)
    print(f"  Loaded {len(X_train)} training samples")
    
    # Load validation data
    print("\nLoading validation data...")
    if val_dir.exists():
        X_val, y_val, _, val_counts = load_landmarks_from_split(val_dir, label_map)
        print(f"  Loaded {len(X_val)} validation samples")
    else:
        print(f"  Warning: {val_dir} does not exist, skipping")
        X_val, y_val, val_counts = None, None, {}
    
    # Load test data
    print("\nLoading test data...")
    if test_dir.exists():
        X_test, y_test, _, test_counts = load_landmarks_from_split(test_dir, label_map)
        print(f"  Loaded {len(X_test)} test samples")
    else:
        print(f"  Warning: {test_dir} does not exist, skipping")
        X_test, y_test, test_counts = None, None, {}
    
    # Save consolidated arrays
    print("\nSaving consolidated arrays...")
    
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    print(f"  Saved X_train.npy: {X_train.shape}")
    print(f"  Saved y_train.npy: {y_train.shape}")
    
    if X_val is not None and y_val is not None:
        np.save(output_dir / "X_val.npy", X_val)
        np.save(output_dir / "y_val.npy", y_val)
        print(f"  Saved X_val.npy: {X_val.shape}")
        print(f"  Saved y_val.npy: {y_val.shape}")
    
    if X_test is not None and y_test is not None:
        np.save(output_dir / "X_test.npy", X_test)
        np.save(output_dir / "y_test.npy", y_test)
        print(f"  Saved X_test.npy: {X_test.shape}")
        print(f"  Saved y_test.npy: {y_test.shape}")
    
    # Save label map
    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"  Saved label_map.json: {len(label_map)} classes")
    
    # Save dataset info
    dataset_info = {
        "created_at": datetime.now().isoformat(),
        "num_classes": len(label_map),
        "num_frames": 30,
        "num_features": 258,
        "splits": {
            "train": {
                "num_samples": len(X_train),
                "class_counts": train_counts,
            },
        },
    }
    
    if X_val is not None:
        dataset_info["splits"]["val"] = {
            "num_samples": len(X_val),
            "class_counts": val_counts,
        }
    
    if X_test is not None:
        dataset_info["splits"]["test"] = {
            "num_samples": len(X_test),
            "class_counts": test_counts,
        }
    
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"  Saved dataset_info.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Classes: {len(label_map)}")
    print(f"Training samples: {len(X_train)}")
    if X_val is not None:
        print(f"Validation samples: {len(X_val)}")
    if X_test is not None:
        print(f"Test samples: {len(X_test)}")
    
    # Class distribution
    print(f"\nClass distribution (train):")
    min_count = min(train_counts.values())
    max_count = max(train_counts.values())
    print(f"  Min samples: {min_count}")
    print(f"  Max samples: {max_count}")
    print(f"  Imbalance ratio: {max_count / min_count:.1f}x")
    
    print(f"\nOutput saved to: {output_dir}")
    print("\nYou can now train with:")
    print(f"  python src/training/train.py --data-dir {output_dir}")


if __name__ == "__main__":
    main()

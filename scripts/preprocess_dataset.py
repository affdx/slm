#!/usr/bin/env python3
"""Pre-extract MediaPipe landmarks from video dataset.

This script extracts landmarks from all videos and saves them as .npy files
for faster training. Run this once before training.

Usage:
    python scripts/preprocess_dataset.py
    python scripts/preprocess_dataset.py --num-frames 30
    python scripts/preprocess_dataset.py --splits train val test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import LandmarkExtractor


def preprocess_split(
    data_dir: Path,
    output_dir: Path,
    num_frames: int = 30,
    overwrite: bool = False,
) -> dict[str, int]:
    """Preprocess a single data split.

    Args:
        data_dir: Path to input directory with video files.
        output_dir: Path to output directory for landmarks.
        num_frames: Number of frames to extract per video.
        overwrite: Whether to overwrite existing files.

    Returns:
        Dictionary with processing statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"processed": 0, "skipped": 0, "errors": 0}

    # Get all gloss directories
    gloss_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    print(f"\nProcessing {data_dir.name}: {len(gloss_dirs)} glosses")

    # Create extractor
    extractor = LandmarkExtractor(num_frames=num_frames)

    try:
        for gloss_dir in tqdm(gloss_dirs, desc=data_dir.name):
            # Create output directory for this gloss
            gloss_output_dir = output_dir / gloss_dir.name
            gloss_output_dir.mkdir(exist_ok=True)

            # Process each video
            video_files = list(gloss_dir.glob("*.mp4")) + list(gloss_dir.glob("*.avi"))

            for video_file in video_files:
                output_file = gloss_output_dir / f"{video_file.stem}.npy"

                # Skip if already exists and not overwriting
                if output_file.exists() and not overwrite:
                    stats["skipped"] += 1
                    continue

                try:
                    # Extract landmarks
                    landmarks = extractor.extract(video_file)

                    # Save to file
                    np.save(output_file, landmarks)
                    stats["processed"] += 1

                except Exception as e:
                    print(f"\nError processing {video_file}: {e}")
                    stats["errors"] += 1

    finally:
        extractor.close()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract MediaPipe landmarks from video dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {split}_landmarks in data dir)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Number of frames to extract per video (default: 30)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing landmark files",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 60)
    print("MediaPipe Landmark Extraction")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Splits: {args.splits}")
    print(f"Frames per video: {args.num_frames}")
    print(f"Overwrite: {args.overwrite}")

    total_stats = {"processed": 0, "skipped": 0, "errors": 0}

    for split in args.splits:
        split_dir = data_dir / split

        if not split_dir.exists():
            print(f"\nWarning: {split_dir} does not exist, skipping")
            continue

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir) / f"{split}_landmarks"
        else:
            output_dir = data_dir / f"{split}_landmarks"

        # Process this split
        stats = preprocess_split(
            split_dir,
            output_dir,
            num_frames=args.num_frames,
            overwrite=args.overwrite,
        )

        print(f"  Processed: {stats['processed']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Errors: {stats['errors']}")

        for key in total_stats:
            total_stats[key] += stats[key]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total processed: {total_stats['processed']}")
    print(f"Total skipped: {total_stats['skipped']}")
    print(f"Total errors: {total_stats['errors']}")
    print("\nDone! You can now train with:")
    print(f"  python src/training/train.py --train-dir {data_dir}/train_landmarks --val-dir {data_dir}/val_landmarks")


if __name__ == "__main__":
    main()

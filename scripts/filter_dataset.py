#!/usr/bin/env python3
"""Filter raw dataset using the trained model as a quality filter.

This script uses the existing model to identify which videos are "learnable" -
videos where the model can correctly identify the sign with reasonable confidence.

Quality Tiers:
- GOOD: Correct prediction with confidence >= high_threshold (default 0.7)
- USABLE: Correct prediction with confidence >= low_threshold (default 0.4)
         OR correct label in top-3 predictions
- BAD: Model cannot identify the sign (wrong prediction, low confidence)

Usage:
    python scripts/filter_dataset.py --raw-dir raw/ --output-dir data/
    python scripts/filter_dataset.py --dry-run  # Just analyze, don't copy
    python scripts/filter_dataset.py --high-threshold 0.8 --low-threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import SignLanguagePredictor
from src.utils.device import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class VideoQuality:
    """Quality assessment for a single video."""
    video_path: Path
    expected_label: str
    predicted_label: str
    confidence: float
    top_k_labels: list[str]
    top_k_confidences: list[float]
    quality_tier: str  # "good", "usable", "bad"
    is_correct: bool
    is_top3: bool
    error: Optional[str] = None


def assess_video_quality(
    predictor: SignLanguagePredictor,
    video_path: Path,
    expected_label: str,
    high_threshold: float = 0.7,
    low_threshold: float = 0.4,
) -> VideoQuality:
    """Assess quality of a single video using the model.
    
    Args:
        predictor: Loaded sign language predictor.
        video_path: Path to the video file.
        expected_label: Expected gloss label for this video.
        high_threshold: Confidence threshold for "good" quality.
        low_threshold: Confidence threshold for "usable" quality.
    
    Returns:
        VideoQuality assessment result.
    """
    try:
        # Run prediction
        result = predictor.predict_from_video(video_path, top_k=5)
        
        # Extract top-k info
        top_k_labels = [g for g, c, i in result.top_k]
        top_k_confidences = [c for g, c, i in result.top_k]
        
        # Check if prediction is correct
        is_correct = result.gloss == expected_label or (
            result.gloss == "unknown" and top_k_labels[0] == expected_label
        )
        actual_confidence = top_k_confidences[0] if top_k_labels else 0.0
        
        # Check if in top-3
        is_top3 = expected_label in top_k_labels[:3]
        
        # Determine quality tier
        if is_correct and actual_confidence >= high_threshold:
            quality_tier = "good"
        elif is_correct and actual_confidence >= low_threshold:
            quality_tier = "usable"
        elif is_top3 and top_k_confidences[top_k_labels.index(expected_label)] >= low_threshold:
            quality_tier = "usable"
        else:
            quality_tier = "bad"
        
        return VideoQuality(
            video_path=video_path,
            expected_label=expected_label,
            predicted_label=top_k_labels[0] if top_k_labels else "unknown",
            confidence=actual_confidence,
            top_k_labels=top_k_labels,
            top_k_confidences=top_k_confidences,
            quality_tier=quality_tier,
            is_correct=is_correct,
            is_top3=is_top3,
        )
        
    except Exception as e:
        logger.warning(f"Error processing {video_path}: {e}")
        return VideoQuality(
            video_path=video_path,
            expected_label=expected_label,
            predicted_label="error",
            confidence=0.0,
            top_k_labels=[],
            top_k_confidences=[],
            quality_tier="bad",
            is_correct=False,
            is_top3=False,
            error=str(e),
        )


def scan_raw_dataset(raw_dir: Path) -> dict[str, list[Path]]:
    """Scan raw dataset directory for videos organized by gloss.
    
    Args:
        raw_dir: Path to raw dataset directory.
    
    Returns:
        Dictionary mapping gloss names to list of video paths.
    """
    dataset = {}
    
    for gloss_dir in sorted(raw_dir.iterdir()):
        if not gloss_dir.is_dir() or gloss_dir.name.startswith('.'):
            continue
        
        videos = list(gloss_dir.glob("*.mp4"))
        if videos:
            dataset[gloss_dir.name] = videos
            logger.info(f"Found {len(videos)} videos for '{gloss_dir.name}'")
    
    return dataset


def filter_dataset(
    raw_dir: Path,
    output_dir: Path,
    model_path: Path,
    high_threshold: float = 0.7,
    low_threshold: float = 0.4,
    include_usable: bool = True,
    dry_run: bool = False,
    report_path: Optional[Path] = None,
    sample_per_class: Optional[int] = None,
) -> dict:
    """Filter dataset using model predictions as quality indicator.
    
    Args:
        raw_dir: Path to raw dataset directory.
        output_dir: Path to filtered output directory.
        model_path: Path to trained model checkpoint.
        high_threshold: Confidence threshold for "good" quality.
        low_threshold: Confidence threshold for "usable" quality.
        include_usable: Whether to include "usable" tier videos.
        dry_run: If True, only analyze without copying files.
        report_path: Path to save detailed CSV report.
        sample_per_class: If set, only process this many videos per class (for quick testing).
    
    Returns:
        Statistics dictionary.
    """
    import random
    # Scan raw dataset
    logger.info(f"Scanning raw dataset: {raw_dir}")
    raw_dataset = scan_raw_dataset(raw_dir)
    total_videos = sum(len(v) for v in raw_dataset.values())
    logger.info(f"Found {total_videos} videos across {len(raw_dataset)} glosses")
    
    # Load predictor
    logger.info(f"Loading model from: {model_path}")
    predictor = SignLanguagePredictor(
        model_path=model_path,
        confidence_threshold=0.0,  # We handle thresholding ourselves
    )
    
    # Process all videos
    all_results: list[VideoQuality] = []
    stats = {
        "total": 0,
        "good": 0,
        "usable": 0,
        "bad": 0,
        "errors": 0,
        "by_gloss": {},
    }
    
    try:
        for gloss, videos in tqdm(raw_dataset.items(), desc="Processing glosses"):
            gloss_stats = {"total": 0, "good": 0, "usable": 0, "bad": 0}
            
            # Optionally sample videos for quick testing
            if sample_per_class is not None and len(videos) > sample_per_class:
                videos = random.sample(videos, sample_per_class)
            
            for video_path in tqdm(videos, desc=f"  {gloss}", leave=False):
                result = assess_video_quality(
                    predictor=predictor,
                    video_path=video_path,
                    expected_label=gloss,
                    high_threshold=high_threshold,
                    low_threshold=low_threshold,
                )
                
                all_results.append(result)
                stats["total"] += 1
                gloss_stats["total"] += 1
                
                if result.error:
                    stats["errors"] += 1
                
                stats[result.quality_tier] += 1
                gloss_stats[result.quality_tier] += 1
                
                # Copy file if not dry run and meets quality criteria
                if not dry_run:
                    if result.quality_tier == "good" or (include_usable and result.quality_tier == "usable"):
                        dest_dir = output_dir / gloss
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = dest_dir / video_path.name
                        
                        if not dest_path.exists():
                            shutil.copy2(video_path, dest_path)
            
            stats["by_gloss"][gloss] = gloss_stats
    
    finally:
        predictor.close()
    
    # Save detailed report
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "video_path", "expected_label", "predicted_label", "confidence",
                "quality_tier", "is_correct", "is_top3", "top_k_labels", "error"
            ])
            for r in all_results:
                writer.writerow([
                    str(r.video_path),
                    r.expected_label,
                    r.predicted_label,
                    f"{r.confidence:.4f}",
                    r.quality_tier,
                    r.is_correct,
                    r.is_top3,
                    "|".join(r.top_k_labels),
                    r.error or "",
                ])
        logger.info(f"Detailed report saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET FILTERING SUMMARY")
    print("=" * 60)
    print(f"Total videos scanned: {stats['total']}")
    print(f"  Good quality:   {stats['good']:5d} ({stats['good']/stats['total']*100:.1f}%)")
    print(f"  Usable quality: {stats['usable']:5d} ({stats['usable']/stats['total']*100:.1f}%)")
    print(f"  Bad quality:    {stats['bad']:5d} ({stats['bad']/stats['total']*100:.1f}%)")
    print(f"  Errors:         {stats['errors']:5d}")
    print()
    
    kept = stats['good'] + (stats['usable'] if include_usable else 0)
    print(f"Videos to keep: {kept} ({kept/stats['total']*100:.1f}%)")
    
    if dry_run:
        print("\n[DRY RUN] No files were copied.")
    else:
        print(f"\nFiltered videos saved to: {output_dir}")
    
    # Show worst performing glosses
    print("\n" + "-" * 60)
    print("GLOSSES WITH LOWEST QUALITY (most 'bad' videos):")
    print("-" * 60)
    
    sorted_glosses = sorted(
        stats["by_gloss"].items(),
        key=lambda x: x[1]["bad"] / x[1]["total"] if x[1]["total"] > 0 else 0,
        reverse=True
    )
    
    for gloss, gs in sorted_glosses[:15]:
        bad_pct = gs["bad"] / gs["total"] * 100 if gs["total"] > 0 else 0
        print(f"  {gloss:20s}: {gs['bad']:3d}/{gs['total']:3d} bad ({bad_pct:5.1f}%)")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter raw dataset using trained model as quality filter"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("raw"),
        help="Path to raw dataset directory (default: raw/)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Path to filtered output directory (default: data/)"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/best.pt"),
        help="Path to trained model checkpoint (default: models/best.pt)"
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for 'good' quality (default: 0.7)"
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for 'usable' quality (default: 0.4)"
    )
    parser.add_argument(
        "--good-only",
        action="store_true",
        help="Only keep 'good' quality videos (exclude 'usable')"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only, don't copy files"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/dataset_quality.csv"),
        help="Path to save detailed CSV report"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N videos per class for quick testing (default: all)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.raw_dir.exists():
        logger.error(f"Raw directory not found: {args.raw_dir}")
        return 1
    
    if not args.model_path.exists():
        logger.error(f"Model not found: {args.model_path}")
        return 1
    
    # Run filtering
    stats = filter_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        include_usable=not args.good_only,
        dry_run=args.dry_run,
        report_path=args.report,
        sample_per_class=args.sample,
    )
    
    # Save stats as JSON
    stats_path = Path("reports/filtering_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stats saved to: {stats_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())

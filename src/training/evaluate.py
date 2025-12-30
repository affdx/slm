#!/usr/bin/env python3
"""Evaluation script for trained MSL models.

Usage:
    python src/training/evaluate.py --model-path models/best.pt --test-dir data/test_landmarks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MSLLandmarkDataset
from src.data.transforms import get_eval_transforms
from src.models.lstm import LSTMClassifier, BiLSTMWithAttention, LSTMWithPooling


def get_model_class(model_type: str):
    """Get model class by type name."""
    models = {
        "lstm": LSTMClassifier,
        "bilstm": LSTMClassifier,
        "bilstm_attention": BiLSTMWithAttention,
        "lstm_pooling": LSTMWithPooling,
    }
    return models.get(model_type, BiLSTMWithAttention)


def load_model(model_path: Path, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {})
    
    # Remove non-constructor args
    config.pop("num_parameters", None)
    
    # Determine model type from config
    if "attention_heads" in config:
        model = BiLSTMWithAttention(**config)
    elif config.get("bidirectional", False):
        model = LSTMClassifier(**config)
    else:
        model = LSTMClassifier(**config)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    """Evaluate model on dataset.
    
    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc="Evaluating"):
            data = data.to(device)
            targets = targets.to(device)
            
            outputs = model(data)
            probs = F.softmax(outputs, dim=-1)
            preds = outputs.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    
    # Top-5 accuracy
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_correct = np.array([label in top5 for label, top5 in zip(all_labels, top5_preds)])
    top5_accuracy = top5_correct.mean()
    
    # Per-class accuracy
    class_correct = {}
    class_total = {}
    for pred, label in zip(all_preds, all_labels):
        class_name = class_names[label]
        if class_name not in class_total:
            class_total[class_name] = 0
            class_correct[class_name] = 0
        class_total[class_name] += 1
        if pred == label:
            class_correct[class_name] += 1
    
    per_class_accuracy = {
        name: class_correct[name] / class_total[name] 
        for name in class_names if name in class_total
    }
    
    # Confusion matrix
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        confusion_matrix[label, pred] += 1
    
    return {
        "accuracy": float(accuracy),
        "top5_accuracy": float(top5_accuracy),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix.tolist(),
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
        "num_samples": len(all_labels),
        "num_classes": num_classes,
    }


def print_results(results: dict, class_names: list[str]) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f} ({results['top5_accuracy']*100:.2f}%)")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Number of classes: {results['num_classes']}")
    
    # Best and worst classes
    per_class = results["per_class_accuracy"]
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
    
    print("\n--- Top 10 Best Classes ---")
    for name, acc in sorted_classes[:10]:
        print(f"  {name}: {acc:.4f}")
    
    print("\n--- Top 10 Worst Classes ---")
    for name, acc in sorted_classes[-10:]:
        print(f"  {name}: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MSL model")
    parser.add_argument("--model-path", type=str, default="models/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--test-dir", type=str, default="data/test_landmarks",
                        help="Path to test data directory")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "mps", "cuda", "cpu"],
                        help="Device to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model, checkpoint = load_model(Path(args.model_path), device)
    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Load test dataset
    print(f"Loading test data from: {args.test_dir}")
    transform = get_eval_transforms()
    test_dataset = MSLLandmarkDataset(args.test_dir, transform=transform)
    print(f"Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Evaluate
    results = evaluate(model, test_loader, device, test_dataset.classes)
    
    # Print results
    print_results(results, test_dataset.classes)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        # Remove non-serializable items for JSON
        save_results = {k: v for k, v in results.items() 
                       if k not in ["predictions", "labels", "confusion_matrix"]}
        save_results["class_names"] = test_dataset.classes
        with open(output_path, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

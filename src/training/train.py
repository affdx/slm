#!/usr/bin/env python3
"""Main training script for Malaysian Sign Language model.

This script trains the BetterLSTM model using pre-extracted NPY landmarks.
Key features:
- Per-feature normalization (saved for inference)
- Gaussian noise augmentation
- Class weights for imbalanced data
- Label smoothing
- Early stopping with patience

Usage:
    python src/training/train.py
    python src/training/train.py --epochs 200 --patience 20
    python src/training/train.py --data-dir data/npy --output-dir models
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import BetterLSTM


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    """Get the best available device: MPS > CUDA > CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def format_time(sec: float) -> str:
    """Format seconds into H:MM:SS or MM:SS."""
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute sqrt-scaled class weights for imbalanced data."""
    counts = np.bincount(y, minlength=num_classes)
    weights = counts.sum() / (counts + 1e-6)
    weights = np.sqrt(weights)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def load_npy_data(data_dir: Path) -> tuple:
    """Load NPY dataset files.
    
    Supports two naming conventions:
    1. New format: X_train.npy, y_train.npy, etc.
    2. Legacy format: X_TRAIN_2.npy, y_TRAIN_2.npy, etc.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, label_map)
    """
    # Try new naming convention first, fall back to legacy
    def load_split(split_name: str) -> tuple:
        # New format (lowercase)
        x_path = data_dir / f"X_{split_name}.npy"
        y_path = data_dir / f"y_{split_name}.npy"
        
        if x_path.exists() and y_path.exists():
            return (
                np.load(x_path).astype(np.float32),
                np.load(y_path).astype(np.int64),
            )
        
        # Legacy format (uppercase with _2 suffix)
        x_path_legacy = data_dir / f"X_{split_name.upper()}_2.npy"
        y_path_legacy = data_dir / f"y_{split_name.upper()}_2.npy"
        
        if x_path_legacy.exists() and y_path_legacy.exists():
            return (
                np.load(x_path_legacy).astype(np.float32),
                np.load(y_path_legacy).astype(np.int64),
            )
        
        raise FileNotFoundError(f"Could not find {split_name} data in {data_dir}")
    
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    
    # Load label map
    label_map_path = data_dir / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
    else:
        num_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
        label_map = {str(i): i for i in range(num_classes)}
    
    return X_train, y_train, X_val, y_val, X_test, y_test, label_map


def compute_normalization_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature normalization statistics from training data."""
    feat_mean = X_train.mean(axis=(0, 1), keepdims=True)
    feat_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return feat_mean, feat_std


def normalize_data(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply per-feature normalization."""
    return (X - mean) / std


def run_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_preds: bool = False,
) -> tuple:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_y = []
    all_pred = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            logits = model(xb)
            loss = criterion(logits, yb)
            
            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
            
            if return_preds:
                all_y.append(yb.cpu().numpy())
                all_pred.append(pred.cpu().numpy())
    
    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    
    if return_preds:
        return avg_loss, accuracy, np.concatenate(all_y), np.concatenate(all_pred)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train BetterLSTM model for MSL")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, default="data/npy",
                        help="Directory containing NPY dataset files")
    
    # Model arguments
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.35,
                        help="Dropout probability")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2,
                        help="Weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="Gaussian noise std for augmentation")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping max norm")
    
    # Scheduler arguments
    parser.add_argument("--scheduler-patience", type=int, default=8,
                        help="Scheduler patience for ReduceLROnPlateau")
    parser.add_argument("--scheduler-factor", type=float, default=0.5,
                        help="Scheduler reduction factor")
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Output directory for model and stats")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    seed_everything(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Malaysian Sign Language Model Training")
    print("=" * 60)
    
    # Load data
    data_dir = Path(args.data_dir)
    print(f"\nLoading data from: {data_dir}")
    
    X_train, y_train, X_val, y_val, X_test, y_test, label_map = load_npy_data(data_dir)
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    
    num_classes = int(max(y_train.max(), y_val.max(), y_test.max())) + 1
    seq_len = X_train.shape[1]
    input_size = X_train.shape[2]
    
    print(f"seq_len={seq_len}, input_size={input_size}, num_classes={num_classes}")
    
    # Create target names for classification report
    idx2name = {int(v): k for k, v in label_map.items()}
    target_names = [idx2name.get(i, str(i)) for i in range(num_classes)]
    
    # Compute and save normalization stats
    print("\nComputing normalization statistics...")
    feat_mean, feat_std = compute_normalization_stats(X_train)
    
    norm_stats_path = output_dir / "norm_stats.npz"
    np.savez(norm_stats_path, mean=feat_mean, std=feat_std)
    print(f"Saved norm stats to: {norm_stats_path}")
    
    # Normalize data
    X_train = normalize_data(X_train, feat_mean, feat_std)
    X_val = normalize_data(X_val, feat_mean, feat_std)
    X_test = normalize_data(X_test, feat_mean, feat_std)
    
    # Compute class weights
    print("\nComputing class weights...")
    counts = np.bincount(y_train, minlength=num_classes)
    print(f"Class counts: min={counts.min()}, max={counts.max()}, "
          f"imbalance ratio={counts.max() / max(1, counts.min()):.2f}")
    
    class_weights = compute_class_weights(y_train, num_classes).to(device)
    
    # Create DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=(device.type == "cuda"))
    
    # Create model
    print("\nCreating BetterLSTM model...")
    model = BetterLSTM(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    
    print(model)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Loss functions
    train_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience
    )
    
    # Training state
    best_val_loss = float("inf")
    best_epoch = -1
    no_improve = 0
    
    best_weights_path = output_dir / "best.pt"
    
    # History tracking
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "test_loss": [], "test_acc": [],
        "val_f1": [], "test_f1": [],
        "lr": [],
    }
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    global_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            # Add gaussian noise for augmentation
            if args.noise_std > 0:
                xb = xb + args.noise_std * torch.randn_like(xb)
            
            optimizer.zero_grad(set_to_none=True)
            
            logits = model(xb)
            loss = train_criterion(logits, yb)
            
            loss.backward()
            
            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            # Track metrics
            with torch.no_grad():
                eval_loss = eval_criterion(logits, yb)
                running_loss += eval_loss.item() * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        
        # Evaluate
        val_loss, val_acc, yv, pv = run_eval(model, val_loader, eval_criterion, device, return_preds=True)
        test_loss, test_acc, yt, pt = run_eval(model, test_loader, eval_criterion, device, return_preds=True)
        
        val_f1 = f1_score(yv, pv, average="macro", zero_division=0)
        test_f1 = f1_score(yt, pt, average="macro", zero_division=0)
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["val_f1"].append(val_f1)
        history["test_f1"].append(test_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Check for improvement
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            
            # Save best model
            torch.save(model.state_dict(), best_weights_path)
        else:
            no_improve += 1
        
        # Timing
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - global_start
        avg_epoch = elapsed / epoch
        eta = avg_epoch * (args.epochs - epoch)
        
        lr_now = optimizer.param_groups[0]["lr"]
        is_best = "(best)" if epoch == best_epoch else ""
        
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"train_loss:{train_loss:.4f} val_loss:{val_loss:.4f} test_loss:{test_loss:.4f} | "
            f"train_acc:{train_acc:.4f} val_acc:{val_acc:.4f} test_acc:{test_acc:.4f} | "
            f"val_F1:{val_f1:.4f} test_F1:{test_f1:.4f} | "
            f"lr:{lr_now:.2e} time:{format_time(epoch_time)} ETA:{format_time(eta)} {is_best}"
        )
        
        # Early stopping
        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch={epoch}. Best epoch={best_epoch}, best val_loss={best_val_loss:.4f}")
            break
    
    # Final evaluation with best weights
    print("\n" + "=" * 60)
    print(f"Loading best weights from: {best_weights_path}")
    print("=" * 60)
    
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    
    final_test_loss, final_test_acc, yt, pt = run_eval(model, test_loader, eval_criterion, device, return_preds=True)
    
    print(f"\nBest Epoch: {best_epoch}")
    print(f"Final Test Loss: {final_test_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
    print(f"Final Test F1 (macro): {f1_score(yt, pt, average='macro', zero_division=0):.4f}")
    print(f"Final Test F1 (weighted): {f1_score(yt, pt, average='weighted', zero_division=0):.4f}")
    
    print("\n=== Per-class Classification Report (Test) ===")
    print(classification_report(yt, pt, target_names=target_names, digits=2, zero_division=0))
    
    # Save class mapping
    class_mapping_path = output_dir / "class_mapping.json"
    with open(class_mapping_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Class mapping saved to: {class_mapping_path}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Save full checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": history["val_acc"][best_epoch - 1],
        "final_test_acc": final_test_acc,
        "history": history,
    }
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Full checkpoint saved to: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFiles saved to {output_dir}:")
    print(f"  - best.pt (model weights)")
    print(f"  - norm_stats.npz (normalization stats)")
    print(f"  - class_mapping.json")
    print(f"  - checkpoint.pt (full checkpoint)")
    print(f"  - training_history.json")


if __name__ == "__main__":
    main()

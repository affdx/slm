#!/usr/bin/env python3
"""Main training script for Malaysian Sign Language model.

Usage:
    python src/training/train.py --epochs 50 --batch-size 32
    python src/training/train.py --device mps --model bilstm
    python src/training/train.py --resume models/last.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MSLVideoDataset, MSLLandmarkDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.lstm import LSTMClassifier, BiLSTMWithAttention, LSTMWithPooling
from src.training.trainer import Trainer, TrainingConfig
from src.training.losses import get_loss_function


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_model(model_type: str, num_classes: int = 90, **kwargs) -> torch.nn.Module:
    """Get model by type name."""
    models = {
        "lstm": LSTMClassifier,
        "bilstm": lambda **kw: LSTMClassifier(bidirectional=True, **kw),
        "bilstm_attention": BiLSTMWithAttention,
        "lstm_pooling": LSTMWithPooling,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    model_cls = models[model_type]
    return model_cls(num_classes=num_classes, **kwargs)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train MSL sign language model")

    # Data arguments
    parser.add_argument("--train-dir", type=str, default="data/train",
                        help="Training data directory")
    parser.add_argument("--val-dir", type=str, default="data/val",
                        help="Validation data directory")
    parser.add_argument("--num-frames", type=int, default=30,
                        help="Number of frames per video")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--cache-landmarks", action="store_true",
                        help="Cache extracted landmarks")

    # Model arguments
    parser.add_argument("--model", type=str, default="lstm",
                        choices=["lstm", "bilstm", "bilstm_attention", "lstm_pooling"],
                        help="Model architecture")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="LSTM hidden size")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")

    # Loss arguments
    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy", "focal", "label_smoothing"],
                        help="Loss function")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor")

    # Scheduler arguments
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "cosine", "step", "none"],
                        help="Learning rate scheduler")
    parser.add_argument("--patience", type=int, default=5,
                        help="Scheduler patience")

    # Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Early stopping patience")

    # Output arguments
    parser.add_argument("--checkpoint-dir", type=str, default="models",
                        help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="runs",
                        help="TensorBoard log directory")

    # Device arguments
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "mps", "cuda", "cpu"],
                        help="Device to use for training")

    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable data augmentation")

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        config = load_config(Path(args.config))
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None and key != "config":
                # Nested config handling
                pass  # Use args values as override

    # Set seed
    set_seed(args.seed)

    print("=" * 60)
    print("Malaysian Sign Language Model Training")
    print("=" * 60)

    # Create transforms
    train_transform = None if args.no_augmentation else get_train_transforms()
    val_transform = get_eval_transforms()

    # Create datasets
    # Check if using pre-extracted landmarks (.npy files) or videos (.mp4 files)
    train_dir = Path(args.train_dir)
    use_landmarks = (train_dir / next(train_dir.iterdir()).name).glob("*.npy").__next__() if train_dir.exists() else False
    
    print(f"\nLoading training data from: {args.train_dir}")
    
    # Detect if directory contains .npy files (landmarks) or .mp4 files (videos)
    is_landmark_dir = any(train_dir.glob("*/*.npy")) if train_dir.exists() else False
    
    if is_landmark_dir:
        print("  (using pre-extracted landmarks)")
        train_dataset = MSLLandmarkDataset(
            args.train_dir,
            transform=train_transform,
        )
        print(f"Loading validation data from: {args.val_dir}")
        print("  (using pre-extracted landmarks)")
        val_dataset = MSLLandmarkDataset(
            args.val_dir,
            transform=val_transform,
        )
    else:
        print("  (extracting landmarks from videos - this may be slow)")
        train_dataset = MSLVideoDataset(
            args.train_dir,
            num_frames=args.num_frames,
            transform=train_transform,
            cache_landmarks=args.cache_landmarks,
        )
        print(f"Loading validation data from: {args.val_dir}")
        val_dataset = MSLVideoDataset(
            args.val_dir,
            num_frames=args.num_frames,
            transform=val_transform,
            cache_landmarks=args.cache_landmarks,
        )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    print(f"\nCreating model: {args.model}")
    model = get_model(
        args.model,
        num_classes=len(train_dataset.classes),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    print(f"Model parameters: {model.count_parameters():,}")

    # Create loss function
    criterion = get_loss_function(
        args.loss,
        num_classes=len(train_dataset.classes),
        label_smoothing=args.label_smoothing,
    )

    # Create training config
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        scheduler_type=args.scheduler,
        scheduler_patience=args.patience,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=args.device,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Save class mapping
    class_mapping_path = Path(args.checkpoint_dir) / "class_mapping.json"
    train_dataset.save_class_mapping(class_mapping_path)
    print(f"Class mapping saved to: {class_mapping_path}")

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    history = trainer.train()

    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation accuracy: {trainer.state.best_val_acc:.4f}")
    print(f"Best validation loss: {trainer.state.best_val_loss:.4f}")
    print(f"Model saved to: {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()

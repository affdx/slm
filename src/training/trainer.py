"""Training loop and utilities for sign language models.

This module provides the main training loop with support for
validation, early stopping, checkpointing, and logging.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if TYPE_CHECKING:
    from src.models.base import BaseModel


def _get_device(preferred: Optional[str] = None) -> torch.device:
    """Get the best available device."""
    if preferred is not None:
        preferred = preferred.lower()
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif preferred == "cpu":
            return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Training parameters
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32

    # Scheduler
    scheduler_type: str = "plateau"  # 'plateau', 'cosine', 'step', 'none'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Gradient clipping
    gradient_clip: float = 1.0

    # Checkpointing
    checkpoint_dir: str = "models"
    save_best_only: bool = False

    # Logging
    log_dir: str = "runs"
    log_interval: int = 10

    # Device
    device: str = "auto"  # 'auto', 'mps', 'cuda', 'cpu'


@dataclass
class TrainingState:
    """State of training for checkpointing."""

    epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    epochs_without_improvement: int = 0
    history: dict = field(default_factory=lambda: {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    })


class TrainingLogger:
    """Simple training logger."""

    def __init__(self, name: str = "training"):
        import logging
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def log_epoch_end(
        self, epoch: int, train_loss: float, val_loss: float,
        train_acc: float, val_acc: float, lr: float
    ) -> None:
        self.logger.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {lr:.2e}"
        )

    def log_model_saved(self, path: Path, is_best: bool = False) -> None:
        msg = f"Model saved to {path}"
        if is_best:
            msg += " (best)"
        self.logger.info(msg)

    def log_early_stopping(self, epoch: int, patience: int) -> None:
        self.logger.info(
            f"Early stopping at epoch {epoch} after {patience} epochs without improvement"
        )


class Trainer:
    """Training loop manager."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: TrainingConfig,
        logger: Optional[TrainingLogger] = None,
    ):
        """Initialize the trainer.

        Args:
            model: PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            criterion: Loss function.
            config: Training configuration.
            logger: Optional training logger.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.logger = logger or TrainingLogger()

        # Set device
        if config.device == "auto":
            self.device = _get_device()
        else:
            self.device = _get_device(config.device)

        self.model = self.model.to(self.device)
        self.logger.info(f"Training on device: {self.device}")

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Initialize state
        self.state = TrainingState()

        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=config.log_dir)

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        if self.config.scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.min_lr,
            )
        elif self.config.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr,
            )
        elif self.config.scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=self.config.scheduler_factor,
            )
        return None

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.state.epoch + 1}",
            leave=False,
        )

        for batch_idx, (data, targets) in enumerate(progress_bar):
            data = data.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "acc": 100.0 * correct / total,
            })

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """Validate the model.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, targets in self.val_loader:
            data = data.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self) -> dict[str, list]:
        """Run the full training loop.

        Returns:
            Training history dictionary.
        """
        self.logger.info(f"Starting training for {self.config.epochs} epochs")

        # Count parameters if model has the method
        if hasattr(self.model, 'count_parameters'):
            self.logger.info(f"Model parameters: {self.model.count_parameters():,}")

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log metrics
            self.logger.log_epoch_end(
                epoch + 1, train_loss, val_loss, train_acc, val_acc, current_lr
            )

            # TensorBoard logging
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("LearningRate", current_lr, epoch)

            # Update history
            self.state.history["train_loss"].append(train_loss)
            self.state.history["train_acc"].append(train_acc)
            self.state.history["val_loss"].append(val_loss)
            self.state.history["val_acc"].append(val_acc)
            self.state.history["lr"].append(current_lr)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Check for improvement
            improved = val_loss < self.state.best_val_loss - self.config.early_stopping_min_delta

            if improved:
                self.state.best_val_loss = val_loss
                self.state.best_val_acc = val_acc
                self.state.epochs_without_improvement = 0

                # Save best model
                self._save_checkpoint("best.pt", is_best=True)
            else:
                self.state.epochs_without_improvement += 1

            # Save latest model
            if not self.config.save_best_only:
                self._save_checkpoint("last.pt")

            # Early stopping
            if (
                self.config.early_stopping
                and self.state.epochs_without_improvement >= self.config.early_stopping_patience
            ):
                self.logger.log_early_stopping(
                    epoch + 1, self.config.early_stopping_patience
                )
                break

        self.writer.close()
        self.logger.info(
            f"Training completed. Best val acc: {self.state.best_val_acc:.4f}"
        )

        return self.state.history

    def _save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save a checkpoint."""
        path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.state.epoch,
            "best_val_loss": self.state.best_val_loss,
            "best_val_acc": self.state.best_val_acc,
            "history": self.state.history,
        }

        # Add model config if available
        if hasattr(self.model, 'get_config'):
            checkpoint["model_config"] = self.model.get_config()

        torch.save(checkpoint, path)
        self.logger.log_model_saved(path, is_best)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a checkpoint and resume training."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.state.epoch = checkpoint.get("epoch", 0)
        self.state.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.state.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        if "history" in checkpoint:
            self.state.history = checkpoint["history"]

        self.logger.info(f"Resumed training from epoch {self.state.epoch + 1}")

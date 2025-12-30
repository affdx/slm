"""Logging configuration for the MSL project.

This module provides a centralized logging configuration with support for
both console and file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "msl",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up and configure a logger.

    Args:
        name: Logger name (default: 'msl').
        level: Logging level (default: INFO).
        log_file: Optional path to log file. If provided, logs will also
                  be written to this file.
        format_string: Optional custom format string for log messages.

    Returns:
        logging.Logger: Configured logger instance.

    Examples:
        >>> logger = setup_logger("training", logging.DEBUG)
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Default format
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "msl") -> logging.Logger:
    """Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name.

    Returns:
        logging.Logger: Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class TrainingLogger:
    """A specialized logger for training progress.

    This logger provides methods for logging training metrics and progress
    in a consistent format.
    """

    def __init__(self, name: str = "training", log_file: Optional[Path] = None):
        """Initialize the training logger.

        Args:
            name: Logger name.
            log_file: Optional path to log file.
        """
        self.logger = setup_logger(name, log_file=log_file)
        self.epoch = 0
        self.step = 0

    def log_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Log the start of a training epoch."""
        self.epoch = epoch
        self.logger.info(f"Epoch {epoch}/{total_epochs} started")

    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """Log metrics at the end of an epoch."""
        self.logger.info(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {lr:.2e}"
        )

    def log_batch(
        self,
        batch: int,
        total_batches: int,
        loss: float,
        acc: float,
    ) -> None:
        """Log metrics for a training batch."""
        self.step += 1
        if batch % 10 == 0 or batch == total_batches - 1:
            self.logger.debug(
                f"Batch {batch}/{total_batches} | Loss: {loss:.4f} | Acc: {acc:.4f}"
            )

    def log_model_saved(self, path: Path, is_best: bool = False) -> None:
        """Log when a model checkpoint is saved."""
        msg = f"Model saved to {path}"
        if is_best:
            msg += " (best)"
        self.logger.info(msg)

    def log_early_stopping(self, epoch: int, patience: int) -> None:
        """Log early stopping event."""
        self.logger.info(
            f"Early stopping triggered at epoch {epoch} after {patience} epochs without improvement"
        )

    def info(self, msg: str) -> None:
        """Log an info message."""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log a warning message."""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log an error message."""
        self.logger.error(msg)

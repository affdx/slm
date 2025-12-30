"""Base model class for sign language classification.

This module provides the base class and common utilities for
all model architectures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for sign language classification models.

    All model architectures should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(
        self,
        input_size: int = 258,
        num_classes: int = 90,
        **kwargs: Any,
    ):
        """Initialize the base model.

        Args:
            input_size: Number of input features per frame (default: 258).
            num_classes: Number of output classes (default: 90).
            **kwargs: Additional arguments for subclasses.
        """
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        pass

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with probabilities.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            Tuple of (predicted_classes, probabilities).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path, optimizer: Optional[torch.optim.Optimizer] = None,
             epoch: Optional[int] = None, **kwargs: Any) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save the checkpoint.
            optimizer: Optional optimizer to save state.
            epoch: Optional current epoch number.
            **kwargs: Additional data to save.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": self.get_config(),
            "epoch": epoch,
            **kwargs,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> tuple["BaseModel", dict]:
        """Load model from checkpoint.

        Args:
            path: Path to the checkpoint file.
            device: Device to load the model onto.
            **kwargs: Additional arguments for model initialization.

        Returns:
            Tuple of (model, checkpoint_dict).
        """
        path = Path(path)
        if device is None:
            checkpoint = torch.load(path, weights_only=False)
        else:
            checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Get model config from checkpoint and merge with kwargs
        config = checkpoint.get("model_config", {})
        # Remove non-constructor args
        config.pop("num_parameters", None)
        config.update(kwargs)

        # Create model instance with saved config
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])

        if device is not None:
            model = model.to(device)

        return model, checkpoint

    def get_config(self) -> dict[str, Any]:
        """Get model configuration as a dictionary."""
        return {
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "num_parameters": self.count_parameters(),
        }


def init_weights(module: nn.Module) -> None:
    """Initialize weights using Xavier/Kaiming initialization.

    Args:
        module: Neural network module to initialize.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

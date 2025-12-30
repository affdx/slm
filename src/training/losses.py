"""Loss functions for sign language classification.

This module provides various loss functions including standard
cross-entropy, focal loss, and label smoothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Focal loss down-weights easy examples and focuses on hard negatives.
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """Initialize focal loss.

        Args:
            alpha: Optional class weights tensor of shape (num_classes,).
            gamma: Focusing parameter (default: 2.0).
            reduction: 'none', 'mean', or 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predictions of shape (batch_size, num_classes).
            targets: Ground truth labels of shape (batch_size,).

        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy loss with label smoothing.

    Label smoothing helps prevent over-confidence and improves
    generalization.
    """

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        """Initialize label smoothing loss.

        Args:
            num_classes: Number of classes.
            smoothing: Smoothing factor (default: 0.1).
            reduction: 'none', 'mean', or 'sum'.
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing loss.

        Args:
            inputs: Predictions of shape (batch_size, num_classes).
            targets: Ground truth labels of shape (batch_size,).

        Returns:
            Label smoothing loss value.
        """
        log_probs = F.log_softmax(inputs, dim=-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        # Compute loss
        loss = (-smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """Combine multiple loss functions with weights."""

    def __init__(
        self,
        losses: list[nn.Module],
        weights: Optional[list[float]] = None,
    ):
        """Initialize combined loss.

        Args:
            losses: List of loss functions.
            weights: Optional list of weights for each loss.
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            inputs: Predictions of shape (batch_size, num_classes).
            targets: Ground truth labels of shape (batch_size,).

        Returns:
            Weighted sum of losses.
        """
        total_loss: torch.Tensor = torch.tensor(0.0, device=inputs.device)
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss = total_loss + weight * loss_fn(inputs, targets)
        return total_loss


def get_loss_function(
    loss_type: str = "cross_entropy",
    num_classes: int = 90,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Get a loss function by name.

    Args:
        loss_type: Type of loss ('cross_entropy', 'focal', 'label_smoothing').
        num_classes: Number of classes.
        class_weights: Optional class weights for weighted CE or focal loss.
        label_smoothing: Label smoothing factor (used if > 0).
        focal_gamma: Gamma parameter for focal loss.

    Returns:
        Loss function module.

    Raises:
        ValueError: If loss_type is not recognized.
    """
    if loss_type == "cross_entropy":
        if label_smoothing > 0:
            return LabelSmoothingLoss(
                num_classes=num_classes,
                smoothing=label_smoothing,
            )
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_type == "focal":
        return FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
        )

    elif loss_type == "label_smoothing":
        return LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=label_smoothing if label_smoothing > 0 else 0.1,
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

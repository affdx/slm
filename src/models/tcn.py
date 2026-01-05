"""TCN (Temporal Convolutional Network) models for sign language classification.

This module provides the CausalTCNClassifier architecture - a causal
temporal convolutional network optimized for real-time sign language recognition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from .base import BaseModel, init_weights


class CausalConv1d(nn.Module):
    """Causal 1D convolution that only pads on the left (strictly causal).

    This ensures that the model only uses past information, making it suitable
    for real-time streaming inference.

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels.
        kernel_size: Size of the convolution kernel (default: 3).
        dilation: Dilation rate (default: 1).
        bias: Whether to use bias (default: True).
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            c_in, c_out, kernel_size=kernel_size, dilation=dilation, padding=0, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with left padding only.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Output tensor of shape (batch, channels, time).
        """
        # Pad left only to maintain causality
        pad_left = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad_left, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Temporal block with residual connection for TCN.

    Each block consists of two causal convolutions with normalization,
    activation, and dropout, followed by a residual connection.

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels.
        kernel_size: Size of the convolution kernel (default: 3).
        dilation: Dilation rate (default: 1).
        dropout: Dropout probability (default: 0.25).
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.25):
        super().__init__()
        self.conv1 = CausalConv1d(c_in, c_out, kernel_size, dilation)
        self.conv2 = CausalConv1d(c_out, c_out, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(c_out)
        self.norm2 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

        self.down = None
        if c_in != c_out:
            self.down = nn.Conv1d(c_in, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal block.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Output tensor of shape (batch, channels, time).
        """
        # First convolution
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.drop(out)

        # Second convolution
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.drop(out)

        # Residual connection
        res = x if self.down is None else self.down(x)
        return out + res


class CausalTCNClassifier(BaseModel):
    """Causal Temporal Convolutional Network for sign language recognition.

    This architecture uses dilated causal convolutions to capture temporal
    patterns while maintaining strict causality (only uses past information).
    Suitable for real-time streaming inference.

    Key features:
    - Causal convolutions (no future information)
    - Dilated convolutions for temporal receptive field
    - Residual connections for stable training
    - LayerNorm and GELU activation
    """

    def __init__(
        self,
        input_size: int = 258,
        num_classes: int = 90,
        channels: int = 256,
        levels: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.25,
        **kwargs: Any,
    ):
        """Initialize the CausalTCNClassifier.

        Args:
            input_size: Number of input features per frame (default: 258).
            num_classes: Number of output classes (default: 90).
            channels: Number of hidden channels (default: 256).
            levels: Number of temporal blocks/levels (default: 6).
            kernel_size: Size of convolution kernel (default: 3).
            dropout: Dropout probability (default: 0.25).
            **kwargs: Additional arguments for base class.
        """
        super().__init__(input_size=input_size, num_classes=num_classes, **kwargs)

        self.channels = channels
        self.levels = levels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        # Project input features to hidden dimension
        self.in_proj = nn.Linear(input_size, channels)

        # Temporal blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(levels):
            dilation = 2 ** i
            self.blocks.append(
                TemporalBlock(channels, channels, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            )

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            Output tensor of shape (batch_size, num_classes).
        """
        # Project to hidden dimension: (B, T, D) -> (B, T, C)
        h = self.in_proj(x)

        # Transpose for conv1d: (B, T, C) -> (B, C, T)
        h = h.transpose(1, 2).contiguous()

        # Apply temporal blocks
        for block in self.blocks:
            h = block(h)  # (B, C, T)

        # Take last timestep (causal): (B, C, T) -> (B, C)
        last = h[:, :, -1]

        # Classification head
        output = self.head(last)  # (B, num_classes)

        return output

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "model": "CausalTCNClassifier",
            "channels": self.channels,
            "levels": self.levels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout_rate,
        })
        return config


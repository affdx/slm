"""LSTM-based models for sign language classification.

This module provides the BetterLSTM architecture - an optimized
Bidirectional LSTM with scalar attention for sequence classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any

from .base import BaseModel, init_weights


class AttentionPooling(nn.Module):
    """Scalar attention pooling for sequence aggregation.

    This attention mechanism learns a single weight per timestep,
    effectively learning "which frames matter" for classification.
    Simpler and more interpretable than multi-head attention.
    """

    def __init__(self, hidden_dim: int):
        """Initialize attention pooling.

        Args:
            hidden_dim: Dimension of hidden states to attend over.
        """
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply attention pooling.

        Args:
            h: Hidden states of shape (batch, seq_len, hidden_dim).

        Returns:
            Tuple of (context_vector, attention_weights).
            - context_vector: Shape (batch, hidden_dim)
            - attention_weights: Shape (batch, seq_len)
        """
        # Project and compute attention scores
        a = torch.tanh(self.proj(h))  # (B, T, H)
        score = self.v(a).squeeze(-1)  # (B, T)

        # Softmax to get attention weights
        w = torch.softmax(score, dim=1)  # (B, T)

        # Weighted sum of hidden states
        out = torch.sum(h * w.unsqueeze(-1), dim=1)  # (B, H)

        return out, w


class BetterLSTM(BaseModel):
    """Optimized BiLSTM with scalar attention for sign language recognition.

    This architecture achieves 93.86% accuracy with only 969K parameters.
    Key features:
    - Bidirectional LSTM with LayerNorm
    - Scalar attention pooling (simpler than multi-head)
    - GELU activation in MLP head
    - Smaller hidden size (128) for better generalization
    """

    def __init__(
        self,
        input_size: int = 258,
        num_classes: int = 90,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.35,
        **kwargs: Any,
    ):
        """Initialize the BetterLSTM model.

        Args:
            input_size: Number of input features per frame (default: 258).
            num_classes: Number of output classes (default: 90).
            hidden_size: LSTM hidden state size (default: 128).
            num_layers: Number of stacked LSTM layers (default: 2).
            dropout: Dropout probability (default: 0.35).
            **kwargs: Additional arguments.
        """
        super().__init__(input_size=input_size, num_classes=num_classes, **kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # LayerNorm after LSTM (stabilizes training)
        lstm_output_size = hidden_size * 2  # bidirectional
        self.norm = nn.LayerNorm(lstm_output_size)

        # Scalar attention pooling
        self.attn = AttentionPooling(lstm_output_size)

        # MLP classification head with GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
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
        # BiLSTM encoding
        h, _ = self.lstm(x)  # (B, T, H*2)

        # Layer normalization
        h = self.norm(h)

        # Attention pooling
        ctx, _ = self.attn(h)  # (B, H*2)

        # Classification
        output = self.mlp(ctx)

        return output

    def forward_with_attention(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns attention weights.

        Useful for visualization and interpretability.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            Tuple of (logits, attention_weights).
        """
        h, _ = self.lstm(x)
        h = self.norm(h)
        ctx, attn_weights = self.attn(h)
        output = self.mlp(ctx)
        return output, attn_weights

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
        })
        return config

"""LSTM-based models for sign language classification.

This module provides LSTM and Bidirectional LSTM architectures
for sequence classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any

from .base import BaseModel, init_weights


class LSTMClassifier(BaseModel):
    """Standard LSTM classifier for sign language recognition.

    This is the baseline model architecture using stacked LSTM layers
    followed by fully connected layers for classification.
    """

    def __init__(
        self,
        input_size: int = 258,
        num_classes: int = 90,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False,
        **kwargs: Any,
    ):
        """Initialize the LSTM classifier.

        Args:
            input_size: Number of input features per frame.
            num_classes: Number of output classes.
            hidden_size: LSTM hidden state size.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability.
            bidirectional: Whether to use bidirectional LSTM.
            **kwargs: Additional arguments.
        """
        super().__init__(input_size=input_size, num_classes=num_classes, **kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Calculate FC input size
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
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
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]

        # Classification
        output = self.fc(hidden)

        return output

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "bidirectional": self.bidirectional,
        })
        return config


class BiLSTMWithAttention(BaseModel):
    """Bidirectional LSTM with attention mechanism.

    This model uses attention to weight the importance of different
    time steps in the sequence.
    """

    def __init__(
        self,
        input_size: int = 258,
        num_classes: int = 90,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_heads: int = 4,
        **kwargs: Any,
    ):
        """Initialize the BiLSTM with attention.

        Args:
            input_size: Number of input features per frame.
            num_classes: Number of output classes.
            hidden_size: LSTM hidden state size.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability.
            attention_heads: Number of attention heads.
            **kwargs: Additional arguments.
        """
        super().__init__(input_size=input_size, num_classes=num_classes, **kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.attention_heads = attention_heads

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Multi-head attention
        lstm_output_size = hidden_size * 2  # bidirectional
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_size)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
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
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual connection and layer norm
        attn_out = self.layer_norm(lstm_out + attn_out)

        # Global average pooling over time
        pooled = attn_out.mean(dim=1)

        # Classification
        output = self.fc(pooled)

        return output

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "attention_heads": self.attention_heads,
        })
        return config


class LSTMWithPooling(BaseModel):
    """LSTM with multiple pooling strategies.

    This model combines max pooling and average pooling over
    the sequence for more robust feature extraction.
    """

    def __init__(
        self,
        input_size: int = 258,
        num_classes: int = 90,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        **kwargs: Any,
    ):
        """Initialize the LSTM with pooling.

        Args:
            input_size: Number of input features per frame.
            num_classes: Number of output classes.
            hidden_size: LSTM hidden state size.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability.
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
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Batch normalization
        lstm_output_size = hidden_size * 2
        self.batch_norm = nn.BatchNorm1d(lstm_output_size * 2)  # max + avg pooling

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
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
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Max pooling over time
        max_pool, _ = lstm_out.max(dim=1)

        # Average pooling over time
        avg_pool = lstm_out.mean(dim=1)

        # Concatenate pooled features
        pooled = torch.cat([max_pool, avg_pool], dim=1)

        # Batch normalization
        pooled = self.batch_norm(pooled)

        # Classification
        output = self.fc(pooled)

        return output

    def get_config(self) -> dict[str, Any]:
        """Get model configuration."""
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
        })
        return config

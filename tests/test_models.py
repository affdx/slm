"""Unit tests for model architectures."""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm import LSTMClassifier, BiLSTMWithAttention, LSTMWithPooling


class TestLSTMClassifier:
    """Tests for LSTMClassifier model."""

    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return LSTMClassifier(
            input_size=258,
            num_classes=90,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
        )

    def test_forward_pass(self, model):
        """Test that forward pass produces correct output shape."""
        batch_size = 4
        seq_length = 30
        input_size = 258

        x = torch.randn(batch_size, seq_length, input_size)
        output = model(x)

        assert output.shape == (batch_size, 90)

    def test_forward_pass_single_sample(self, model):
        """Test forward pass with batch size of 1."""
        x = torch.randn(1, 30, 258)
        output = model(x)

        assert output.shape == (1, 90)

    def test_predict(self, model):
        """Test prediction method."""
        x = torch.randn(4, 30, 258)
        preds, probs = model.predict(x)

        assert preds.shape == (4,)
        assert probs.shape == (4, 90)
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)

    def test_count_parameters(self, model):
        """Test parameter counting."""
        num_params = model.count_parameters()
        assert num_params > 0
        assert isinstance(num_params, int)

    def test_get_config(self, model):
        """Test configuration retrieval."""
        config = model.get_config()

        assert config["input_size"] == 258
        assert config["num_classes"] == 90
        assert config["hidden_size"] == 128
        assert config["num_layers"] == 2

    def test_save_and_load(self, model, tmp_path):
        """Test model saving and loading."""
        save_path = tmp_path / "test_model.pt"
        model.save(save_path, epoch=5)

        loaded_model, checkpoint = LSTMClassifier.load(save_path)

        assert checkpoint["epoch"] == 5
        assert loaded_model.input_size == model.input_size
        assert loaded_model.num_classes == model.num_classes

        # Test that loaded model produces same output
        x = torch.randn(2, 30, 258)
        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            original_output = model(x)
            loaded_output = loaded_model(x)

        assert torch.allclose(original_output, loaded_output)


class TestBiLSTMWithAttention:
    """Tests for BiLSTMWithAttention model."""

    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return BiLSTMWithAttention(
            input_size=258,
            num_classes=90,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            attention_heads=4,
        )

    def test_forward_pass(self, model):
        """Test that forward pass produces correct output shape."""
        batch_size = 4
        x = torch.randn(batch_size, 30, 258)
        output = model(x)

        assert output.shape == (batch_size, 90)

    def test_attention_mechanism(self, model):
        """Test that attention produces different weights for different inputs."""
        x1 = torch.randn(2, 30, 258)
        x2 = torch.randn(2, 30, 258)

        model.eval()
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)

        # Outputs should be different for different inputs
        assert not torch.allclose(out1, out2)


class TestLSTMWithPooling:
    """Tests for LSTMWithPooling model."""

    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return LSTMWithPooling(
            input_size=258,
            num_classes=90,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
        )

    def test_forward_pass(self, model):
        """Test that forward pass produces correct output shape."""
        batch_size = 4
        x = torch.randn(batch_size, 30, 258)
        output = model(x)

        assert output.shape == (batch_size, 90)

    def test_pooling_features(self, model):
        """Test that model handles variable sequence lengths."""
        # The model should work with different sequence lengths
        x_short = torch.randn(2, 15, 258)
        x_long = torch.randn(2, 60, 258)

        model.eval()
        with torch.no_grad():
            out_short = model(x_short)
            out_long = model(x_long)

        assert out_short.shape == (2, 90)
        assert out_long.shape == (2, 90)


class TestBidirectionalLSTM:
    """Tests for bidirectional LSTM variant."""

    @pytest.fixture
    def model(self):
        """Create a bidirectional LSTM model."""
        return LSTMClassifier(
            input_size=258,
            num_classes=90,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
        )

    def test_forward_pass(self, model):
        """Test that bidirectional model produces correct output."""
        x = torch.randn(4, 30, 258)
        output = model(x)

        assert output.shape == (4, 90)

    def test_bidirectional_config(self, model):
        """Test that bidirectional flag is in config."""
        config = model.get_config()
        assert config["bidirectional"] is True


@pytest.mark.parametrize("model_class", [
    LSTMClassifier,
    BiLSTMWithAttention,
    LSTMWithPooling,
])
def test_gradient_flow(model_class):
    """Test that gradients flow through all models."""
    model = model_class(input_size=258, num_classes=90, hidden_size=64, num_layers=1)
    model.train()

    x = torch.randn(2, 30, 258, requires_grad=True)
    y = torch.randint(0, 90, (2,))

    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, y)
    loss.backward()

    # Check that gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


@pytest.mark.parametrize("device", ["cpu"])
def test_device_transfer(device):
    """Test that models can be transferred to different devices."""
    model = LSTMClassifier(input_size=258, num_classes=90)
    model = model.to(device)

    x = torch.randn(2, 30, 258).to(device)
    output = model(x)

    assert output.device.type == device

import pytest

torch = pytest.importorskip("torch", reason="模型测试需要项目既有的 PyTorch 环境")

from src.models.lstm_trajectory import LSTMTrajectoryPredictor
from src.models.transformer_trajectory import TransformerTrajectoryPredictor


def test_lstm_forward_shape() -> None:
    model = LSTMTrajectoryPredictor()
    assert model(torch.randn(4, 20, 2)).shape == (4, 10, 2)


def test_transformer_forward_shape() -> None:
    model = TransformerTrajectoryPredictor()
    assert model(torch.randn(4, 20, 2)).shape == (4, 10, 2)

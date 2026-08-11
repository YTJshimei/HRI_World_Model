import pytest

torch = pytest.importorskip("torch", reason="模型测试需要项目既有的 PyTorch 环境")

from src.models.lstm_trajectory import LSTMTrajectoryPredictor
from src.models.normalized_trajectory import NormalizedTrajectoryPredictor, local_coordinates
from src.models.residual_transformer import (
    ResidualTransformer,
    constant_velocity_prediction,
)


def test_local_coordinates_use_last_history_frame_as_origin() -> None:
    history = torch.randn(3, 20, 2)
    future = torch.randn(3, 10, 2)
    local_history, local_future, origin, scale = local_coordinates(history, future)
    torch.testing.assert_close(local_history[:, -1], torch.zeros(3, 2))
    torch.testing.assert_close(local_future * scale + origin, future)


def test_normalized_predictor_returns_world_shape() -> None:
    model = NormalizedTrajectoryPredictor(LSTMTrajectoryPredictor())
    assert model(torch.randn(4, 20, 2)).shape == (4, 10, 2)
    assert model.forward_relative(torch.randn(4, 20, 2)).shape == (4, 10, 2)


def test_residual_transformer_initially_equals_constant_velocity() -> None:
    history = torch.randn(4, 20, 2)
    model = ResidualTransformer()
    model.eval()
    with torch.inference_mode():
        actual = model(history)
        expected = constant_velocity_prediction(history, 10)
    torch.testing.assert_close(actual, expected)

import numpy as np
import pytest

from src.evaluation.real_trajectory_metrics import real_trajectory_metrics


def test_real_metrics_velocity_and_heading() -> None:
    target = np.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], dtype=np.float32)
    prediction = np.array([[[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]], dtype=np.float32)
    metrics = real_trajectory_metrics(prediction, target, 1.0, 0.25)
    assert metrics["velocity_error"] == pytest.approx(2**0.5)
    assert metrics["heading_error_rad"] == pytest.approx(np.pi / 2)
    assert metrics["inference_latency_ms_per_sample"] == pytest.approx(0.25)

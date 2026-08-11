import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="诊断测试需要项目既有的 PyTorch 环境")

from src.evaluation.trajectory_diagnostics import (
    ConstantVelocityModule,
    benchmark_inference,
    metrics_by_type,
)


def test_metrics_by_type() -> None:
    target = np.zeros((4, 2, 2), dtype=np.float32)
    prediction = np.zeros_like(target)
    prediction[:2] = np.array([3.0, 4.0], dtype=np.float32)
    labels = np.array(["straight", "straight", "left_turn", "left_turn"])
    result = metrics_by_type(prediction, target, labels)
    assert result["straight"]["ADE"] == pytest.approx(5.0)
    assert result["straight"]["FDE"] == pytest.approx(5.0)
    assert result["left_turn"]["ADE"] == pytest.approx(0.0)


def test_benchmark_enforces_minimum_repetitions() -> None:
    with pytest.raises(ValueError, match="warmup"):
        benchmark_inference(
            ConstantVelocityModule(10), torch.randn(2, 20, 2), torch.device("cpu"), 49, 200
        )

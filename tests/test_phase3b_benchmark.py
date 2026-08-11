import pytest

torch = pytest.importorskip("torch", reason="Benchmark contract requires PyTorch")

from src.evaluation.skeleton_benchmark import benchmark_skeleton_model


def test_phase3b_benchmark_enforces_strict_protocol() -> None:
    model = torch.nn.Identity()
    sample = (torch.zeros(1), torch.zeros(1), torch.zeros(1))
    with pytest.raises(ValueError, match="warmup"):
        benchmark_skeleton_model(
            model, sample, torch.device("cpu"), torch, warmup=49, repetitions=200
        )
    with pytest.raises(ValueError, match="repetitions"):
        benchmark_skeleton_model(
            model, sample, torch.device("cpu"), torch, warmup=50, repetitions=199
        )

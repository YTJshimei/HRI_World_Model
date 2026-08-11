import numpy as np
import pytest

from src.evaluation.trajectory_metrics import ade_fde


def test_ade_fde() -> None:
    target = np.zeros((1, 2, 2), dtype=np.float32)
    prediction = np.array([[[3.0, 4.0], [0.0, 2.0]]], dtype=np.float32)
    ade, fde = ade_fde(prediction, target)
    assert ade == pytest.approx(3.5)
    assert fde == pytest.approx(2.0)

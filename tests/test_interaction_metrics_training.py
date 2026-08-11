import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Interaction training tests require PyTorch")

from src.data.synthetic_interaction import generate_interaction_split
from src.evaluation.interaction_metrics import interaction_metrics
from src.models.action_conditioned_world_models import InteractionPrediction
from src.training.train_interaction import InteractionLossWeights, interaction_loss


def test_perfect_counterfactual_prediction_has_perfect_new_metrics() -> None:
    split = generate_interaction_split(8, 33, "metric_fixture", noise_std=0.0, occlusion_rate=0.0)
    metrics = interaction_metrics(
        split.future_by_action, split.natural_future, split, sample_rate_hz=10.0
    )
    assert metrics["Global_MPJPE"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["Action_Effect_Error"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["Action_Direction_Accuracy"] == pytest.approx(1.0)
    assert metrics["Counterfactual_Ranking_Accuracy"] == pytest.approx(1.0)
    assert metrics["Human_Robot_Distance_Error"] == pytest.approx(0.0, abs=1e-7)


def test_action_agnostic_prediction_has_zero_sensitivity() -> None:
    split = generate_interaction_split(6, 37, "metric_fixture", noise_std=0.0)
    prediction = np.broadcast_to(
        split.natural_future[:, None], split.future_by_action.shape
    ).copy()
    metrics = interaction_metrics(prediction, split.natural_future, split)
    assert metrics["Action_Sensitivity"] == pytest.approx(0.0)
    assert metrics["Action_Effect_Error"] > 0.0


def test_interaction_loss_is_zero_for_perfect_prediction() -> None:
    split = generate_interaction_split(3, 41, "loss_fixture", noise_std=0.0)
    output = InteractionPrediction(
        torch.from_numpy(split.future_by_action),
        torch.from_numpy(split.natural_future),
    )
    total, components = interaction_loss(
        output,
        torch.from_numpy(split.future_by_action),
        torch.from_numpy(split.natural_future),
        torch.from_numpy(split.action_supervision_mask),
        InteractionLossWeights(),
    )
    assert total.item() == pytest.approx(0.0)
    assert all(value.item() == pytest.approx(0.0) for value in components.values())

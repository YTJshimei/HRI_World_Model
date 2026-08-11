import pytest

torch = pytest.importorskip("torch", reason="Phase 4 model tests require PyTorch")

from src.models.action_conditioned_world_models import (
    ActionAgnosticHumanModel,
    ActionConditionedLSTM,
    ActionConditionedResidualModel,
    ActionConditionedRootPoseModel,
)


def inputs(batch=2):
    history = torch.randn(batch, 20, 17, 3)
    robot = torch.randn(batch, 20, 7)
    actions = torch.tensor([[0, 1, 2, 3, 4]]).expand(batch, -1)
    confidence = torch.ones(batch, 20, 17)
    visibility = torch.ones(batch, 20, 17, dtype=torch.bool)
    profiles = torch.tensor([0, 4])[:batch]
    return history, robot, actions, confidence, visibility, profiles


@pytest.mark.parametrize(
    "model",
    (
        ActionAgnosticHumanModel(),
        ActionConditionedLSTM(),
        ActionConditionedRootPoseModel(),
        ActionConditionedResidualModel(),
    ),
)
def test_phase4_models_forward_all_actions_shape(model) -> None:
    model.eval()
    with torch.inference_mode():
        output = model(*inputs())
    assert output.future_by_action.shape == (2, 5, 10, 17, 3)
    assert output.natural_future.shape == (2, 10, 17, 3)


@pytest.mark.parametrize(
    "model", (ActionConditionedLSTM(), ActionConditionedRootPoseModel(), ActionConditionedResidualModel())
)
def test_action_conditioning_switch_and_action_order_equivariance(model) -> None:
    model.eval()
    values = inputs()
    with torch.inference_mode():
        disabled = model(*values, action_conditioning=False).future_by_action
        enabled = model(*values, action_conditioning=True).future_by_action
    torch.testing.assert_close(disabled, disabled[:, :1].expand_as(disabled))
    assert (enabled[:, 1:] - enabled[:, :1]).abs().max().item() > 1e-7

    permutation = torch.tensor([4, 2, 0, 3, 1])
    permuted_values = list(values)
    permuted_values[2] = values[2][:, permutation]
    with torch.inference_mode():
        permuted = model(*permuted_values).future_by_action
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(enabled, permuted[:, inverse], atol=1e-6, rtol=1e-6)


def test_w0_ignores_all_actions() -> None:
    model = ActionAgnosticHumanModel().eval()
    with torch.inference_mode():
        prediction = model(*inputs()).future_by_action
    torch.testing.assert_close(prediction, prediction[:, :1].expand_as(prediction))


def test_human_context_is_encoded_once_for_five_action_queries() -> None:
    model = ActionConditionedLSTM().eval()
    calls = []
    handle = model.human_encoder.register_forward_hook(lambda *unused: calls.append(1))
    with torch.inference_mode():
        model(*inputs())
    handle.remove()
    assert len(calls) == 1


def test_w3_keep_prediction_is_exactly_natural() -> None:
    model = ActionConditionedResidualModel().eval()
    with torch.inference_mode():
        output = model(*inputs())
    torch.testing.assert_close(output.future_by_action[:, 0], output.natural_future)


def test_action_models_remain_compact() -> None:
    for model in (
        ActionConditionedLSTM(), ActionConditionedRootPoseModel(), ActionConditionedResidualModel()
    ):
        assert sum(parameter.numel() for parameter in model.parameters()) < 350_000

import pytest

torch = pytest.importorskip("torch", reason="Skeleton model tests require PyTorch")

from src.models.hybrid_root_pose import HybridRootPoseModel
from src.models.decoupled_root_pose import DecoupledRootPoseModel
from src.models.skeleton_baselines import (
    JointConstantVelocity,
    joint_constant_velocity_prediction,
    joint_constant_velocity_prediction_reference,
)
from src.models.skeleton_lstm import SkeletonLSTM
from src.models.skeleton_transformer import (
    ResidualSkeletonTransformer,
    SpatialTemporalSkeletonTransformer,
)


def inputs():
    history = torch.randn(2, 20, 17, 3)
    confidence = torch.ones(2, 20, 17)
    visibility = torch.ones(2, 20, 17, dtype=torch.bool)
    visibility[:, -2:, 9] = False
    confidence[~visibility] = 0.0
    return history, confidence, visibility


@pytest.mark.parametrize(
    "model",
    [
        SkeletonLSTM(),
        SpatialTemporalSkeletonTransformer(),
        ResidualSkeletonTransformer(),
        HybridRootPoseModel(),
        DecoupledRootPoseModel(),
    ],
)
def test_s1_s2_s3_s4_forward_shape(model) -> None:
    model.eval()
    with torch.inference_mode():
        assert model(*inputs()).shape == (2, 10, 17, 3)


def test_s2_s3_have_matching_parameter_counts() -> None:
    s2 = SpatialTemporalSkeletonTransformer()
    s3 = ResidualSkeletonTransformer()
    assert sum(value.numel() for value in s2.parameters()) == sum(
        value.numel() for value in s3.parameters()
    )


def test_missing_joint_values_cannot_leak_through_mask() -> None:
    history, confidence, visibility = inputs()
    model = SkeletonLSTM().eval()
    modified = history.clone()
    modified[~visibility] = 1_000_000.0
    with torch.inference_mode():
        original_prediction = model(history, confidence, visibility)
        modified_prediction = model(modified, confidence, visibility)
    torch.testing.assert_close(original_prediction, modified_prediction)


def test_s3_initial_prediction_equals_s0_prior() -> None:
    history, confidence, visibility = inputs()
    s0 = JointConstantVelocity()
    s3 = ResidualSkeletonTransformer().eval()
    with torch.inference_mode():
        torch.testing.assert_close(
            s3(history, confidence, visibility),
            s0(history, confidence, visibility),
        )


def test_vectorized_s0_is_equivalent_to_frozen_reference() -> None:
    generator = torch.Generator().manual_seed(3407)
    history = torch.randn(5, 20, 17, 3, generator=generator)
    visibility = torch.rand(5, 20, 17, generator=generator) > 0.25
    visibility[0] = False
    visibility[1, :, 11:13] = False
    visibility[2, :-1] = False
    reference = joint_constant_velocity_prediction_reference(history, visibility, 10)
    vectorized = joint_constant_velocity_prediction(history, visibility, 10)
    assert (reference - vectorized).abs().max().item() <= 1e-6


def test_s4_components_reconstruct_global_and_have_zero_pelvis() -> None:
    history, confidence, visibility = inputs()
    model = HybridRootPoseModel().eval()
    with torch.inference_mode():
        root, local = model.forward_components(history, confidence, visibility)
        prediction = model(history, confidence, visibility)
    assert root.shape == (2, 10, 3)
    assert local.shape == (2, 10, 17, 3)
    torch.testing.assert_close(prediction, root[..., None, :] + local)
    torch.testing.assert_close(local[..., [11, 12], :].mean(dim=-2), torch.zeros_like(root), atol=1e-6, rtol=0)


def test_s4_parameter_budget() -> None:
    assert sum(parameter.numel() for parameter in HybridRootPoseModel().parameters()) < 300_000


def test_s4b_parameter_budget_and_component_shapes() -> None:
    model = DecoupledRootPoseModel().eval()
    with torch.inference_mode():
        root, local = model.forward_components(*inputs())
    assert root.shape == (2, 10, 3)
    assert local.shape == (2, 10, 17, 3)
    assert sum(parameter.numel() for parameter in model.parameters()) < 300_000

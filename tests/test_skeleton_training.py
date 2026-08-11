from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="Skeleton training test requires PyTorch")

from src.training.train_skeleton import (
    SkeletonLossWeights,
    skeleton_loss_components,
    train_skeleton_model,
    weighted_skeleton_loss,
)


class TinySkeletonModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1, 1, 1, 3))

    def forward(self, history, confidence, visibility):
        del confidence, visibility
        return history[:, -1:, :, :] + self.bias


def test_skeleton_checkpoint_selected_only_from_validation(tmp_path: Path) -> None:
    history = torch.zeros(4, 2, 17, 3)
    future = torch.ones(4, 1, 17, 3)
    confidence = torch.ones(4, 2, 17)
    visibility = torch.ones(4, 2, 17, dtype=torch.bool)
    actions = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            history, future, confidence, visibility, actions
        ),
        batch_size=2,
    )
    checkpoint = tmp_path / "best_skeleton.pt"
    model = TinySkeletonModel()
    result = train_skeleton_model(
        model,
        loader,
        loader,
        torch.device("cpu"),
        epochs=2,
        checkpoint_path=checkpoint,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint.is_file()
    assert result.best_epoch in (1, 2)
    assert saved["best_validation_MPJPE"] == pytest.approx(
        result.best_validation_mpjpe
    )


def test_structural_loss_components_are_zero_for_exact_prediction() -> None:
    target = torch.randn(2, 4, 17, 3)
    components = skeleton_loss_components(target, target)
    assert all(value.item() == pytest.approx(0.0) for value in components.values())


def test_weighted_loss_does_not_add_parameters_or_require_test_data() -> None:
    target = torch.zeros(1, 3, 17, 3)
    prediction = target.clone()
    prediction[..., 0] = 0.2
    weights = SkeletonLossWeights(
        global_loss=1.0, root_loss=1.0, local_loss=1.0, bone_loss=0.1, velocity_loss=0.01
    )
    loss, components = weighted_skeleton_loss(prediction, target, weights)
    assert loss.item() >= components["global"].item()

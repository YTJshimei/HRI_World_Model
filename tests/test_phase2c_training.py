from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="训练测试需要项目既有的 PyTorch 环境")

from src.training.train_trajectory_phase2c import train_with_best_validation_checkpoint


def test_best_validation_checkpoint_is_saved_and_restored(tmp_path: Path) -> None:
    history = torch.randn(12, 20, 2)
    future = torch.randn(12, 10, 2)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(history, future), batch_size=4
    )
    model = torch.nn.Sequential(
        torch.nn.Flatten(), torch.nn.Linear(40, 20), torch.nn.Unflatten(1, (10, 2))
    )
    checkpoint = tmp_path / "best.pt"
    result = train_with_best_validation_checkpoint(
        model, loader, loader, torch.device("cpu"), 2, checkpoint
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint.is_file()
    assert result.best_epoch in (1, 2)
    assert saved["best_validation_ADE"] == pytest.approx(result.best_validation_ade)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved["model_state_dict"][name])

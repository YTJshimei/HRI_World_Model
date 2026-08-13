"""Contracts for Phase 5B-v3-R1D runtime-conditioned episode offsets."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.models.runtime_conditioned_episode_offset import RuntimeConditionedEpisodeOffset

ROOT = Path(__file__).resolve().parents[1]


def test_offset_head_is_exactly_linear_128_to_one():
    torch = pytest.importorskip("torch")
    head = RuntimeConditionedEpisodeOffset()
    assert sum(parameter.numel() for parameter in head.parameters()) == 129
    assert head.architecture_audit()["layers"] == ["Linear(128,1)"]
    assert head(torch.randn(4, 128)).shape == (4,)


def test_one_episode_offset_preserves_ranking_and_pairwise_differences():
    values = np.asarray((3.0, -2.0, 4.0, 1.0, 8.0, 0.5))
    transformed = values + 1.234
    assert np.array_equal(np.argsort(-values, kind="stable"), np.argsort(-transformed, kind="stable"))
    assert np.allclose(values[:, None]-values[None], transformed[:, None]-transformed[None], atol=1e-12)


def test_offset_head_rejects_candidate_or_wrong_width_shapes():
    torch = pytest.importorskip("torch")
    head = RuntimeConditionedEpisodeOffset()
    with pytest.raises(ValueError, match="E,128"):
        head(torch.randn(2, 6, 128))
    with pytest.raises(ValueError, match="128"):
        head(torch.randn(2, 127))


def test_r1d_frozen_input_checksums():
    expected = {
        "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json": "ac3a620b1fb254f1a89114f3b0daa1e3ed7dc6a570cfcf083c23355a70ee542a",
        "results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv": "ae8dc040459d9dcfaabdd480309ea10ea535ed44643484e12626ad93c24939f1",
        "results_dev/phase5b_v3_r1a_runtime_anchor_realign/runtime_anchor_map.csv": "88bf047e0a26a9e79ca946dbe9ec277b24ce1e84fbaf956c4378826c29d3eb0f",
        "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt": "dbeacc97cba71515591586ce8343a5747dbb8c7034d4dbd250c3b4a58318a0ff",
        "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt": "2dea4137e30324daaa0344f5c5770d4758bcb563f3016ccb4d61bfa38c355b6d",
    }
    for relative, expected_sha in expected.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected_sha


def test_selector_is_mae_then_sign_then_earlier_epoch():
    from scripts import run_phase5b_v3_r1d_runtime_conditioned_offset as stage
    rows = [
        {"epoch": 1, "Benefit_MAE": 1.0, "safe_beneficial_sign_accuracy": .5},
        {"epoch": 2, "Benefit_MAE": .9, "safe_beneficial_sign_accuracy": .4},
        {"epoch": 3, "Benefit_MAE": .9, "safe_beneficial_sign_accuracy": .6},
        {"epoch": 4, "Benefit_MAE": .9, "safe_beneficial_sign_accuracy": .6},
    ]
    assert stage.select_epoch(rows)["epoch"] == 3


def test_r1d_has_no_threshold_or_decision_chain():
    from scripts import run_phase5b_v3_r1d_runtime_conditioned_offset as stage
    source = inspect.getsource(stage.main)
    assert "select_threshold" not in source
    assert "arbitrate" not in source
    assert "decision_evaluation" not in source
    assert 'build_development_split("test"' not in source
    assert "clip(" not in source and "temperature" not in source.lower()
    assert stage.TEST_READS == 0

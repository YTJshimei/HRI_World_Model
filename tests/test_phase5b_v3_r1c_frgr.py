"""Contracts for Phase 5B-v3-R1C frozen runtime-generic re-anchoring."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.frozen_runtime_generic_reanchor import FRGR_ALPHA, frozen_runtime_generic_reanchor

ROOT = Path(__file__).resolve().parents[1]


def test_frgr_generic_is_exact_zero_and_alpha_is_one():
    values = np.asarray((2.0, -1.0, 4.0, 8.0, 3.0))
    episodes = ("a", "a", "a", "b", "b")
    transformed, offsets = frozen_runtime_generic_reanchor(values, episodes, {"a": 1, "b": 4})
    assert FRGR_ALPHA == 1.0
    assert transformed[1] == 0.0 and transformed[4] == 0.0
    assert np.array_equal(offsets, (-1.0, -1.0, -1.0, 3.0, 3.0))


def test_frgr_preserves_pairwise_differences_and_ranking():
    values = np.asarray((3.0, -2.0, 1.0, 7.0, 9.0, 8.0))
    episodes = ("a", "a", "a", "b", "b", "b")
    transformed, _ = frozen_runtime_generic_reanchor(values, episodes, {"a": 2, "b": 3})
    for indices in ((0, 1, 2), (3, 4, 5)):
        old, new = values[list(indices)], transformed[list(indices)]
        assert np.allclose(old[:, None] - old[None], new[:, None] - new[None], atol=1e-12)
        assert np.array_equal(np.argsort(-old, kind="stable"), np.argsort(-new, kind="stable"))


def test_frgr_rejects_unknown_or_cross_episode_anchor():
    with pytest.raises(ValueError, match="missing"):
        frozen_runtime_generic_reanchor(np.ones(2), ("a", "b"), {"a": 0})
    with pytest.raises(ValueError, match="crosses"):
        frozen_runtime_generic_reanchor(np.ones(2), ("a", "b"), {"a": 1, "b": 1})


def test_frozen_contract_checksums_are_unchanged():
    expected = {
        "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json": "ac3a620b1fb254f1a89114f3b0daa1e3ed7dc6a570cfcf083c23355a70ee542a",
        "results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv": "ae8dc040459d9dcfaabdd480309ea10ea535ed44643484e12626ad93c24939f1",
        "results_dev/phase5b_v3_r1a_runtime_anchor_realign/runtime_anchor_map.csv": "88bf047e0a26a9e79ca946dbe9ec277b24ce1e84fbaf956c4378826c29d3eb0f",
        "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt": "dbeacc97cba71515591586ce8343a5747dbb8c7034d4dbd250c3b4a58318a0ff",
        "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt": "2dea4137e30324daaa0344f5c5770d4758bcb563f3016ccb4d61bfa38c355b6d",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_r1c_main_contains_no_training_or_decision_chain():
    from scripts import run_phase5b_v3_r1c_frozen_runtime_generic_reanchor as stage
    source = inspect.getsource(stage.main)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "select_cracs" not in source
    assert "select_threshold" not in source
    assert "arbitrate" not in source
    assert 'build_development_split("test"' not in source
    assert stage.TEST_READS == 0

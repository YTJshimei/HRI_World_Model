"""Pre-training stop-gate tests for the Phase 5B-v3-R1 GARA proposal."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from scripts import audit_phase5b_v3_r1_gara_anchor as audit
from src.data import adverse_response_dataset as adverse
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.multimodal import phase5b_v2_dataset as runtime_bridge

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_manifest_and_model_checkpoint_hashes():
    paths = {
        ROOT / "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json": audit.EXPECTED_MANIFEST_SHA,
        ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt": audit.EXPECTED_R1_SHA,
        ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt": audit.EXPECTED_HARM_SHA,
    }
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == expected for path, expected in paths.items())


def test_gt_generic_benefit_is_exact_zero():
    episodes = build_development_split("train", 3, GENERATOR_SEED, RISK_SEED)
    assert all(episode.candidates[episode.generic_action_index].benefit == 0.0 for episode in episodes)


def test_target_anchor_uses_label_side_gt_natural_future():
    generator = inspect.getsource(adverse.build_development_split)
    bridge = inspect.getsource(runtime_bridge.runtime_constant_velocity_prior)
    assert "base.natural_future[index]" in generator
    assert "history[-1]" in bridge and "history[-2]" in bridge
    assert "natural_future" not in tuple(inspect.signature(runtime_bridge.runtime_constant_velocity_prior).parameters)


def test_generic_anchor_excludes_hold_and_uses_first_minimum_tie_break():
    assert adverse.ACTION_IDS == (0, 1, 2, 3, 4)
    source = inspect.getsource(adverse.build_development_split)
    assert "np.argmin(generic_costs.total)" in source


def test_preflight_cannot_train_or_run_decision_chain():
    source = inspect.getsource(audit.main)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "select_threshold" not in source
    assert "decision_evaluation" not in source
    assert "build_development_split(\"test\"" not in source

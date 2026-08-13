"""Contract tests for the Phase 5B-v3-R1A derived Benefit Target v2."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np

from scripts import audit_phase5b_v3_r1a_runtime_anchor_realign as audit
from src.data.adverse_response_dataset import ACTION_IDS, GENERATOR_SEED, RISK_SEED, build_development_split
from src.multimodal.phase5b_v2_dataset import (
    build_v2_temporal_samples,
    replay_runtime_generic_policy,
    runtime_constant_velocity_prior,
)

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_v3_is_unchanged_and_test_is_not_built():
    manifest = ROOT / "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == audit.EXPECTED_MANIFEST_SHA
    assert "build_development_split(\"test\"" not in inspect.getsource(audit.main)


def test_runtime_anchor_signature_excludes_all_gt_and_identity_inputs():
    parameters = tuple(inspect.signature(replay_runtime_generic_policy).parameters)
    assert parameters == ("human_history", "robot_history", "confidence", "visibility", "target_follow_distance")
    forbidden = ("future", "cost", "benefit", "harm", "unsafe", "profile", "oracle")
    assert not any(token in name.lower() for name in parameters for token in forbidden)


def test_runtime_anchor_is_a0_a4_and_hold_is_excluded():
    episode = build_development_split("train", 1, GENERATOR_SEED, RISK_SEED)[0]
    replay = replay_runtime_generic_policy(
        episode.human_history, episode.robot_history, episode.confidence,
        episode.visibility, episode.target_follow_distance,
    )
    assert tuple(replay.action_ids) == ACTION_IDS == (0, 1, 2, 3, 4)
    assert replay.anchor_action_id in ACTION_IDS and replay.gt_read_count == 0


def test_runtime_prior_is_exact_frozen_history_constant_velocity():
    episode = build_development_split("train", 1, GENERATOR_SEED, RISK_SEED)[0]
    expected = episode.human_history[-1][None] + np.arange(1, 11, dtype=np.float32)[:, None, None] * (
        episode.human_history[-1] - episode.human_history[-2]
    )[None]
    assert np.array_equal(runtime_constant_velocity_prior(episode.human_history), expected)


def test_extracted_runtime_replay_preserves_v2_bridge_outputs():
    episode = build_development_split("train", 1, GENERATOR_SEED, RISK_SEED)[0]
    samples = build_v2_temporal_samples([episode])
    replay = replay_runtime_generic_policy(
        episode.human_history, episode.robot_history, episode.confidence,
        episode.visibility, episode.target_follow_distance,
    )
    from src.multimodal.temporal_dataset import _candidate_future
    for sample, simulation in zip(samples, replay.simulations):
        assert np.array_equal(sample.streams["candidate_robot_future"], _candidate_future(simulation.robot_future_xy, episode.robot_history))


def test_runtime_anchor_id_is_frozen_before_label_side_cost_read():
    source = inspect.getsource(audit.derive_split)
    selection = source.index("runtime_anchor_action = int(replay.anchor_action_id)")
    gt_read = source.index("runtime_anchor_gt_cost = float(episode.gt_costs")
    assert selection < gt_read


def test_v2_anchor_is_zero_constant_shift_pairwise_and_rank_invariant():
    episode = build_development_split("train", 1, GENERATOR_SEED, RISK_SEED)[0]
    replay = replay_runtime_generic_policy(
        episode.human_history, episode.robot_history, episode.confidence,
        episode.visibility, episode.target_follow_distance,
    )
    anchor = int(np.flatnonzero(np.asarray(ACTION_IDS) == replay.anchor_action_id)[0])
    old = np.asarray([candidate.benefit for candidate in episode.candidates], float)
    new = float(episode.gt_costs[anchor]) - np.asarray(episode.gt_costs, float)
    delta = new - old
    assert new[anchor] == 0.0
    assert float(delta.max() - delta.min()) <= audit.TOLERANCE
    np.testing.assert_allclose(old[:, None] - old[None], new[:, None] - new[None], atol=audit.TOLERANCE, rtol=0)
    assert audit.rank_signature(ACTION_IDS, old) == audit.rank_signature(ACTION_IDS, new)


def test_audit_forbids_training_thresholds_and_decision_execution():
    source = inspect.getsource(audit.main)
    assert ".backward(" not in source and "torch.optim" not in source
    assert "select_threshold" not in source and "decision_evaluation" not in source
    assert "arbitrate" not in source

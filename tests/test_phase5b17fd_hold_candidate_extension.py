import inspect
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.adverse_response_dataset import (
    GENERATOR_SEED, RISK_SEED, POPULATION_PROFILE, build_development_split,
)
from src.data.adverse_response_protocol import derive_adverse_response_events
from src.data.hold_candidate import (
    HOLD_ANGULAR_DECELERATION_LIMIT_RADPS2,
    HOLD_LINEAR_DECELERATION_LIMIT_MPS2,
    build_hold_candidate_outcome, hold_robot_rollout,
    simulate_hold_interaction_future,
)
from src.data.robot_action_schema import (
    HOLD_ACTION_ID, RobotAction, RobotActionV3, candidate_action_vector_v3,
)
from src.data.synthetic_interaction import PROFILE_BY_ID
from src.multimodal.phase5b_v3_dataset import build_hold_temporal_sample_v3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_SHA = "b50cfe7c7077759d4fd85c78ab12cf0d54769a7bbc3bcee51b57e81eaea6604e"


def _robot(speed=1.0, angular=.2):
    result = np.zeros((20, 7), np.float32)
    result[-1, :5] = (1.0, 2.0, .3, speed, angular)
    result[-1, 5:] = (1.4, .1)
    return result


def test_hold_has_independent_action_id() -> None:
    assert HOLD_ACTION_ID == int(RobotActionV3.HOLD) == 7
    assert HOLD_ACTION_ID not in {int(action) for action in RobotAction}
    assert HOLD_ACTION_ID != int(RobotAction.KEEP)


def test_v3_action_vector_is_independent_twelve_dimensional_encoding() -> None:
    hold = candidate_action_vector_v3(HOLD_ACTION_ID)
    keep = candidate_action_vector_v3(RobotActionV3.KEEP)
    assert hold.shape == keep.shape == (12,)
    assert hold[HOLD_ACTION_ID] == 1 and keep[0] == 1
    assert not np.array_equal(hold, keep)


def test_hold_respects_linear_and_angular_deceleration_limits() -> None:
    rollout = hold_robot_rollout(_robot(), future_frames=10)
    velocity = np.r_[1.0, rollout.states[:, 3]]
    angular = np.r_[.2, rollout.states[:, 4]]
    assert np.max(np.abs(np.diff(velocity))) <= HOLD_LINEAR_DECELERATION_LIMIT_MPS2 / 10 + 1e-7
    assert np.max(np.abs(np.diff(angular))) <= HOLD_ANGULAR_DECELERATION_LIMIT_RADPS2 / 10 + 1e-7
    assert np.all(np.diff(velocity) <= 1e-7)
    assert np.all(np.diff(angular) <= 1e-7)


def test_hold_remains_zero_after_stopping() -> None:
    rollout = hold_robot_rollout(_robot(speed=.06, angular=.02), future_frames=10)
    v, w = rollout.states[:, 3], rollout.states[:, 4]
    first_v_zero, first_w_zero = np.flatnonzero(v == 0)[0], np.flatnonzero(w == 0)[0]
    assert np.all(v[first_v_zero:] == 0)
    assert np.all(w[first_w_zero:] == 0)
    expected = np.repeat(rollout.states[first_v_zero:first_v_zero + 1, :2], len(v) - first_v_zero - 1, axis=0)
    np.testing.assert_allclose(rollout.states[first_v_zero + 1:, :2], expected, atol=1e-7)


@pytest.fixture(scope="module")
def episode_and_outcome():
    episode = build_development_split("train", 1, GENERATOR_SEED, RISK_SEED)[0]
    outcome = build_hold_candidate_outcome(
        episode, POPULATION_PROFILE, PROFILE_BY_ID[episode.profile_id]
    )
    return episode, outcome


def test_hold_robot_human_rollout_and_cost_exist(episode_and_outcome) -> None:
    episode, outcome = episode_and_outcome
    assert outcome.gt_simulation.robot_future_state.shape == (10, 5)
    assert outcome.gt_simulation.future_global.shape == (10, 17, 3)
    assert np.isfinite((outcome.gt_total_cost, outcome.generic_total_cost, outcome.benefit, outcome.regret)).all()
    sample = build_hold_temporal_sample_v3(episode, outcome)
    assert sample.candidate_robot_future.shape == (10, 5)
    assert sample.candidate_action.shape == (12,)


def test_hold_harm_is_derived_from_rollout(episode_and_outcome) -> None:
    episode, outcome = episode_and_outcome
    replay = derive_adverse_response_events(
        episode.human_history, episode.natural_future,
        outcome.gt_simulation.future_global,
    )
    assert outcome.events == replay
    assert outcome.harm_v2 == (outcome.gt_unsafe or replay.adverse_human_kinematic_response)


def test_hold_generator_has_no_hardcoded_safe_or_fixed_cost_shortcut() -> None:
    source = inspect.getsource(build_hold_candidate_outcome)
    assert "harm = bool(unsafe or events.adverse_human_kinematic_response)" in source
    assert "if action" not in source.lower()
    assert "fixed" not in source.lower()
    assert "bonus" not in source.lower()


def test_hold_human_future_uses_context_profile_and_risk(episode_and_outcome) -> None:
    episode, outcome = episode_and_outcome
    other = simulate_hold_interaction_future(
        episode.human_history, episode.natural_future, episode.robot_history,
        PROFILE_BY_ID[0], None,
    )
    assert not np.array_equal(outcome.gt_simulation.future_global, other.future_global)


def test_hold_sample_rejects_test_materialization(episode_and_outcome) -> None:
    episode, outcome = episode_and_outcome
    from dataclasses import replace
    with pytest.raises(ValueError, match="TRAIN/VALIDATION"):
        build_hold_temporal_sample_v3(replace(episode, split="test"), outcome)


def test_manifest_v2_is_unchanged() -> None:
    path = PROJECT_ROOT / "results_dev/phase5b17c_adverse_response_expansion/phase5b_manifest_v2.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == V2_SHA


def test_generated_manifest_v3_keeps_episode_branches_together_and_test_sealed() -> None:
    path = PROJECT_ROOT / "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json"
    if not path.exists():
        pytest.skip("readiness artifact not generated in this checkout")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    episode_splits = {}
    for row in manifest["episodes"]:
        episode_splits.setdefault(row["episode_id"], set()).add(row["split"])
        assert len(row["candidate_ids"]) == 6
        assert row["candidate_ids"][-1].endswith(f":{HOLD_ACTION_ID}")
        if row["split"] == "test":
            assert row["harm_v2_labels"] == "SEALED_NOT_MATERIALIZED"
            assert row["benefit_labels"] == "SEALED_NOT_MATERIALIZED"
            assert row["gt_costs"] == "SEALED_NOT_MATERIALIZED"
    assert all(len(splits) == 1 for splits in episode_splits.values())


def test_v3_readiness_marks_old_hold_scores_ood_not_evidence() -> None:
    path = PROJECT_ROOT / "results_dev/phase5b17fd_hold_candidate_extension/summary.json"
    if not path.exists():
        pytest.skip("readiness artifact not generated in this checkout")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert "OUT-OF-DISTRIBUTION" in summary["old_v2_model_HOLD_status"]
    assert "NOT SCORED" in summary["old_v2_model_HOLD_status"]
    assert summary["test_reads"] == 0

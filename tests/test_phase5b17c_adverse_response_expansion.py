import inspect
import json

import numpy as np

from scripts import run_phase5b17c_adverse_response_expansion as run
from src.data import adverse_response_dataset as dataset
from src.data import adverse_response_protocol as protocol
from src.data.synthetic_interaction import (
    PROFILE_BY_ID, generate_interaction_split, sample_adverse_response_risk_factors,
    simulate_risk_conditioned_interaction_future,
)


def _one(seed=17):
    split = generate_interaction_split(1, seed, "test_fixture", profile_ids=(0,), noise_std=0, occlusion_rate=0)
    risk = sample_adverse_response_risk_factors(np.random.default_rng(seed + 1))
    simulation = simulate_risk_conditioned_interaction_future(
        split.human_history[0], split.natural_future[0], split.robot_history[0], 4,
        PROFILE_BY_ID[0], risk,
    )
    return split, risk, simulation


def test_adverse_event_is_derived_from_trajectory_not_label_flag():
    split, risk, simulation = _one()
    event = protocol.derive_adverse_response_events(
        split.human_history[0], split.natural_future[0], simulation.future_global
    )
    assert isinstance(event.adverse_human_kinematic_response, bool)
    source = inspect.getsource(protocol.derive_adverse_response_events)
    assert "candidate_future" in source and "natural_future" in source
    assert "risk_factors" not in source and "profile_id" not in source
    assert not hasattr(risk, "harm") and not hasattr(risk, "adverse_event")


def test_risk_factors_change_actual_trajectory():
    split, risk, first = _one()
    second_risk = dataset.AdverseResponseRiskFactors(.15, .10, .10, .35, .40, 4.5)
    second = simulate_risk_conditioned_interaction_future(
        split.human_history[0], split.natural_future[0], split.robot_history[0], 4,
        PROFILE_BY_ID[0], second_risk,
    )
    assert np.max(np.abs(first.future_global - second.future_global)) > 1e-5


def test_harm_v2_is_not_derived_from_benefit_total_cost_or_best_action():
    definition = run.harm_definition()
    assert not definition["depends_on_benefit"]
    assert not definition["depends_on_total_cost_comparison"]
    assert not definition["depends_on_best_action"]
    source = inspect.getsource(protocol.derive_adverse_response_events)
    assert all(word not in source for word in ("benefit", "total_cost", "best_action"))


def test_harm_v2_covers_every_gt_unsafe():
    episodes = dataset.build_development_split("train", 8, 91, 92)
    assert all(not candidate.gt_unsafe or candidate.harm_v2 for episode in episodes for candidate in episode.candidates)


def test_episode_split_and_held_out_profile_discipline():
    train = dataset.build_development_split("train", 7, 101, 102)
    validation = dataset.build_development_split("validation", 7, 103, 104)
    assert {episode.episode_id for episode in train}.isdisjoint({episode.episode_id for episode in validation})
    assert not {episode.profile_id for episode in train}.intersection(dataset.HELD_OUT_PROFILE_IDS)
    assert all(len(episode.candidates) == 5 for episode in train + validation)


def test_sealed_test_manifest_never_contains_labels():
    rows = dataset.sealed_test_manifest_rows(5, 111, 112)
    assert all(row["harm_v2_labels"] == "SEALED_NOT_MATERIALIZED" for row in rows)
    assert all(row["benefit_labels"] == "SEALED_NOT_MATERIALIZED" for row in rows)
    source = inspect.getsource(run.main)
    assert '"test_reads":0' in source and '"test_harm_labels_read": 0' in inspect.getsource(run.leakage_audit)


def test_c7_and_tradeoff_readiness_gates_use_episode_counts():
    source = inspect.getsource(run.readiness)
    assert 'c[("train", "C7")]["harm_positive_episodes"] >= 5' in source
    assert 'c[("validation", "C7")]["harm_positive_episodes"] >= 5' in source
    assert 'o["train"]["beneficial_harm_episodes"] >= 5' in source
    assert 'o["validation"]["beneficial_harm_episodes"] >= 3' in source


def test_safe_beneficial_definition_remains_conjunction():
    source = inspect.getsource(run.summary_row)
    assert 'row["benefit"] > BENEFIT_EPSILON and not row["harm_v2"] and row["feasible"]' in source


def test_action_and_profile_shortcut_audit_use_conditional_rates():
    rows = [
        {"split": "train", "dimension": "action", "group": "KEEP", "candidate_count": 10, "harm_positive_rate": .1},
        {"split": "train", "dimension": "action", "group": "SPEED_UP", "candidate_count": 10, "harm_positive_rate": .9},
    ]
    audit = run.shortcut_audit(rows, "action")
    assert audit["passed"] and audit["warning"]
    rows[1]["harm_positive_rate"] = .96
    assert not run.shortcut_audit(rows, "action")["passed"]


def test_manifest_v2_canonical_checksum_is_reproducible():
    value = {"label": protocol.LABEL, "episodes": [{"episode_id": "train:e0", "candidate_ids": ["train:e0:0"]}]}
    assert run.canonical_sha(value) == run.canonical_sha(json.loads(json.dumps(value)))
    assert len(run.canonical_sha(value)) == 64


def test_protocol_thresholds_are_predeclared_not_model_selected():
    value = protocol.protocol_definition()
    assert not value["threshold_selection_used_model_performance"]
    assert "synthetic motion support" in value["threshold_source"]
    assert not value["event_families"]["INTERACTION_DISRUPTION_EVENT"]["included"]


def test_no_model_training_optimizer_backward_qwen_or_lora():
    source = inspect.getsource(run)
    assert "torch.optim" not in source and ".backward(" not in source
    assert "Qwen" not in source and "LoRA" not in source
    assert '"optimizer_steps":0' in source and '"backward_calls":0' in source


def test_v1_path_is_read_only_and_v2_has_distinct_name():
    source = inspect.getsource(run.main)
    assert 'args.output_dir/"phase5b_manifest_v2.json"' in source
    assert "write_json(args.manifest_v1" not in source
    assert 'v1_before=frozen_before["manifest_v1"]' in source
    assert 'v1_after=frozen_after["manifest_v1"]' in source


def test_v1_v2_comparison_is_development_only_and_has_action_distribution():
    source = inspect.getsource(run.v1_distributions)
    assert 'row["split"] in ("train", "validation")' in source
    assert "action_candidates" in source and "beneficial_rate_development" in source
    source_v2 = inspect.getsource(run.v2_distributions)
    assert "action_candidates_development" in source_v2

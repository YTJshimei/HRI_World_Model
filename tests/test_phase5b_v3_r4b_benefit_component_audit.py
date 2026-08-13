"""Contracts for the Phase 5B-v3-R4B GT cost-component audit."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.decision.decision_cost import DecisionCostWeights
from src.evaluation.benefit_component_audit import COMPONENTS, episode_cost_components


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json": "ac3a620b1fb254f1a89114f3b0daa1e3ed7dc6a570cfcf083c23355a70ee542a",
    "results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv": "ae8dc040459d9dcfaabdd480309ea10ea535ed44643484e12626ad93c24939f1",
    "results_dev/phase5b_v3_r1a_runtime_anchor_realign/runtime_anchor_map.csv": "88bf047e0a26a9e79ca946dbe9ec277b24ce1e84fbaf956c4378826c29d3eb0f",
    "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt": "dbeacc97cba71515591586ce8343a5747dbb8c7034d4dbd250c3b4a58318a0ff",
    "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt": "2dea4137e30324daaa0344f5c5770d4758bcb563f3016ccb4d61bfa38c355b6d",
}


def test_r4b_frozen_contract_checksums():
    for relative, expected in EXPECTED.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_r4b_component_names_and_weights_are_exactly_formal_cost_contract():
    assert COMPONENTS == ("task", "safety", "human_response", "disturbance", "uncertainty")
    weights = DecisionCostWeights()
    assert tuple(weights.__dataclass_fields__) == COMPONENTS
    assert tuple(getattr(weights, name) for name in COMPONENTS) == (1.0, 3.0, 1.4, 0.55, 0.85)


def test_expanded_subcomponents_are_only_literal_formal_formula_terms():
    from src.evaluation import benefit_component_audit as audit

    source = inspect.getsource(audit.formal_subcomponents)
    expected = {
        "task.final_error", "task.mean_error", "task.progress_failure", "task.visibility_proxy",
        "safety.violation_proxy", "safety.unsafe_duration", "safety.close_gap", "safety.infeasible_penalty",
        "human_response.effect_magnitude", "human_response.speed_effect", "human_response.lateral_effect", "human_response.heading_effect",
        "disturbance.robot_speed_action", "disturbance.target_distance_action", "disturbance.lateral_action", "disturbance.human_effect_magnitude",
        "uncertainty.coordinate_uncertainty",
    }
    for name in expected:
        assert f'"{name}"' in source
    assert source.count('":') == len(expected)


def test_one_validation_episode_exactly_reconstructs_target_v2():
    from scripts import run_phase5b_v3_r1b_gara_fair_test as r1b
    from src.data.adverse_response_dataset import GENERATOR_SEED, POPULATION_PROFILE, RISK_SEED, build_development_split

    args = SimpleNamespace(
        manifest_v3=ROOT / "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json",
        target_v2=ROOT / "results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv",
        anchor_map=ROOT / "results_dev/phase5b_v3_r1a_runtime_anchor_realign/runtime_anchor_map.csv",
        r1_checkpoint=ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt",
        harm_checkpoint=ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt",
    )
    _, labels, anchors = r1b.load_contract(args)
    # The generator's final seeded permutation depends on the frozen split size;
    # materialize all 240 validation episodes before selecting one.
    episode = build_development_split("validation", 240, GENERATOR_SEED + 1000, RISK_SEED + 1000)[0]
    result = episode_cost_components(episode, POPULATION_PROFILE)
    action_ids = result["action_ids"]
    anchor_action = int(anchors[episode.episode_id]["runtime_anchor_action_id"])
    anchor_index = int(np.flatnonzero(action_ids == anchor_action)[0])
    weights = DecisionCostWeights()
    for index, action_id in enumerate(action_ids):
        reconstructed = sum(
            getattr(weights, name) * (result["components"][name][anchor_index] - result["components"][name][index])
            for name in COMPONENTS if name != "uncertainty"
        )
        target = float(labels[f"{episode.episode_id}:{int(action_id)}"]["benefit_v2_runtime_anchor"])
        assert abs(reconstructed - target) <= 1e-6


def test_target_v2_formal_generation_disables_uncertainty_only():
    from src.data import adverse_response_dataset as dataset

    source = inspect.getsource(dataset.build_development_split)
    assert source.count("include_uncertainty=False") == 2
    assert "include_human_response=False" not in source
    assert "include_disturbance=False" not in source


def test_benefit_safety_component_is_not_harm_v3():
    from scripts import run_phase5b_v3_r4b_benefit_component_audit as stage

    source = inspect.getsource(stage.main)
    assert '"not_equivalent_to_HARM_v3":True' in source
    assert '"negative_Benefit_is_Harm":False' in source
    assert 'ranking=json.loads' in source and 'harm=json.loads' in source


def test_r4b_has_zero_training_test_backward_or_decision_chain():
    from scripts import run_phase5b_v3_r4b_benefit_component_audit as stage

    source = inspect.getsource(stage)
    main = inspect.getsource(stage.main)
    assert stage.TEST_READS == stage.OPTIMIZER_STEPS == stage.BACKWARD_CALLS == 0
    assert 'build_development_split("test"' not in main
    for forbidden in ("torch.optim", ".backward(", "optimizer.step(", "decision_evaluation", "arbitrate("):
        assert forbidden not in source
    assert '"no_decision_chain_run":True' in source


def test_identifiability_thresholds_and_stop_rule_are_frozen():
    from scripts import run_phase5b_v3_r4b_benefit_component_audit as stage

    source = inspect.getsource(stage.dominant_label)
    assert ">=.40" in source and ">=.70" in source
    stop_source = inspect.getsource(stage.stable_stop_explanation)
    assert '"required_count":6' in stop_source and "count>=6" in stop_source


def test_exact_reconstruction_tolerance_comes_only_from_float32_serialization():
    from scripts import run_phase5b_v3_r4b_benefit_component_audit as stage

    assert stage.half_float32_ulp(32.0) == 0.5 * abs(float(np.spacing(np.float32(32.0))))
    main = inspect.getsource(stage.main)
    assert "max_normalized_error>1.0" in main
    assert "source float32 serialization" in main
    assert "TOLERANCE=2" not in inspect.getsource(stage)

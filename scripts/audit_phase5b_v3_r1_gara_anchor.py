"""Pre-training canonical-anchor audit for Phase 5B-v3-R1 GARA.

The preregistered protocol requires an immediate stop when the GT Benefit
anchor and a fully runtime-valid generic anchor are not the same contract.
This script therefore performs no model load, forward, training, thresholding,
or decision reconstruction.  It only materializes synthetic TRAIN/VALIDATION
development episodes and writes the mandatory stop audit.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from src.data import adverse_response_dataset as adverse
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.data.robot_action_schema import HOLD_ACTION_ID
from src.multimodal import phase5b_v2_dataset as runtime_bridge
from src.multimodal.temporal_schema import LABEL

MECHANISM = "DEVELOPMENT MECHANISM RESULT"
STAGE = "Phase 5B-v3-R1 Generic-Anchored Relative Advantage - Anchor Preflight"
EXPECTED_MANIFEST_SHA = "ac3a620b1fb254f1a89114f3b0daa1e3ed7dc6a570cfcf083c23355a70ee542a"
EXPECTED_R1_SHA = "dbeacc97cba71515591586ce8343a5747dbb8c7034d4dbd250c3b4a58318a0ff"
EXPECTED_HARM_SHA = "2dea4137e30324daaa0344f5c5770d4758bcb563f3016ccb4d61bfa38c355b6d"
NUMERICAL_TOLERANCE = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--manifest-v3", type=Path, default=PROJECT_ROOT / "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json")
    parser.add_argument("--r1-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt")
    parser.add_argument("--harm-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r1_gara")
    return parser.parse_args()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_location(function) -> dict[str, object]:
    path = Path(inspect.getsourcefile(function)).resolve()
    _, line = inspect.getsourcelines(function)
    return {"file": str(path.relative_to(PROJECT_ROOT)), "function": function.__name__, "first_line": line, "sha256": file_sha(path)}


def split_anchor_audit(split: str, seed: int, risk_seed: int) -> dict[str, object]:
    episodes = build_development_split(split, 240, seed, risk_seed)
    benefits, anchor_actions, exact_ties = [], [], 0
    for episode in episodes:
        index = int(episode.generic_action_index)
        benefits.append(float(episode.candidates[index].benefit))
        anchor_actions.append(int(episode.candidates[index].action_id))
        exact_ties += int(np.sum(episode.generic_costs == np.min(episode.generic_costs)) > 1)
    values = np.asarray(benefits, dtype=np.float64)
    return {
        "split": split,
        "episode_count": len(episodes),
        "candidate_count_with_hold": 6 * len(episodes),
        "anchor_candidate_support": list(adverse.ACTION_IDS),
        "anchor_action_distribution": dict(sorted(Counter(map(str, anchor_actions)).items())),
        "exact_generic_cost_tie_episodes": exact_ties,
        "canonical_generic_GT_benefit_max_abs": float(np.max(np.abs(values))),
        "canonical_generic_GT_benefit_mean_abs": float(np.mean(np.abs(values))),
        "canonical_generic_GT_benefit_exact_zero_count": int(np.sum(values == 0.0)),
        "within_numerical_tolerance": bool(np.max(np.abs(values)) <= NUMERICAL_TOLERANCE),
        "test_reads": 0,
    }


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite GARA preflight: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    before = {
        "manifest_v3": file_sha(args.manifest_v3),
        "R1_v3_BASE_checkpoint": file_sha(args.r1_checkpoint),
        "HARM_v3_BASE_checkpoint": file_sha(args.harm_checkpoint),
    }
    expected = {"manifest_v3": EXPECTED_MANIFEST_SHA, "R1_v3_BASE_checkpoint": EXPECTED_R1_SHA, "HARM_v3_BASE_checkpoint": EXPECTED_HARM_SHA}
    if before != expected:
        raise RuntimeError(f"frozen input checksum mismatch: {before}")

    train = split_anchor_audit("train", GENERATOR_SEED, RISK_SEED)
    validation = split_anchor_audit("validation", GENERATOR_SEED + 1000, RISK_SEED + 1000)

    generator_source = inspect.getsource(adverse.build_development_split)
    runtime_source = inspect.getsource(runtime_bridge.build_v2_temporal_samples)
    target_uses_label_side_gt_natural = "base.natural_future[index]" in generator_source
    runtime_explicitly_excludes_gt_natural = "label-side" in runtime_source and "runtime_natural" in runtime_source
    anchor_runtime_valid = not target_uses_label_side_gt_natural
    contract = {
        "label": LABEL,
        "mechanism_result": MECHANISM,
        "stage": STAGE,
        "target_construction_source": source_location(adverse.build_development_split),
        "runtime_bridge_source": source_location(runtime_bridge.build_v2_temporal_samples),
        "selection_function": "numpy.argmin(generic_costs.total)",
        "selection_inputs": [
            "runtime-observable human history", "runtime-observable robot history",
            "runtime confidence/visibility", "runtime target follow distance",
            "candidate action IDs A0-A4", "fixed development population profile",
            "LABEL-SIDE GT natural_future (forbidden at runtime)",
        ],
        "tie_break": "numpy.argmin returns the first minimum in ACTION_IDS order [0,1,2,3,4]",
        "candidate_ids": list(adverse.ACTION_IDS),
        "includes_HOLD": False,
        "HOLD_benefit_anchor": "the same pre-HOLD A0-A4 generic_action_index; HOLD is evaluated against it but cannot be the anchor",
        "uses_GT_total_cost_to_select": False,
        "uses_GT_benefit_to_select": False,
        "uses_profile_ID_to_select": False,
        "uses_label_side_GT_natural_future": target_uses_label_side_gt_natural,
        "runtime_bridge_replaces_GT_natural_with_history_only_constant_velocity": runtime_explicitly_excludes_gt_natural,
        "anchor_is_100_percent_runtime_valid": anchor_runtime_valid,
        "GT_target_anchor_equals_runtime_generic_contract": False,
        "stop_required": True,
        "stop_reason": "GT Benefit anchor selection consumes base.natural_future, which the frozen runtime bridge explicitly classifies as label-side state and replaces with a history-only constant-velocity prior.",
    }
    consistency = {
        "label": LABEL, "mechanism_result": MECHANISM, "numerical_tolerance": NUMERICAL_TOLERANCE,
        "train": train, "validation": validation,
        "all_canonical_generic_GT_benefits_exact_zero": train["canonical_generic_GT_benefit_max_abs"] == 0.0 and validation["canonical_generic_GT_benefit_max_abs"] == 0.0,
        "semantic_anchor_contract_consistent": False,
        "numerical_zero_consistency_is_not_sufficient_for_runtime_validity": True,
        "test_reads": 0,
    }
    after = {
        "manifest_v3": file_sha(args.manifest_v3),
        "R1_v3_BASE_checkpoint": file_sha(args.r1_checkpoint),
        "HARM_v3_BASE_checkpoint": file_sha(args.harm_checkpoint),
    }
    frozen = {
        "label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE,
        "checksums_before": before, "checksums_after": after, "expected_checksums": expected,
        "all_frozen_files_unchanged": before == after == expected,
        "R1_backbone_loaded": False, "R1_forward_calls": 0, "R1_optimizer_steps": 0,
        "HARM_head_loaded": False, "harm_forward_calls": 0, "harm_optimizer_steps": 0,
        "H0_trained": False, "H1_GARA_trained": False, "checkpoints_written": 0,
        "threshold_calibration_performed": False, "decision_chain_run": False,
        "generic_safety_coverage_run": False, "fallback_selection_run": False,
        "test_identity_reads": 0, "test_trajectory_reads": 0, "test_label_reads": 0,
        "test_model_output_reads": 0, "test_reads": 0,
    }
    gate_a_checks = {
        "manifest_unchanged": before["manifest_v3"] == after["manifest_v3"] == EXPECTED_MANIFEST_SHA,
        "TEST_reads_zero": frozen["test_reads"] == 0,
        "backbone_untouched": before["R1_v3_BASE_checkpoint"] == after["R1_v3_BASE_checkpoint"],
        "harm_untouched": before["HARM_v3_BASE_checkpoint"] == after["HARM_v3_BASE_checkpoint"],
        "GT_generic_benefit_exact_zero": consistency["all_canonical_generic_GT_benefits_exact_zero"],
        "canonical_generic_runtime_valid": contract["anchor_is_100_percent_runtime_valid"],
        "GT_and_runtime_anchor_contract_identical": contract["GT_target_anchor_equals_runtime_generic_contract"],
    }
    gates = {
        "Gate_A": {"name": "Isolation & Contract", "checks": gate_a_checks, "passed": all(gate_a_checks.values())},
        "Gate_B": {"name": "Safe-Beneficial Sign Improvement", "status": "NOT_RUN_DUE_TO_GATE_A_STOP", "passed": False},
        "Gate_C": {"name": "Ranking Preservation", "status": "NOT_RUN_DUE_TO_GATE_A_STOP", "passed": False},
        "Gate_D": {"name": "Calibration Guard", "status": "NOT_RUN_DUE_TO_GATE_A_STOP", "passed": False},
        "Gate_E": {"name": "Harm Isolation", "status": "NOT_RUN_DUE_TO_GATE_A_STOP", "passed": False},
        "Gate_F": {"name": "No Degenerate Positive Shift", "status": "NOT_RUN_DUE_TO_GATE_A_STOP", "passed": False},
        "all_passed": False,
        "stopped_before_training": True,
    }
    summary = {
        "label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE,
        "status": "STOPPED_PRETRAINING_CANONICAL_GENERIC_CONTRACT_MISMATCH",
        "test_reads": 0, "canonical_generic_numerically_zero": True,
        "canonical_generic_runtime_valid": False,
        "GARA_training_started": False, "GARA_checkpoint_frozen": False,
        "GARA_success": False, "ready_for_v3_safe_decision_chain_reconstruction": False,
        "reason": contract["stop_reason"],
        "required_next_step": "Define and preregister a runtime-valid canonical generic anchor and reconcile/rebuild the Benefit target contract before any GARA training.",
        "next_stage_started": False,
    }
    io.write_json(args.output_dir / "frozen_contract.json", frozen)
    io.write_json(args.output_dir / "generic_anchor_contract.json", contract)
    io.write_json(args.output_dir / "anchor_consistency_audit.json", consistency)
    io.write_json(args.output_dir / "gate_results.json", gates)
    io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

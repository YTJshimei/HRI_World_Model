"""Phase 5B-v3-R2 decoupled pair-conditioned Benefit readout (DPCBR).

This development-only synthetic experiment trains two minimal absolute Benefit
readouts over frozen R1-v3 representations.  Frozen B0 remains the sole ranking
source and HARM-v3 remains the sole risk source.  TEST is never materialized.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b15_decision_bottleneck as b15
from scripts import run_phase5b16_candidate_ranking as b16
from scripts import run_phase5b_v3_r1b_gara_fair_test as r1b
from scripts import run_phase5b_v3_r1c_frozen_runtime_generic_reanchor as r1c
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.data.robot_action_schema import HOLD_ACTION_ID
from src.evaluation.context_value_metrics import pearson, spearman
from src.models.independent_harm_head import RiskPreservingBypassHead
from src.models.pair_conditioned_benefit import AbsoluteCandidateBenefitReadout, PairConditionedBenefitReadout
from src.multimodal.phase5b_v3_dataset import build_v3_temporal_samples
from src.multimodal.temporal_schema import LABEL

MECHANISM = "DEVELOPMENT MECHANISM RESULT"
STAGE = "Phase 5B-v3-R2 Decoupled Pair-Conditioned Benefit Readout"
TEST_READS = 0
LAMBDA_RANK = 0.0
TOLERANCE = 1e-10
EXPECTED_B0_MAE = 1.9629752593426275
EXPECTED_RANKING = {
    "mean_feasible_pairwise_accuracy": 0.8130555555555555,
    "gt_best_top1_accuracy": 0.8416666666666667,
    "gt_best_top2_recall": 0.9541666666666667,
    "mean_gt_best_rank": 1.2291666666666667,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--epochs", type=int, choices=(30,), default=30)
    parser.add_argument("--patience", type=int, choices=(5,), default=5)
    parser.add_argument("--batch-size", type=int, choices=(64,), default=64)
    parser.add_argument("--learning-rate", type=float, choices=(3e-4,), default=3e-4)
    parser.add_argument("--manifest-v3", type=Path, default=PROJECT_ROOT / "results_dev/phase5b17fd_hold_candidate_extension/phase5b_manifest_v3.json")
    parser.add_argument("--target-v2", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r1a_runtime_anchor_realign/benefit_target_v2_labels.csv")
    parser.add_argument("--anchor-map", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r1a_runtime_anchor_realign/runtime_anchor_map.csv")
    parser.add_argument("--r1-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/r1_v3_base_cracs.pt")
    parser.add_argument("--harm-checkpoint", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline/checkpoints/harm_v3_base_phs.pt")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r2_pair_conditioned_benefit")
    return parser.parse_args()


def state_sha(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def array_sha(value) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def select_epoch(rows):
    """Pre-registered selector: minimum MAE, maximum safe sign, earlier epoch."""
    if not rows:
        raise ValueError("selector requires validation rows")
    return min(rows, key=lambda row: (float(row["Benefit_MAE"]), -float(row["safe_beneficial_sign_accuracy"]), int(row["epoch"])))


def _forward(head, candidate, generic):
    if isinstance(head, PairConditionedBenefitReadout):
        return head(candidate, generic)
    if isinstance(head, AbsoluteCandidateBenefitReadout):
        return head(candidate)
    raise TypeError(f"unsupported Benefit head: {type(head).__name__}")


def regression_loss(head, data, indices, torch, device):
    local = torch.as_tensor(indices, dtype=torch.long)
    candidate = data["context"][local].to(device)
    generic = data["generic"][local].to(device)
    target = torch.as_tensor(data["target"][indices] / data["scale"], dtype=torch.float32, device=device)
    log_variance = data["log_variance"][local].to(device)
    feasible = torch.tensor([data["samples"][index].targets.feasible for index in indices], dtype=torch.bool, device=device)
    prediction = _forward(head, candidate, generic)
    error = prediction[feasible] - target[feasible]
    nll = .5 * (error.square() * torch.exp(-log_variance[feasible]) + log_variance[feasible]).mean()
    return nll


def predict(head, data, batch_size, torch, device):
    values = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(data["context"]), batch_size):
            candidate = data["context"][start:start + batch_size].to(device)
            generic = data["generic"][start:start + batch_size].to(device)
            values.append(_forward(head, candidate, generic).cpu())
    return torch.cat(values).numpy().astype(np.float64) * data["scale"]


def validation_row(name, head, data, epoch, train_nll, batch_size, torch, device):
    prediction = predict(head, data, batch_size, torch, device)
    calibration = r1c.calibration_row(data["samples"], prediction, data["target"], name)
    sign = r1b.sign_summary(data["samples"], prediction, data["target"], name)
    return {
        "synthetic_interaction": LABEL,
        "mechanism_result": MECHANISM,
        "model": name,
        "epoch": epoch,
        "train_heteroscedastic_NLL": train_nll,
        "Benefit_MAE": calibration["Benefit_MAE"],
        "safe_beneficial_sign_accuracy": sign["safe_beneficial_sign_accuracy"],
        "safe_beneficial_positive_count": sign["predicted_positive_count"],
        "lambda_rank": LAMBDA_RANK,
        "ranking_loss": 0.0,
        "parameter_checksum": state_sha(head.state_dict()),
    }


def train_head(name, head, train_data, validation_data, batches, args, torch, device):
    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    rows, states, best_epoch, stale = [], {}, None, 0
    started = time.perf_counter()
    for epoch, epoch_batches in enumerate(batches, 1):
        head.train(); losses = []
        for indices in epoch_batches:
            nll = regression_loss(head, train_data, indices, torch, device)
            optimizer.zero_grad(set_to_none=True); nll.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0, error_if_nonfinite=True))
            optimizer.step(); losses.append((float(nll.detach()), gradient))
        row = validation_row(name, head, validation_data, epoch, float(np.mean([value[0] for value in losses])), args.batch_size, torch, device)
        row["mean_gradient_norm"] = float(np.mean([value[1] for value in losses]))
        if not all(math.isfinite(float(row[key])) for key in ("train_heteroscedastic_NLL", "Benefit_MAE", "safe_beneficial_sign_accuracy", "mean_gradient_norm")):
            raise FloatingPointError(f"non-finite {name} training state")
        rows.append(row); states[epoch] = copy.deepcopy(head.state_dict())
        selected = select_epoch(rows)
        if selected["epoch"] != best_epoch:
            best_epoch, stale = selected["epoch"], 0
        else:
            stale += 1
        row["selector_current_best_epoch"] = best_epoch; row["selector_stale_epochs"] = stale
        print(f"{name} epoch={epoch:02d} NLL={row['train_heteroscedastic_NLL']:.5f} MAE={row['Benefit_MAE']:.5f} sign={row['safe_beneficial_sign_accuracy']:.4f} best={best_epoch} stale={stale}", flush=True)
        if stale >= args.patience:
            break
    selected = select_epoch(rows); head.load_state_dict(states[selected["epoch"]]); head.eval()
    for row in rows:
        row["selector_final_selected"] = row["epoch"] == selected["epoch"]
    return {
        "head": head,
        "rows": rows,
        "selected": selected,
        "epochs_completed": len(rows),
        "training_time_s": time.perf_counter() - started,
        "optimizer_parameter_count": sum(parameter.numel() for group in optimizer.param_groups for parameter in group["params"]),
    }


def absolute_metrics(samples, prediction, sigma, target, name, formal_ranking):
    prediction = np.asarray(prediction, np.float64); target = np.asarray(target, np.float64)
    feasible = np.asarray([sample.targets.feasible for sample in samples], bool)
    error = prediction[feasible] - target[feasible]
    finite_sigma = np.maximum(np.asarray(sigma, np.float64)[feasible], 1e-6)
    row = {
        "synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "model": name,
        "Benefit_MAE": float(np.mean(np.abs(error))),
        "Benefit_Pearson": pearson(prediction[feasible], target[feasible]),
        "Benefit_Spearman": spearman(prediction[feasible], target[feasible]),
        "Benefit_Uncertainty_NLL": float(np.mean(.5 * (error / finite_sigma) ** 2 + np.log(finite_sigma) + .5*np.log(2*np.pi))),
        "global_bias": float(np.mean(error)), "median_error": float(np.median(error)),
        "formal_ranking_source": "Frozen B0 only; this readout is not used to sort candidates",
    }
    row.update({f"formal_{key}": formal_ranking[key] for key in EXPECTED_RANKING})
    return row


def subgroup_rows(samples, predictions, target, group, predicate):
    return [{"group": group, **r1b.sign_summary(samples, value, target, model, predicate)} for model, value in predictions.items()]


def hold_rows(samples, predictions, target):
    hold = np.asarray([sample.split_metadata["candidate_action_id_audit"] == HOLD_ACTION_ID for sample in samples], bool)
    feasible = np.asarray([sample.targets.feasible for sample in samples], bool)
    harm = np.asarray([sample.split_metadata["harm_v2_evaluation_only"] for sample in samples], bool)
    beneficial = hold & (target > TOLERANCE); safe = beneficial & feasible & ~harm; nonbeneficial = hold & (target <= TOLERANCE)
    rows = []
    for name, value in predictions.items():
        rows.append({
            "synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "model": name,
            "beneficial_HOLD_count": int(beneficial.sum()), "beneficial_HOLD_predicted_positive": int(np.sum(value[beneficial] > 0)),
            "beneficial_HOLD_sign_accuracy": float(np.mean(value[beneficial] > 0)),
            "safe_beneficial_HOLD_count": int(safe.sum()), "safe_beneficial_HOLD_predicted_positive": int(np.sum(value[safe] > 0)),
            "safe_beneficial_HOLD_sign_accuracy": float(np.mean(value[safe] > 0)),
            "nonbeneficial_HOLD_count": int(nonbeneficial.sum()), "nonbeneficial_HOLD_predicted_positive": int(np.sum(value[nonbeneficial] > 0)),
            "nonbeneficial_HOLD_FPR": float(np.mean(value[nonbeneficial] > 0)),
            "formal_HOLD_ranking_source": "Frozen B0 only",
        })
    return rows


def ranking_invariance(samples, target, sigma, before, after):
    before_metrics, _ = r1b.metrics(samples, before, sigma, target, "FROZEN_B0_BEFORE")
    after_metrics, _ = r1b.metrics(samples, after, sigma, target, "FROZEN_B0_AFTER")
    changed = []
    for episode_id, indices in b15.group_episode(samples).items():
        if r1c.rank_signature(samples, indices, before) != r1c.rank_signature(samples, indices, after):
            changed.append(episode_id)
    keys = tuple(EXPECTED_RANKING)
    return {
        "label": LABEL, "mechanism_result": MECHANISM,
        "formal_ranking_source": "Frozen B0 prediction only",
        "A0_used_for_ranking": False, "A1_used_for_ranking": False,
        "B0_prediction_max_abs_diff": float(np.max(np.abs(after-before))),
        "B0_prediction_exact": bool(np.array_equal(before, after)),
        "rank_signature_changes": len(changed), "changed_episode_ids": changed,
        "metrics_before": {key: before_metrics[key] for key in keys},
        "metrics_after": {key: after_metrics[key] for key in keys},
        "historical_expected": EXPECTED_RANKING,
        "metrics_exact": all(before_metrics[key] == after_metrics[key] for key in keys),
        "historical_metrics_within_tolerance": all(abs(before_metrics[key]-EXPECTED_RANKING[key]) <= 1e-12 for key in keys),
    }


def shortcut_rows(samples, prediction, anchors):
    getters = {
        "candidate_action": lambda sample: str(sample.split_metadata["candidate_action_id_audit"]),
        "runtime_generic_action": lambda sample: str(anchors[sample.episode_id]["runtime_anchor_action_id"]),
        "motion": lambda sample: str(sample.split_metadata["motion_type_evaluation_only"]),
        "context": lambda sample: "|".join(map(str, sample.split_metadata["contexts_evaluation_only"])) or "NONE",
        "HOLD": lambda sample: str(sample.split_metadata["candidate_action_id_audit"] == HOLD_ACTION_ID),
        "profile_audit": lambda sample: str(sample.split_metadata["person_profile_id"]),
    }
    rows = []; overall = float(np.mean(prediction)); total_variance = float(np.var(prediction))
    for dimension, getter in getters.items():
        groups = defaultdict(list)
        for sample, value in zip(samples, prediction):
            groups[getter(sample)].append(float(value))
        between = float(sum(len(values)*(np.mean(values)-overall)**2 for values in groups.values()) / len(prediction))
        ratio = between / max(total_variance, 1e-12)
        rows.append({
            "synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "dimension": dimension,
            "group_count": len(groups), "between_variance_ratio": ratio,
            "group_mean_min": float(min(np.mean(values) for values in groups.values())),
            "group_mean_max": float(max(np.mean(values) for values in groups.values())),
            "near_deterministic_shortcut": ratio >= .95,
            "profile_ID_runtime_input": False if dimension == "profile_audit" else "",
        })
    return rows


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite R2 result: {args.output_dir}")
    args.output_dir.mkdir(parents=True); (args.output_dir / "checkpoints").mkdir()
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    checksums_before, labels, anchors = r1b.load_contract(args)
    episodes = {
        "train": build_development_split("train", 240, GENERATOR_SEED, RISK_SEED),
        "validation": build_development_split("validation", 240, GENERATOR_SEED+1000, RISK_SEED+1000),
    }
    samples = {split: build_v3_temporal_samples(value) for split, value in episodes.items()}
    targets = {split: r1b.apply_target_v2(value, labels) for split, value in samples.items()}

    payload = torch.load(args.r1_checkpoint, map_location=device, weights_only=False)
    from src.models.rich_temporal_small_transformer_v3 import RichTemporalSmallTransformerV3
    backbone = RichTemporalSmallTransformerV3().to(device); backbone.load_state_dict(payload["model_state_dict"]); backbone.eval()
    for parameter in backbone.parameters(): parameter.requires_grad_(False)
    backbone_before = state_sha(backbone.state_dict())
    frozen = {split: r1b.extract_frozen(backbone, value, payload["normalizer"], args.batch_size, torch, device) for split, value in samples.items()}
    for split in samples:
        generic_index, identity = r1b.generic_indices(samples[split], anchors)
        frozen[split].update({
            "generic": frozen[split]["context"][generic_index], "generic_indices": generic_index,
            "target": targets[split], "samples": samples[split],
            "scale": float(payload["normalizer"]["benefit_scale"]), "generic_identity": identity,
        })
        if frozen[split]["context"].shape != frozen[split]["generic"].shape or frozen[split]["context"].shape[1] != 128:
            raise RuntimeError("z_i/z_g runtime representation contract failed")

    b0 = frozen["validation"]["old_benefit"]
    sigma = np.exp(.5*frozen["validation"]["log_variance"].numpy().astype(np.float64))*frozen["validation"]["scale"]
    b0_metric, _ = r1b.metrics(samples["validation"], b0, sigma, targets["validation"], "B0_FROZEN_RANKING")
    b0_sign = r1b.sign_summary(samples["validation"], b0, targets["validation"], "B0_FROZEN_RANKING")
    if b0_sign["safe_beneficial_count"] != 115 or b0_sign["predicted_positive_count"] != 42 or abs(b0_metric["Benefit_MAE"]-EXPECTED_B0_MAE) > 1e-12:
        raise RuntimeError("strict frozen B0 Target-v2 reproduction failed")

    harm_payload = torch.load(args.harm_checkpoint, map_location=device, weights_only=False)
    harm_head = RiskPreservingBypassHead().to(device); harm_head.load_state_dict(harm_payload["model_state_dict"]); harm_head.eval()
    for parameter in harm_head.parameters(): parameter.requires_grad_(False)
    with torch.inference_mode(): harm_before = harm_head(frozen["validation"]["bypass"].to(device)).cpu().numpy()

    batches, batch_audit = b16.make_episode_batches(samples["train"], args.epochs, args.batch_size, args.seed)
    torch.manual_seed(args.seed); a0 = AbsoluteCandidateBenefitReadout(); a0_initial = state_sha(a0.state_dict())
    torch.manual_seed(args.seed); a1 = PairConditionedBenefitReadout(); a1_initial = state_sha(a1.state_dict())
    a0_result = train_head("A0_ABSOLUTE", a0, frozen["train"], frozen["validation"], batches, args, torch, device)
    a1_result = train_head("A1_PAIR_CONDITIONED", a1, frozen["train"], frozen["validation"], batches, args, torch, device)
    predictions = {
        "B0_FROZEN_RANKING": b0,
        "A0_ABSOLUTE": predict(a0_result["head"], frozen["validation"], args.batch_size, torch, device),
        "A1_PAIR_CONDITIONED": predict(a1_result["head"], frozen["validation"], args.batch_size, torch, device),
    }

    formal_ranking = {key: b0_metric[key] for key in EXPECTED_RANKING}
    comparison = [absolute_metrics(samples["validation"], value, sigma, targets["validation"], name, formal_ranking) for name, value in predictions.items()]
    calibration = [r1c.calibration_row(samples["validation"], value, targets["validation"], name) for name, value in predictions.items()]
    sign_rows = [r1b.sign_summary(samples["validation"], value, targets["validation"], name) for name, value in predictions.items()]
    signs = {row["model"]: row for row in sign_rows}; calibrations = {row["model"]: row for row in calibration}

    target = targets["validation"]
    safe = np.asarray([sample.targets.feasible and not sample.split_metadata["harm_v2_evaluation_only"] for sample in samples["validation"]], bool) & (target > TOLERANCE)
    a0_positive = predictions["A0_ABSOLUTE"] > 0; a1_positive = predictions["A1_PAIR_CONDITIONED"] > 0
    recovered = safe & ~a0_positive & a1_positive; regressed = safe & a0_positive & ~a1_positive
    recovery_rows = [{
        "synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "record_type": "SUMMARY",
        "safe_beneficial_count": int(safe.sum()), "gross_recovery": int(recovered.sum()),
        "regression": int(regressed.sum()), "net_recovery": int(recovered.sum()-regressed.sum()),
    }]
    for index in np.flatnonzero(recovered | regressed):
        sample = samples["validation"][index]
        recovery_rows.append({
            "synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
            "record_type": "RECOVERED" if recovered[index] else "REGRESSED",
            "candidate_id": sample.sample_id, "episode_id": sample.episode_id,
            "candidate_action": int(sample.split_metadata["candidate_action_id_audit"]),
            "GT_Benefit": float(target[index]), "A0_prediction": float(predictions["A0_ABSOLUTE"][index]),
            "A1_prediction": float(predictions["A1_PAIR_CONDITIONED"][index]),
        })

    predicates = {
        "C7": lambda sample: any(str(value).startswith("C7") for value in sample.split_metadata["contexts_evaluation_only"]),
        "STOP": lambda sample: sample.split_metadata["motion_type_evaluation_only"] == "stop",
    }
    c7_rows = subgroup_rows(samples["validation"], predictions, target, "C7", predicates["C7"])
    stop_rows = subgroup_rows(samples["validation"], predictions, target, "STOP", predicates["STOP"])
    holds = hold_rows(samples["validation"], predictions, target)

    changed_episodes = {episode for episode, row in anchors.items() if row["split"] == "validation" and row["anchor_agrees"] == "False"}
    anchor_rows = []
    for group, keep in (("ANCHOR_SAME", lambda episode: episode not in changed_episodes), ("ANCHOR_CHANGED", lambda episode: episode in changed_episodes)):
        mask = np.asarray([keep(sample.episode_id) for sample in samples["validation"]], bool)
        subset = [sample for sample, selected in zip(samples["validation"], mask) if selected]
        for name in ("A0_ABSOLUTE", "A1_PAIR_CONDITIONED"):
            sign = r1b.sign_summary(subset, predictions[name][mask], target[mask], name)
            cal = r1c.calibration_row(subset, predictions[name][mask], target[mask], name)
            anchor_rows.append({
                "synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "group": group, "model": name,
                "safe_beneficial_count": sign["safe_beneficial_count"], "predicted_positive_count": sign["predicted_positive_count"],
                "safe_beneficial_sign_accuracy": sign["safe_beneficial_sign_accuracy"], "Benefit_MAE": cal["Benefit_MAE"],
            })

    historical = [row for row in r1b.read_rows(PROJECT_ROOT / "results_dev/phase5b_v3_r1a_runtime_anchor_realign/historical_sign_failure_reclassification.csv") if row["category"] == "A_STILL_NEW_SAFE_BENEFICIAL_AND_PREDICTED_NONPOSITIVE"]
    by_id = {sample.sample_id: index for index, sample in enumerate(samples["validation"])}
    historical_indices = np.asarray([by_id[row["candidate_id"]] for row in historical], int)
    historical_rows = []
    for name, value in predictions.items():
        historical_rows.append({
            "synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "model": name,
            "historical_true_failure_count": len(historical_indices),
            "recovered_positive_count": int(np.sum(value[historical_indices] > 0)),
            "remaining_failure_count": int(np.sum(value[historical_indices] <= 0)),
            "GARA_reference": 14, "FRGR_reference": 28, "RCEOC_reference": 1,
        })

    a1_head = a1_result["head"]
    candidate_weight, generic_weight = [value.detach().cpu() for value in a1_head.weight_halves()]
    cosine = float(torch.nn.functional.cosine_similarity(candidate_weight, -generic_weight, dim=0))
    weight_geometry = {
        "label": LABEL, "mechanism_result": MECHANISM,
        "candidate_weight_L2": float(candidate_weight.norm()), "generic_weight_L2": float(generic_weight.norm()),
        "cosine_w_candidate_vs_negative_w_generic": cosine,
        "near_GARA_geometry": cosine >= .9,
        "interpretation": "Descriptive only: proximity to one suggests GARA-like antisymmetry; no constraint was imposed.",
    }
    shortcuts = shortcut_rows(samples["validation"], predictions["A1_PAIR_CONDITIONED"], anchors)

    b0_after = r1b.extract_frozen(backbone, samples["validation"], payload["normalizer"], args.batch_size, torch, device)["old_benefit"]
    ranking = ranking_invariance(samples["validation"], target, sigma, b0, b0_after)
    with torch.inference_mode(): harm_after = harm_head(frozen["validation"]["bypass"].to(device)).cpu().numpy()
    checksums_after = {
        "manifest_v3": r1b.file_sha(args.manifest_v3), "Benefit_Target_v2": r1b.file_sha(args.target_v2),
        "runtime_anchor_map": r1b.file_sha(args.anchor_map), "R1_v3_BASE": r1b.file_sha(args.r1_checkpoint),
        "HARM_v3_BASE": r1b.file_sha(args.harm_checkpoint),
    }
    backbone_after = state_sha(backbone.state_dict())
    harm_isolation = {
        "label": LABEL, "mechanism_result": MECHANISM,
        "HARM_checkpoint_SHA_before": checksums_before["HARM_v3_BASE"], "HARM_checkpoint_SHA_after": checksums_after["HARM_v3_BASE"],
        "HARM_checkpoint_unchanged": checksums_before["HARM_v3_BASE"] == checksums_after["HARM_v3_BASE"],
        "harm_optimizer_created": False, "harm_parameters_require_grad": any(parameter.requires_grad for parameter in harm_head.parameters()),
        "validation_harm_logits_before_SHA": array_sha(harm_before), "validation_harm_logits_after_SHA": array_sha(harm_after),
        "validation_harm_logits_max_abs_diff": float(np.max(np.abs(harm_after-harm_before))),
        "harm_outputs_exact": bool(np.array_equal(harm_before, harm_after)),
    }

    a0s, a1s = signs["A0_ABSOLUTE"], signs["A1_PAIR_CONDITIONED"]
    a0c, a1c = calibrations["A0_ABSOLUTE"], calibrations["A1_PAIR_CONDITIONED"]
    stop_by = {row["model"]: row for row in stop_rows}; hold_by = {row["model"]: row for row in holds}
    finite = all(np.isfinite(value).all() for value in predictions.values())
    gates = {
        "Gate_A": {"name": "Isolation", "checks": {
            "frozen_checksums_unchanged": checksums_before == checksums_after,
            "R1_backbone_state_unchanged": backbone_before == backbone_after,
            "B0_ranking_exact": ranking["metrics_exact"] and ranking["rank_signature_changes"] == 0,
            "Harm_exact": harm_isolation["harm_outputs_exact"], "TEST_reads_zero": TEST_READS == 0,
            "only_A0_A1_trainable": True,
        }},
        "Gate_B": {"name": "Safe-Beneficial Sign", "checks": {
            "A1_minus_A0_at_least_0_15": a1s["safe_beneficial_sign_accuracy"] >= a0s["safe_beneficial_sign_accuracy"] + .15,
            "A1_accuracy_at_least_0_55": a1s["safe_beneficial_sign_accuracy"] >= .55,
        }},
        "Gate_C": {"name": "Absolute Calibration", "checks": {
            "A1_MAE_not_above_A0": a1c["Benefit_MAE"] <= a0c["Benefit_MAE"],
            "A1_MAE_not_above_Frozen_B0": a1c["Benefit_MAE"] <= EXPECTED_B0_MAE,
        }},
        "Gate_D": {"name": "Ranking Preservation", "checks": {
            "Frozen_B0_ranking_strictly_preserved": ranking["B0_prediction_exact"] and ranking["metrics_exact"] and ranking["historical_metrics_within_tolerance"] and ranking["rank_signature_changes"] == 0,
        }},
        "Gate_E": {"name": "No Degenerate Positive Shift", "checks": {
            "GT_negative_FPR_increase_at_most_0_05": a1s["GT_negative_false_positive_rate"] <= a0s["GT_negative_false_positive_rate"] + .05,
            "safe_beneficial_precision_drop_at_most_0_10": a1s["safe_beneficial_precision"] >= a0s["safe_beneficial_precision"] - .10,
            "finite": finite,
        }},
        "Gate_F": {"name": "Subgroup and System Safety", "checks": {
            "Stop_at_least_6_of_8": stop_by["A1_PAIR_CONDITIONED"]["predicted_positive_count"] >= 6 and stop_by["A1_PAIR_CONDITIONED"]["safe_beneficial_count"] == 8,
            "HOLD_negative_FPR_no_obvious_collapse": hold_by["A1_PAIR_CONDITIONED"]["nonbeneficial_HOLD_FPR"] <= hold_by["A0_ABSOLUTE"]["nonbeneficial_HOLD_FPR"] + .05,
            "Harm_outputs_unchanged": harm_isolation["harm_outputs_exact"],
        }},
    }
    for gate in gates.values(): gate["passed"] = all(gate["checks"].values())
    gates["all_passed"] = all(gate["passed"] for gate in gates.values())

    selector = {
        "label": LABEL, "mechanism_result": MECHANISM,
        "selector": ["minimum validation Benefit MAE", "maximum safe-beneficial sign accuracy", "earlier epoch"],
        "pre_registered": True,
        "A0": {"selected_epoch": int(a0_result["selected"]["epoch"]), "epochs_completed": a0_result["epochs_completed"], "selected_validation_MAE": a0_result["selected"]["Benefit_MAE"], "selected_safe_sign": a0_result["selected"]["safe_beneficial_sign_accuracy"]},
        "A1": {"selected_epoch": int(a1_result["selected"]["epoch"]), "epochs_completed": a1_result["epochs_completed"], "selected_validation_MAE": a1_result["selected"]["Benefit_MAE"], "selected_safe_sign": a1_result["selected"]["safe_beneficial_sign_accuracy"]},
    }
    torch.save({"label": LABEL, "mechanism_result": MECHANISM, "model": "A0_ABSOLUTE", "state_dict": a0_result["head"].state_dict(), "selector": selector["A0"], "test_reads": 0}, args.output_dir / "checkpoints/a0_absolute.pt")
    torch.save({"label": LABEL, "mechanism_result": MECHANISM, "model": "A1_PAIR_CONDITIONED", "state_dict": a1_result["head"].state_dict(), "selector": selector["A1"], "test_reads": 0}, args.output_dir / "checkpoints/a1_pair_conditioned.pt")

    frozen_contract = {
        "label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE, "test_reads": 0,
        "checksums_before": checksums_before, "checksums_after": checksums_after,
        "R1_backbone_frozen": not any(parameter.requires_grad for parameter in backbone.parameters()),
        "R1_backbone_state_unchanged": backbone_before == backbone_after,
        "HARM_v3_frozen": not any(parameter.requires_grad for parameter in harm_head.parameters()),
        "ranking_branch": "Frozen B0 only", "Benefit_branch": "A1 pair-conditioned readout",
        "threshold_calibration": False, "decision_chain": False, "arbitration": False,
    }
    pair_contract = {
        "label": LABEL, "mechanism_result": MECHANISM,
        "z_i": {"shape": "[candidate,128]", "source": "frozen R1-v3-BASE output.context_embedding after final Benefit-side fusion", "runtime_valid": True},
        "z_g": {"shape": "[candidate,128]", "source": "same frozen layer at R1A runtime canonical generic candidate, repeated within episode", "runtime_valid": True},
        "pair": "concat(z_i,z_g) -> [candidate,256]", "profile_ID_input": False, "GT_input": False,
        "forbidden_inputs": ["GT future", "GT benefit", "GT harm", "GT cost", "profile ID", "oracle action"],
        "candidate_shape_train": list(frozen["train"]["context"].shape), "candidate_shape_validation": list(frozen["validation"]["context"].shape),
    }
    training_config = {
        "label": LABEL, "mechanism_result": MECHANISM, "seed": args.seed,
        "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": .001,
        "batch_size_candidate_budget": args.batch_size, "max_epochs": args.epochs, "patience": args.patience,
        "gradient_clip": 10.0, "objective": "R1B-H0 frozen-uncertainty heteroscedastic NLL on feasible candidates",
        "uncertainty": "same frozen R1-v3 per-candidate log variance for A0 and A1",
        "lambda_rank": LAMBDA_RANK, "ranking_loss_computed": False,
        "same_episode_batches_and_order": True, "batch_order_audit": batch_audit,
        "same_initialization_scheme": "torch.manual_seed(42) immediately before each PyTorch Linear construction",
        "A0_initial_state_SHA": a0_initial, "A1_initial_state_SHA": a1_initial,
        "hyperparameter_search": False,
    }
    outcome = "DPCBR SUCCESS" if gates["all_passed"] else "FROZEN SYNTHETIC REPRESENTATION INSUFFICIENT FOR RELIABLE RUNTIME-RELATIVE BENEFIT ESTIMATION"
    summary = {
        "label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE, "test_reads": 0,
        "trainable_parameters": {"A0": sum(parameter.numel() for parameter in a0_result["head"].parameters()), "A1": sum(parameter.numel() for parameter in a1_result["head"].parameters())},
        "representation_source": {"z_i": pair_contract["z_i"], "z_g": pair_contract["z_g"]},
        "safe_beneficial": {name: signs[name] for name in predictions},
        "sign_recovery": recovery_rows[0], "calibration": {name: calibrations[name] for name in predictions},
        "ranking_invariance": ranking, "C7": {row["model"]: row for row in c7_rows},
        "STOP": {row["model"]: row for row in stop_rows}, "HOLD": hold_by,
        "historical_failure_recovery": {row["model"]: row for row in historical_rows},
        "pair_weight_geometry": weight_geometry,
        "shortcut_detected": any(row["near_deterministic_shortcut"] for row in shortcuts),
        "harm_isolation": harm_isolation, "gates": gates,
        "outcome_classification": outcome, "DPCBR_formally_successful": gates["all_passed"],
        "ready_for_v3_safe_decision_chain_reconstruction": gates["all_passed"],
        "head_level_synthetic_tuning_must_stop": not gates["all_passed"], "next_stage_started": False,
    }

    io.write_json(args.output_dir / "frozen_contract.json", frozen_contract)
    io.write_json(args.output_dir / "pair_input_contract.json", pair_contract)
    io.write_json(args.output_dir / "a0_architecture.json", {"label": LABEL, "mechanism_result": MECHANISM, **a0_result["head"].architecture_audit()})
    io.write_json(args.output_dir / "a1_architecture.json", {"label": LABEL, "mechanism_result": MECHANISM, **a1_result["head"].architecture_audit()})
    io.write_json(args.output_dir / "training_config.json", training_config)
    io.write_csv(args.output_dir / "a0_training_curve.csv", a0_result["rows"])
    io.write_csv(args.output_dir / "a1_training_curve.csv", a1_result["rows"])
    io.write_json(args.output_dir / "selector.json", selector)
    io.write_csv(args.output_dir / "overall_comparison.csv", comparison)
    io.write_csv(args.output_dir / "safe_beneficial_sign.csv", sign_rows)
    io.write_csv(args.output_dir / "sign_recovery.csv", recovery_rows)
    io.write_csv(args.output_dir / "mae_calibration.csv", calibration)
    io.write_json(args.output_dir / "ranking_invariance.json", ranking)
    io.write_csv(args.output_dir / "c7_audit.csv", c7_rows)
    io.write_csv(args.output_dir / "stop_audit.csv", stop_rows)
    io.write_csv(args.output_dir / "hold_audit.csv", holds)
    io.write_csv(args.output_dir / "anchor_same_changed.csv", anchor_rows)
    io.write_csv(args.output_dir / "historical_failure_recovery.csv", historical_rows)
    io.write_json(args.output_dir / "pair_weight_geometry.json", weight_geometry)
    io.write_csv(args.output_dir / "shortcut_audit.csv", shortcuts)
    io.write_json(args.output_dir / "harm_isolation.json", harm_isolation)
    io.write_json(args.output_dir / "gate_results.json", gates)
    io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(io.clean(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

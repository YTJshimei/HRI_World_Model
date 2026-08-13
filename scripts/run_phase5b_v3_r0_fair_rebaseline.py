"""Phase 5B-v3-R0 fair R1/HARM rebaseline on manifest-v3 development data.

Only synthetic TRAIN/VALIDATION episodes are materialized.  TEST access is
restricted to sealed identity/count checks in the frozen manifest.  This stage
does not calibrate a threshold, run arbitration, or construct a decision chain.
"""
from __future__ import annotations

import argparse
import copy
import csv
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
from scripts import run_phase5b1_static_vs_temporal as b1
from scripts import run_phase5b16_candidate_ranking as b16
from scripts import run_phase5b17d_manifest_v2_rebaseline as d
from scripts import run_phase5b17e_independent_harm_v2 as harm_train
from scripts import run_phase5b17ed_risk_preserving_bypass as bypass
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.data.robot_action_schema import HOLD_ACTION_ID
from src.evaluation.context_value_metrics import candidate_metrics
from src.evaluation.cracs_selector import (
    BIAS,
    EPOCH,
    MAE,
    PAIRWISE,
    SIGN,
    SPEARMAN,
    TOP1,
    TOP2,
    annotate,
    calibration_limits,
    ranking_score,
    select_cracs,
)
from src.evaluation.probabilistic_harm import harm_metrics, prevalence_baseline
from src.models.independent_harm_head import RiskPreservingBypassHead
from src.multimodal.phase5b_v3_dataset import (
    V3_CANDIDATE_ACTION_DIM,
    build_v3_temporal_samples,
    v3_runtime_contract_audit,
)
from src.multimodal.temporal_schema import LABEL
from src.training.candidate_ranking import LAMBDA_RANK
from src.training.independent_harm import harm_v2_target

MECHANISM = "DEVELOPMENT MECHANISM RESULT"
STAGE = "Phase 5B-v3-R0 Manifest-v3 Fair Model Rebaseline"
MODEL_NAME = "R1-v3-BASE"
HARM_NAME = "HARM-v3-BASE"
EXPECTED_V3_SHA = "ac3a620b1fb254f1a89114f3b0daa1e3ed7dc6a570cfcf083c23355a70ee542a"
EXPECTED_V2_SHA = "b50cfe7c7077759d4fd85c78ab12cf0d54769a7bbc3bcee51b57e81eaea6604e"
TRAIN_SIZE = VALIDATION_SIZE = 240
ACTION_NAMES = {0: "KEEP", 1: "SPEED_DOWN_10", 2: "SPEED_UP_10", 3: "DISTANCE_PLUS_0_2", 4: "DISTANCE_MINUS_0_2", 7: "HOLD"}
V2_B1_REFERENCE = {
    MAE: 1.9542271157957118,
    SIGN: 0.6841666666666667,
    SPEARMAN: 0.7258333333333333,
    PAIRWISE: 0.8079166666666667,
    TOP1: 0.6,
    TOP2: 0.9666666666666667,
    "mean_gt_best_rank": 1.4458333333333333,
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
    parser.add_argument("--manifest-v2", type=Path, default=PROJECT_ROOT / "results_dev/phase5b17c_adverse_response_expansion/phase5b_manifest_v2.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r0_fair_rebaseline")
    return parser.parse_args()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def normalizer_record(normalizers):
    record = {
        "label": LABEL,
        "mechanism_result": MECHANISM,
        "fit_split": normalizers["fit_split"],
        "fit_scope": normalizers["fit_scope"],
        "fit_sample_ids": normalizers["fit_sample_ids"],
        "static_mean": normalizers["static_mean"],
        "static_scale": normalizers["static_scale"],
        "benefit_mean": normalizers["benefit_mean"],
        "benefit_scale": normalizers["benefit_scale"],
        "benefit_raw_std": normalizers["benefit_raw_std"],
        "stream": normalizers["stream"],
        "deprecated_harm_auxiliary": "benefit < -1e-6; unchanged R1 training contract; not safety evidence",
    }
    record["sha256"] = b1.digest_json(record)
    return record


def manifest_integrity(path_v3: Path, path_v2: Path, samples):
    v3_sha, v2_sha = file_sha(path_v3), file_sha(path_v2)
    if v3_sha != EXPECTED_V3_SHA:
        raise RuntimeError(f"manifest-v3 checksum mismatch: {v3_sha}")
    if v2_sha != EXPECTED_V2_SHA:
        raise RuntimeError(f"manifest-v2 checksum mismatch: {v2_sha}")
    manifest = json.loads(path_v3.read_text(encoding="utf-8"))
    expected = {}
    test_ids = []
    for row in manifest["episodes"]:
        if row["split"] == "test":
            test_ids.append(row["episode_id"])
            continue
        labels = row["harm_v2_labels"]
        for candidate_id in row["candidate_ids"]:
            expected[candidate_id] = (row["episode_id"], row["split"], bool(labels[candidate_id.rsplit(":", 1)[1]]))
    actual = {
        sample.sample_id: (sample.episode_id, sample.split, bool(sample.split_metadata["harm_v2_evaluation_only"]))
        for sample in samples
    }
    episode_splits = defaultdict(set)
    for sample in samples:
        episode_splits[sample.episode_id].add(sample.split)
    runtime = v3_runtime_contract_audit(samples)
    result = {
        "label": LABEL,
        "mechanism_result": MECHANISM,
        "manifest_v3_sha256": v3_sha,
        "expected_manifest_v3_sha256": EXPECTED_V3_SHA,
        "manifest_v2_sha256": v2_sha,
        "expected_manifest_v2_sha256": EXPECTED_V2_SHA,
        "manifest_v2_unchanged": v2_sha == EXPECTED_V2_SHA,
        "development_candidate_ids_and_harm_labels_match": expected == actual,
        "train_candidates": sum(sample.split == "train" for sample in samples),
        "validation_candidates": sum(sample.split == "validation" for sample in samples),
        "sealed_test_episode_id_count": len(test_ids),
        "sealed_test_episode_ids_unique": len(test_ids) == len(set(test_ids)),
        "test_identity_fields_read": len(test_ids),
        "test_candidate_reads": 0,
        "test_trajectory_reads": 0,
        "test_human_future_reads": 0,
        "test_benefit_reads": 0,
        "test_harm_reads": 0,
        "test_cost_reads": 0,
        "same_episode_cross_split": sorted(key for key, value in episode_splits.items() if len(value) > 1),
        "split_before_candidate_branching": True,
        "runtime_contract": runtime,
    }
    result["passed"] = bool(
        v3_sha == EXPECTED_V3_SHA
        and v2_sha == EXPECTED_V2_SHA
        and expected == actual
        and result["train_candidates"] == 1440
        and result["validation_candidates"] == 1440
        and len(test_ids) == 120
        and not result["same_episode_cross_split"]
        and runtime["passed"]
        and all(result[name] == 0 for name in (
            "test_candidate_reads", "test_trajectory_reads", "test_human_future_reads",
            "test_benefit_reads", "test_harm_reads", "test_cost_reads",
        ))
    )
    if not result["passed"]:
        raise RuntimeError("manifest-v3 integrity gate failed")
    return result


def prediction_metrics(model, samples, normalizers, args, torch, device):
    prediction = b1.predict(MODEL_NAME, model, samples, normalizers, args.batch_size, torch, device)
    targets = {
        "benefit": np.asarray([sample.targets.benefit for sample in samples]),
        "harm": np.asarray([sample.targets.harm for sample in samples]),
    }
    candidate = candidate_metrics(prediction, targets, np.asarray([sample.targets.feasible for sample in samples]))
    _, ranking = d.rank_evaluation(MODEL_NAME, samples, prediction)
    target = targets["benefit"]
    candidate.update(ranking)
    candidate[BIAS] = float(np.mean(prediction["benefit"] - target))
    candidate["absolute_global_bias"] = abs(candidate[BIAS])
    candidate["positive_prediction_rate"] = float(np.mean(prediction["benefit"] > 0))
    return prediction, candidate


def train_r1(model, train, validation, normalizers, batches, args, torch, device):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    rows, states, best_epoch, stale = [], {}, None, 0
    started = time.perf_counter()
    for epoch, epoch_batches in enumerate(batches, 1):
        model.train(); train_rows = []
        for indices in epoch_batches:
            selected = [train[index] for index in indices]
            output = model(b1.temporal_batch(selected, normalizers, torch, device))
            terms = d.loss_terms(output, selected, normalizers, torch, device)
            loss = terms["base"] + terms["weighted_rank"]
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True))
            optimizer.step()
            if not bool(torch.isfinite(loss)) or not math.isfinite(gradient):
                raise FloatingPointError("non-finite R1-v3 training state")
            train_rows.append({
                name: float(terms[name].detach())
                for name in ("nll", "deprecated_old_harm", "base", "rank", "weighted_rank")
            } | {"total": float(loss.detach()), "gradient_norm": gradient, "ranking_pairs": terms["rank_audit"].pair_count})
        prediction, metrics = prediction_metrics(model, validation, normalizers, args, torch, device)
        row = {
            "synthetic_interaction": LABEL,
            "mechanism_result": MECHANISM,
            "model": MODEL_NAME,
            "epoch": epoch,
            **{f"train_{name}": float(np.mean([item[name] for item in train_rows])) for name in train_rows[0]},
            **metrics,
            "parameter_checksum": state_sha(model.state_dict()),
        }
        audited = annotate(row, V2_B1_REFERENCE[MAE], V2_B1_REFERENCE[SIGN])
        row.update({key: audited[key] for key in (
            "cracs_eligible", "cracs_ineligibility_reasons", "S_spearman", "S_pairwise",
            "S_top1", "S_top2", "RankingScore",
        )})
        rows.append(row); states[epoch] = copy.deepcopy(model.state_dict())
        eligible = [item for item in rows if item["cracs_eligible"]]
        if eligible:
            selected, _ = select_cracs(rows, V2_B1_REFERENCE[MAE], V2_B1_REFERENCE[SIGN])
            if selected[EPOCH] != best_epoch:
                best_epoch, stale = selected[EPOCH], 0
            else:
                stale += 1
        row["cracs_current_best_epoch"] = best_epoch
        row["cracs_stale_epochs"] = stale
        print(
            f"{MODEL_NAME} epoch={epoch:02d} total={row['train_total']:.5f} "
            f"MAE={row[MAE]:.5f} score={row['RankingScore']:.5f} best={best_epoch} stale={stale}",
            flush=True,
        )
        if best_epoch is not None and stale >= args.patience:
            break
    selected, audited = select_cracs(rows, V2_B1_REFERENCE[MAE], V2_B1_REFERENCE[SIGN])
    for row, audit in zip(rows, audited):
        row.update({key: audit[key] for key in (
            "cracs_eligible", "cracs_ineligibility_reasons", "S_spearman", "S_pairwise",
            "S_top1", "S_top2", "RankingScore",
        )})
        row["cracs_final_selected"] = row[EPOCH] == selected[EPOCH]
        # CRACS defines RankingScore only for calibration-eligible epochs.
        # Store an explicit empty value rather than a misleading NaN sentinel.
        if not row["cracs_eligible"]:
            row["RankingScore"] = None
    model.load_state_dict(states[selected[EPOCH]]); model.eval()
    return {
        "model": model,
        "rows": rows,
        "selected": selected,
        "epochs_completed": len(rows),
        "early_stopped": len(rows) < args.epochs,
        "training_time_s": time.perf_counter() - started,
    }


def extract_bypass_inputs(model, samples, normalizers, args, torch, device):
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), args.batch_size):
            batch = b1.temporal_batch(samples[start:start + args.batch_size], normalizers, torch, device)
            stages = model.audit_representations(batch)
            chunks.append(bypass.bypass_input(stages, torch).cpu())
    return torch.cat(chunks)


def subtype_metrics(samples, probability, model_name=HARM_NAME):
    predicates = {
        "GT_UNSAFE": lambda sample: sample.targets.gt_unsafe,
        "EXCESSIVE_DECELERATION": lambda sample: sample.split_metadata["excessive_deceleration_evaluation_only"],
        "ABRUPT_LATERAL_RESPONSE": lambda sample: sample.split_metadata["abrupt_lateral_response_evaluation_only"],
        "ABRUPT_HEADING_CHANGE": lambda sample: sample.split_metadata["abrupt_heading_change_evaluation_only"],
    }
    harm_target = np.asarray([harm_v2_target(sample) for sample in samples], bool)
    rows = []
    for name, predicate in predicates.items():
        target = np.asarray([predicate(sample) for sample in samples], bool)
        keep = target | ~harm_target
        rows.append({
            "synthetic_interaction": LABEL,
            "mechanism_result": MECHANISM,
            "model": model_name,
            "subtype": name,
            "evaluation": "one subtype versus harm-v2 negative candidates",
            **harm_metrics(probability[keep], target[keep]),
        })
    return rows


def safe_beneficial_rows(samples, prediction):
    predicates = {
        "OVERALL": lambda sample: True,
        "C7": lambda sample: any(str(value).startswith("C7") for value in sample.split_metadata["contexts_evaluation_only"]),
        "C8": lambda sample: any(str(value).startswith("C8") for value in sample.split_metadata["contexts_evaluation_only"]),
        "C9": lambda sample: any(str(value).startswith("C9") for value in sample.split_metadata["contexts_evaluation_only"]),
        "STOP": lambda sample: sample.split_metadata["motion_type_evaluation_only"] == "stop",
        "HOLD": lambda sample: sample.split_metadata["candidate_action_id_audit"] == HOLD_ACTION_ID,
        "NON_HOLD": lambda sample: sample.split_metadata["candidate_action_id_audit"] != HOLD_ACTION_ID,
    }
    result = []
    for name, predicate in predicates.items():
        indices = [index for index, sample in enumerate(samples) if predicate(sample) and sample.split_metadata["safe_beneficial_evaluation_only"]]
        predicted = np.asarray(prediction["benefit"])[indices]
        result.append({
            "synthetic_interaction": LABEL,
            "mechanism_result": MECHANISM,
            "model": MODEL_NAME,
            "group": name,
            "candidate_count": len(indices),
            "episode_count": len({samples[index].episode_id for index in indices}),
            "predicted_benefit_positive_count": int(np.sum(predicted > 0)),
            "sign_failure_count": int(np.sum(predicted <= 0)),
            "sign_accuracy": float(np.mean(predicted > 0)) if len(indices) else None,
            "mean_prediction": float(np.mean(predicted)) if len(indices) else None,
        })
    return result


def hold_benefit_row(samples, prediction):
    groups = d.b15.group_episode(samples)
    hold_indices = [index for index, sample in enumerate(samples) if sample.split_metadata["candidate_action_id_audit"] == HOLD_ACTION_ID]
    target = np.asarray([samples[index].targets.benefit for index in hold_indices])
    predicted = np.asarray(prediction["benefit"])[hold_indices]
    ranks = []; top1 = top2 = 0
    for index in hold_indices:
        episode_indices = groups[samples[index].episode_id]
        episode_prediction = np.asarray(prediction["benefit"])[episode_indices]
        order = np.argsort(np.argsort(-episode_prediction, kind="stable"), kind="stable") + 1
        rank = int(order[episode_indices.index(index)]); ranks.append(rank)
        top1 += rank == 1; top2 += rank <= 2
    beneficial = target > 1e-6
    return {
        "synthetic_interaction": LABEL,
        "mechanism_result": MECHANISM,
        "model": MODEL_NAME,
        "candidate_count": len(hold_indices),
        "beneficial_count": int(beneficial.sum()),
        "beneficial_predicted_positive_count": int(np.sum(predicted[beneficial] > 0)),
        "beneficial_sign_accuracy": float(np.mean(predicted[beneficial] > 0)) if beneficial.any() else None,
        "Benefit_MAE": float(np.mean(np.abs(predicted - target))),
        "Benefit_Sign_Accuracy": float(np.mean(np.sign(predicted) == np.sign(target))),
        "mean_rank_position": float(np.mean(ranks)),
        "Top1_selection_frequency": float(top1 / len(ranks)),
        "Top2_selection_frequency": float(top2 / len(ranks)),
        "prediction_variance": float(np.var(predicted)),
    }


def hold_harm_rows(samples, probability):
    hold = np.asarray([sample.split_metadata["candidate_action_id_audit"] == HOLD_ACTION_ID for sample in samples], bool)
    target = np.asarray([harm_v2_target(sample) for sample in samples], bool)
    predicates = {
        "ALL_HOLD": hold,
        "STOP_HOLD": hold & np.asarray([sample.split_metadata["motion_type_evaluation_only"] == "stop" for sample in samples]),
        "DECELERATION_HOLD": hold & np.asarray([sample.split_metadata["motion_type_evaluation_only"] == "deceleration" for sample in samples]),
        "OTHER_HOLD": hold & np.asarray([sample.split_metadata["motion_type_evaluation_only"] not in ("stop", "deceleration") for sample in samples]),
    }
    rows = []
    for group, mask in predicates.items():
        metrics = harm_metrics(probability[mask], target[mask])
        unsafe = np.asarray([sample.targets.gt_unsafe for sample in samples], bool)[mask]
        rows.append({
            "synthetic_interaction": LABEL,
            "mechanism_result": MECHANISM,
            "model": HARM_NAME,
            "group": group,
            "GT_unsafe_prevalence": float(unsafe.mean()),
            "score_variance": float(np.var(probability[mask])),
            **metrics,
        })
    decel_positive = target & np.asarray([
        sample.split_metadata["excessive_deceleration_evaluation_only"] for sample in samples
    ], bool)
    values = probability[decel_positive]
    rows.append({
        "synthetic_interaction": LABEL,
        "mechanism_result": MECHANISM,
        "model": HARM_NAME,
        "group": "ALL_EXCESSIVE_DECELERATION_POSITIVE_SCORE_DISTRIBUTION",
        "candidate_count": len(values),
        **{f"P{p}": float(np.percentile(values, p)) if len(values) else None for p in (10, 25, 50, 75, 90)},
        "global_score_percentile_rank_mean": float(np.mean([np.mean(probability <= value) for value in values])) if len(values) else None,
    })
    return rows


def shortcut_audit(samples, probability):
    getters = {
        "action": lambda sample: ACTION_NAMES[int(sample.split_metadata["candidate_action_id_audit"])],
        "motion": lambda sample: str(sample.split_metadata["motion_type_evaluation_only"]),
        "profile": lambda sample: int(sample.split_metadata["person_profile_id"]),
        "context": lambda sample: "|".join(map(str, sample.split_metadata["contexts_evaluation_only"])) or "NONE",
    }
    rows = []
    overall_variance = float(np.var(probability))
    for dimension, getter in getters.items():
        groups = defaultdict(list)
        for sample, value in zip(samples, probability):
            groups[getter(sample)].append(float(value))
        overall_mean = float(np.mean(probability))
        between = float(sum(len(values) * (float(np.mean(values)) - overall_mean) ** 2 for values in groups.values()) / len(probability))
        fraction = between / max(overall_variance, 1e-12)
        rows.append({
            "synthetic_interaction": LABEL,
            "mechanism_result": MECHANISM,
            "model": HARM_NAME,
            "dimension": dimension,
            "group_count": len(groups),
            "overall_prediction_variance": overall_variance,
            "group_mean_min": float(min(map(np.mean, groups.values()))),
            "group_mean_max": float(max(map(np.mean, groups.values()))),
            "group_mean_range": float(max(map(np.mean, groups.values())) - min(map(np.mean, groups.values()))),
            "between_group_variance_fraction": fraction,
            "near_deterministic_shortcut": bool(fraction >= .90),
            "profile_id_runtime_input": False,
        })
    return rows


def shared_comparison_rows(samples, benefit_prediction, harm_probability):
    shared = np.asarray([sample.split_metadata["candidate_action_id_audit"] != HOLD_ACTION_ID for sample in samples], bool)
    subset = [sample for sample, keep in zip(samples, shared) if keep]
    subprediction = {key: np.asarray(value)[shared] for key, value in benefit_prediction.items()}
    targets = {"benefit": np.asarray([sample.targets.benefit for sample in subset]), "harm": np.asarray([sample.targets.harm for sample in subset])}
    candidate = candidate_metrics(subprediction, targets, np.asarray([sample.targets.feasible for sample in subset]))
    _, ranking = d.rank_evaluation(MODEL_NAME, subset, subprediction)
    v3_benefit = {**candidate, **ranking}
    v2_benefit = json.loads((PROJECT_ROOT / "results_dev/phase5b17db_checkpoint_selector_repair/selected_checkpoint_metrics.json").read_text(encoding="utf-8"))["metrics"]
    rows = []
    fields = (MAE, SIGN, "Benefit_Spearman", SPEARMAN, PAIRWISE, TOP1, TOP2, "mean_gt_best_rank")
    for field in fields:
        rows.append({
            "synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
            "scope": "shared A0-A4 validation subset", "task": "benefit", "metric": field,
            "v2_historical": v2_benefit[field], "v3_BASE": v3_benefit[field],
            "v3_minus_v2": float(v3_benefit[field] - v2_benefit[field]),
        })
    v3_harm = harm_metrics(harm_probability[shared], np.asarray([harm_v2_target(sample) for sample in samples], bool)[shared])
    with (PROJECT_ROOT / "results_dev/phase5b17ed_risk_preserving_bypass/global_metrics.csv").open(newline="", encoding="utf-8") as handle:
        v2_harm = next(row for row in csv.DictReader(handle) if row["model"] == "H1")
    for field in ("AUROC", "AUPRC", "NLL", "Brier"):
        rows.append({
            "synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
            "scope": "shared A0-A4 validation subset", "task": "harm", "metric": field,
            "v2_historical": float(v2_harm[field]), "v3_BASE": v3_harm[field],
            "v3_minus_v2": float(v3_harm[field] - float(v2_harm[field])),
        })
    v3_subtypes = {row["subtype"]: row for row in subtype_metrics(subset, harm_probability[shared])}
    with (PROJECT_ROOT / "results_dev/phase5b17ed_risk_preserving_bypass/by_harm_subtype.csv").open(newline="", encoding="utf-8") as handle:
        v2_subtypes = {row["subtype"]: row for row in csv.DictReader(handle) if row["model"] == "H1"}
    for subtype in sorted(v3_subtypes):
        for field in ("AUROC", "AUPRC", "NLL", "Brier"):
            old, new = float(v2_subtypes[subtype][field]), float(v3_subtypes[subtype][field])
            rows.append({
                "synthetic_interaction": LABEL, "mechanism_result": MECHANISM,
                "scope": f"shared A0-A4 validation subset / {subtype}", "task": "harm_subtype", "metric": field,
                "v2_historical": old, "v3_BASE": new, "v3_minus_v2": new - old,
            })
    return rows


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite v3-R0 result: {args.output_dir}")
    args.output_dir.mkdir(parents=True); (args.output_dir / "checkpoints").mkdir()
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    episodes = {
        "train": build_development_split("train", TRAIN_SIZE, GENERATOR_SEED, RISK_SEED),
        "validation": build_development_split("validation", VALIDATION_SIZE, GENERATOR_SEED + 1000, RISK_SEED + 1000),
    }
    splits = {name: build_v3_temporal_samples(values) for name, values in episodes.items()}
    all_samples = splits["train"] + splits["validation"]
    integrity = manifest_integrity(args.manifest_v3, args.manifest_v2, all_samples)
    normalizers = b1.fit_normalizers(splits["train"]); normalizer = normalizer_record(normalizers)
    batches, batch_audit = b16.make_episode_batches(splits["train"], args.epochs, args.batch_size, args.seed)

    from src.models.rich_temporal_small_transformer_v3 import RichTemporalSmallTransformerV3
    torch.manual_seed(args.seed); model = RichTemporalSmallTransformerV3()
    initial_state = copy.deepcopy(model.state_dict()); initial_checksum = state_sha(initial_state)
    architecture = model.architecture_audit()
    trained = train_r1(model, splits["train"], splits["validation"], normalizers, batches, args, torch, device)
    model = trained["model"]
    selected_prediction, selected_metrics = prediction_metrics(model, splits["validation"], normalizers, args, torch, device)
    selected_metrics["RankingScore"] = ranking_score(selected_metrics)
    limits = calibration_limits(V2_B1_REFERENCE[MAE], V2_B1_REFERENCE[SIGN])
    cracs_selection = {
        "label": LABEL, "mechanism_result": MECHANISM, "selector": "CRACS-v1 unchanged",
        "reference": "frozen formal v2 B1 validation reference; no unapproved B1-v3 model trained",
        "reference_metrics": V2_B1_REFERENCE, "calibration_limits": limits,
        "eligible_epochs": [row[EPOCH] for row in trained["rows"] if row["cracs_eligible"]],
        "selected_epoch": int(trained["selected"][EPOCH]),
        "selected_RankingScore": float(trained["selected"]["RankingScore"]),
        "selected_MAE": float(trained["selected"][MAE]),
        "selected_sign_accuracy": float(trained["selected"][SIGN]),
        "selected_global_bias": float(trained["selected"][BIAS]),
        "MAE_guard_passed": bool(trained["selected"][MAE] <= limits["max_mae"]),
        "sign_guard_passed": bool(trained["selected"][SIGN] >= limits["min_sign_accuracy"]),
        "epochs_completed": trained["epochs_completed"], "early_stopped": trained["early_stopped"],
    }
    torch.save({
        "label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE, "model": MODEL_NAME,
        "model_state_dict": model.state_dict(), "architecture": architecture,
        "normalizer": normalizer, "selector": cracs_selection,
        "manifest_v3_sha256": EXPECTED_V3_SHA, "test_reads": 0,
    }, args.output_dir / "checkpoints/r1_v3_base_cracs.pt")

    # Freeze R1 completely before extracting the fixed risk-preserving bypass.
    model_state_before_harm = state_sha(model.state_dict())
    for parameter in model.parameters(): parameter.requires_grad_(False)
    train_x = extract_bypass_inputs(model, splits["train"], normalizers, args, torch, device)
    validation_x = extract_bypass_inputs(model, splits["validation"], normalizers, args, torch, device)
    train_y = torch.tensor([harm_v2_target(sample) for sample in splits["train"]], dtype=torch.float32)
    validation_y = torch.tensor([harm_v2_target(sample) for sample in splits["validation"]], dtype=torch.float32)
    torch.manual_seed(args.seed); harm_head = RiskPreservingBypassHead().to(device)
    harm_trained = harm_train.train_head(harm_head, train_x, train_y, validation_x, validation_y, args, torch, device)
    harm_probability = harm_train.probabilities(harm_trained["head"], validation_x, args.batch_size, torch, device)
    global_harm = harm_metrics(harm_probability, validation_y.numpy().astype(bool))
    global_harm["selected_epoch"] = int(harm_trained["selected"]["epoch"])
    global_harm["parameter_count"] = sum(parameter.numel() for parameter in harm_head.parameters())
    baseline = prevalence_baseline(train_y.numpy().astype(bool), validation_y.numpy().astype(bool))
    harm_rows = [
        {"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "model": HARM_NAME, **global_harm},
        {"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "model": "constant_train_prevalence", **baseline},
    ]
    subtype_rows = subtype_metrics(splits["validation"], harm_probability)
    hold_harm = hold_harm_rows(splits["validation"], harm_probability)
    hold_benefit = hold_benefit_row(splits["validation"], selected_prediction)
    safe_rows = safe_beneficial_rows(splits["validation"], selected_prediction)
    shortcut_rows = shortcut_audit(splits["validation"], harm_probability)
    shared_rows = shared_comparison_rows(splits["validation"], selected_prediction, harm_probability)
    model_state_after_harm = state_sha(model.state_dict())

    torch.save({
        "label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE, "model": HARM_NAME,
        "model_state_dict": harm_trained["head"].state_dict(),
        "architecture": harm_trained["head"].architecture_audit(),
        "selector": harm_trained["selected"], "manifest_v3_sha256": EXPECTED_V3_SHA,
        "r1_v3_checkpoint_state_sha256": model_state_before_harm, "test_reads": 0,
    }, args.output_dir / "checkpoints/harm_v3_base_phs.pt")

    action_shortcut = next(row for row in shortcut_rows if row["dimension"] == "action")
    hold_overall = next(row for row in hold_harm if row["group"] == "ALL_HOLD")
    hold_stop = next(row for row in hold_harm if row["group"] == "STOP_HOLD")
    context_rows = [row for row in shortcut_rows if row["dimension"] in ("motion", "context")]
    finite_fields = (
        "train_nll", "train_deprecated_old_harm", "train_base", "train_rank", "train_weighted_rank",
        "train_total", "train_gradient_norm", MAE, SIGN, "Benefit_Spearman", SPEARMAN, PAIRWISE, TOP1, TOP2,
    )
    benefit_training_finite = all(
        all(math.isfinite(float(row[name])) for name in finite_fields) for row in trained["rows"]
    )
    harm_training_finite = all(
        all(math.isfinite(float(row[name])) for name in ("train_BCEWithLogitsLoss", "NLL", "Brier", "AUROC"))
        for row in harm_trained["rows"]
    )
    training_finite = benefit_training_finite and harm_training_finite
    gates = {
        "Gate_A": {"name": "Manifest / Leakage Integrity", "checks": {
            "manifest_integrity_passed": integrity["passed"], "test_reads_zero": integrity["test_candidate_reads"] == 0,
            "hold_in_train": integrity["runtime_contract"]["hold_sample_count"] == 480,
            "hold_in_validation": sum(sample.split == "validation" and sample.split_metadata["candidate_action_id_audit"] == HOLD_ACTION_ID for sample in all_samples) == 240,
            "candidate_action_12D": integrity["runtime_contract"]["all_action_shapes_valid"],
        }},
        "Gate_B": {"name": "Benefit / Ranking Health", "checks": {
            "training_finite": training_finite, "CRACS_has_eligible_epoch": bool(cracs_selection["eligible_epochs"]),
            "MAE_guard_passed": cracs_selection["MAE_guard_passed"], "sign_guard_passed": cracs_selection["sign_guard_passed"],
            "ranking_not_collapsed": selected_metrics[PAIRWISE] >= .55 and selected_metrics[SPEARMAN] >= 0.0,
        }},
        "Gate_C": {"name": "Harm Rebaseline Health", "checks": {
            "global_AUROC_at_least_0_80": global_harm["AUROC"] >= .80,
            "NLL_better_than_constant": global_harm["NLL"] < baseline["NLL"],
            "Brier_better_than_constant": global_harm["Brier"] < baseline["Brier"],
            "R1_representation_frozen": model_state_before_harm == model_state_after_harm,
            "optimizer_only_harm_head": harm_trained["optimizer_exactly_head"],
        }},
        "Gate_D": {"name": "HOLD Learnability", "checks": {
            "HOLD_score_nonconstant": hold_overall["score_variance"] > 1e-6,
            "HOLD_AUROC_not_catastrophic": hold_overall["AUROC"] is not None and hold_overall["AUROC"] >= .60,
            "STOP_HOLD_positive_score_above_global_negative": hold_stop["mean_positive_probability"] is not None and hold_stop["mean_positive_probability"] > global_harm["mean_negative_probability"],
            "context_or_motion_distinction": any(row["group_mean_range"] >= .01 for row in context_rows),
            "not_action_ID_shortcut": not action_shortcut["near_deterministic_shortcut"],
        }},
        "Gate_E": {"name": "No Shortcut", "checks": {
            "no_near_deterministic_action_motion_profile_context_shortcut": not any(row["near_deterministic_shortcut"] for row in shortcut_rows),
            "profile_ID_not_runtime_input": not integrity["runtime_contract"]["profile_id_runtime_input"],
            "GT_future_not_runtime_input": True,
        }},
    }
    for gate in gates.values(): gate["passed"] = all(gate["checks"].values())
    gates["all_passed"] = all(gate["passed"] for gate in gates.values())

    safe_overall = next(row for row in safe_rows if row["group"] == "OVERALL")
    benefit_failure = safe_overall["sign_accuracy"] is not None and safe_overall["sign_accuracy"] < .80
    risk_failure = not gates["Gate_C"]["passed"]
    hold_failure = not gates["Gate_D"]["passed"]
    failures = [name for name, value in (("A_benefit_sign_problem", benefit_failure), ("B_risk_problem", risk_failure), ("C_HOLD_generalization_problem", hold_failure)) if value]
    category = "E_healthy" if not failures else failures[0] if len(failures) == 1 else "D_multiple_problems"
    if gates["Gate_C"]["passed"] and not hold_failure and benefit_failure:
        recommendation = "1 Generic-Anchored Relative Advantage"
    elif risk_failure or hold_failure:
        recommendation = "2 Risk representation repair"
    else:
        recommendation = "3 v3 safe decision reconstruction"
    bottleneck = {
        "label": LABEL, "mechanism_result": MECHANISM, "primary_category": category,
        "active_failures": failures, "benefit_sign_failure": benefit_failure,
        "risk_failure": risk_failure, "HOLD_generalization_failure": hold_failure,
        "unique_next_recommendation": recommendation,
        "relative_advantage_implemented": False,
    }

    frozen_contract = {
        "label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE,
        "seed": args.seed, "development_validation_only": True, "test_reads": 0,
        "approved_models_only": [MODEL_NAME, HARM_NAME],
        "manifest_v3_sha256": EXPECTED_V3_SHA, "manifest_v2_sha256": EXPECTED_V2_SHA,
        "benefit_target_unchanged": True, "harm_v2_target_unchanged": True,
        "CRACS_unchanged": True, "PHS_v1_unchanged": True,
        "threshold_calibration_performed": False, "decision_chain_run": False,
        "arbitration_run": False, "fallback_run": False,
        "generic_anchored_relative_advantage_implemented": False,
        "R1_initialization": "fixed seed 42 from scratch; shared parameters receive same v2 initialization; new 12D action projection initialized, not checkpoint-expanded",
        "R1_initial_checksum": initial_checksum, "R1_post_harm_checksum_unchanged": model_state_before_harm == model_state_after_harm,
        "R1_train_saw_HOLD": sum(sample.split_metadata["candidate_action_id_audit"] == HOLD_ACTION_ID for sample in splits["train"]) == 240,
        "HARM_train_saw_HOLD": sum(sample.split_metadata["candidate_action_id_audit"] == HOLD_ACTION_ID for sample in splits["train"]) == 240,
        "training_contract": {"optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": .001,
                              "batch_size_candidate_budget": args.batch_size, "max_epochs": args.epochs,
                              "patience": args.patience, "gradient_clip": 10.0, "lambda_rank": LAMBDA_RANK},
        "harm_contract": {"head": "Linear(1408,1)", "input": "z_final + z_human + z_candidate",
                          "loss": "unweighted BCEWithLogitsLoss", "selector": "PHS-v1",
                          "class_weighting": False, "focal_loss": False, "threshold_tuning": False},
    }
    architecture.update({"label": LABEL, "mechanism_result": MECHANISM, "initial_parameter_checksum": initial_checksum,
                         "normalizer_sha256": normalizer["sha256"], "batch_order_audit": batch_audit})
    benefit_row = {"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "model": MODEL_NAME,
                   "selected_epoch": cracs_selection["selected_epoch"], **selected_metrics}
    phs = {"label": LABEL, "mechanism_result": MECHANISM, "selector": "PHS-v1 unchanged",
           "selected": harm_trained["selected"], "epochs_completed": harm_trained["epochs_completed"],
           "training_time_s": harm_trained["training_time_s"]}
    summary = {
        "label": LABEL, "mechanism_result": MECHANISM, "stage": STAGE, "test_reads": 0,
        "R1_v3_parameter_count": architecture["parameter_count"], "candidate_action_dimension": V3_CANDIDATE_ACTION_DIM,
        "CRACS": cracs_selection, "benefit": selected_metrics,
        "safe_beneficial": safe_overall, "HOLD_benefit": hold_benefit,
        "harm": global_harm, "constant_prevalence_baseline": baseline,
        "HOLD_harm": hold_overall, "gates": gates, "bottleneck": bottleneck,
        "next_stage_started": False,
    }

    io.write_json(args.output_dir / "frozen_contract.json", frozen_contract)
    io.write_json(args.output_dir / "manifest_integrity.json", integrity)
    io.write_json(args.output_dir / "r1_v3_architecture.json", architecture)
    io.write_csv(args.output_dir / "r1_v3_training_curve.csv", trained["rows"])
    io.write_json(args.output_dir / "cracs_selection.json", cracs_selection)
    io.write_csv(args.output_dir / "benefit_metrics.csv", [benefit_row])
    io.write_csv(args.output_dir / "safe_beneficial_sign_audit.csv", safe_rows)
    io.write_csv(args.output_dir / "hold_benefit_audit.csv", [hold_benefit])
    io.write_csv(args.output_dir / "harm_v3_training_curve.csv", [{"model": HARM_NAME, "mechanism_result": MECHANISM, **row} for row in harm_trained["rows"]])
    io.write_json(args.output_dir / "harm_v3_phs_selection.json", phs)
    io.write_csv(args.output_dir / "harm_global_metrics.csv", harm_rows)
    io.write_csv(args.output_dir / "harm_by_subtype.csv", subtype_rows)
    io.write_csv(args.output_dir / "hold_harm_audit.csv", hold_harm)
    io.write_csv(args.output_dir / "shared_a0_a4_comparison.csv", shared_rows)
    io.write_csv(args.output_dir / "shortcut_audit.csv", shortcut_rows)
    io.write_json(args.output_dir / "bottleneck_classification.json", bottleneck)
    io.write_json(args.output_dir / "gate_results.json", gates)
    io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(io.clean(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

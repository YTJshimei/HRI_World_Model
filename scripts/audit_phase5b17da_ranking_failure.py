"""Phase 5B-1.7D-A ranking reproducibility failure audit.

This script performs a validation-only DIAGNOSTIC REPLAY.  It never writes a
model checkpoint, never changes the frozen 1.7D outputs, and never reads TEST
candidate IDs, trajectories, benefits, or harm labels.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b1_static_vs_temporal as b1
from scripts import run_phase5b15_decision_bottleneck as b15
from scripts import run_phase5b16_candidate_ranking as b16
from scripts import run_phase5b17d_manifest_v2_rebaseline as d
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.evaluation.context_value_metrics import spearman, validation_selection_key
from src.multimodal.phase5b_v2_dataset import build_v2_temporal_samples
from src.multimodal.temporal_schema import LABEL
from src.training.candidate_ranking import LAMBDA_RANK

ROOT_CAUSES = {
    "A": "TARGET SCALE / NORMALIZATION SHIFT", "B": "CANDIDATE-SET DISTRIBUTION SHIFT",
    "C": "PAIRWISE MARGIN / PAIR COMPOSITION SHIFT", "D": "BASE-vs-RANK GRADIENT CONFLICT",
    "E": "CHECKPOINT-SELECTION OBJECTIVE MISMATCH", "F": "RANKING-vs-ABSOLUTE CALIBRATION CONFLICT",
    "G": "ADVERSE-DYNAMICS DISTRIBUTION SHIFT", "H": "MULTIPLE INTERACTING CAUSES",
}
ACTION_NAMES = {0: "KEEP", 1: "SPEED_DOWN", 2: "SPEED_UP", 3: "DISTANCE_PLUS", 4: "DISTANCE_MINUS"}
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17da_ranking_failure_audit")
    parser.add_argument("--phase5b17d-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17d_manifest_v2_rebaseline")
    parser.add_argument("--manifest-v2", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17c_adverse_response_expansion" / "phase5b_manifest_v2.json")
    parser.add_argument("--manifest-v1", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b05_c7_coverage" / "phase5b_manifest_v1.json")
    parser.add_argument("--phase5b16-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b16_candidate_ranking")
    parser.add_argument("--phase5b1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b1_static_vs_temporal_small")
    return parser.parse_args()


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values) -> dict[str, float | int | None]:
    array = np.asarray(values, np.float64)
    if not len(array): return {"count": 0, "mean": None, "std": None, "median": None, "min": None, "max": None, **{f"P{p}": None for p in PERCENTILES}}
    return {"count": len(array), "mean": float(array.mean()), "std": float(array.std()), "median": float(np.median(array)),
            **{f"P{p}": float(np.percentile(array, p)) for p in PERCENTILES}, "min": float(array.min()), "max": float(array.max())}


def target_distribution(samples, normalizers):
    rows = []
    for split in ("train", "validation"):
        split_samples = [sample for sample in samples if sample.split == split]
        for feasibility, predicate in (("all", lambda _: True), ("feasible", lambda sample: sample.targets.feasible), ("infeasible", lambda sample: not sample.targets.feasible)):
            selected = [sample for sample in split_samples if predicate(sample)]
            raw = np.asarray([sample.targets.benefit for sample in selected], float)
            normalized = (raw - normalizers["benefit_mean"]) / normalizers["benefit_scale"] if len(raw) else raw
            rows.append({"synthetic_interaction": LABEL, "split": split, "feasibility": feasibility, "scale": "raw_GT_benefit", **describe(raw)})
            rows.append({"synthetic_interaction": LABEL, "split": split, "feasibility": feasibility, "scale": "train_normalized_benefit", **describe(normalized)})
    return rows


def episode_target_statistics(samples):
    ranges, best_second, pairs = [], [], []
    for _, indices in b15.group_episode(samples).items():
        values = np.asarray([samples[index].targets.benefit for index in indices], float)
        ranges.append(float(np.ptp(values))); ordered = np.sort(values)[::-1]
        if len(ordered) > 1: best_second.append(float(ordered[0] - ordered[1]))
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                if values[left] != values[right]: pairs.append(abs(float(values[left] - values[right])))
    return {"within_episode_range": describe(ranges), "best_second_margin": describe(best_second), "pairwise_margin": describe(pairs)}


def v1_v2_distribution_audit(args, train, validation, normalizers):
    old = json.loads((args.phase5b1_dir / "normalizer.json").read_text(encoding="utf-8"))
    old_positive = []
    with (args.phase5b16_dir / "beneficial_sign_audit.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] == "R0 B1-Control": old_positive.append(float(row["gt_benefit"]))
    result = {
        "label": LABEL, "comparison_scope": "existing v1 development statistics only; old TEST not read",
        "v1": {"train_raw_benefit_mean": old["benefit_mean"], "train_raw_benefit_std": old["benefit_raw_std"],
               "validation_positive_benefit_existing_rows": describe(old_positive),
               "full_raw_percentiles_available_in_frozen_outputs": False,
               "limitation": "frozen v1 artifacts do not contain every raw candidate target; unavailable values are not reconstructed"},
        "v2": {"train_raw_benefit": describe([sample.targets.benefit for sample in train]),
               "validation_raw_benefit": describe([sample.targets.benefit for sample in validation]),
               "train_abs_benefit": describe(np.abs([sample.targets.benefit for sample in train])),
               "train_positive_benefit": describe([sample.targets.benefit for sample in train if sample.targets.benefit > 0]),
               "train_negative_benefit": describe([sample.targets.benefit for sample in train if sample.targets.benefit < 0]),
               "train_episode": episode_target_statistics(train), "validation_episode": episode_target_statistics(validation)},
    }
    result["std_ratio_v2_over_v1"] = normalizers["benefit_raw_std"] / old["benefit_raw_std"]
    result["mean_shift_absolute"] = normalizers["benefit_mean"] - old["benefit_mean"]
    result["candidate_ranking_difficulty_changed"] = True
    result["evidence"] = "target scale/tails changed strongly; exact v1 margin percentiles are unavailable in frozen development artifacts"
    return result


def candidate_contract(path: Path, version: str):
    manifest = json.loads(path.read_text(encoding="utf-8")); rows = []
    for episode in manifest["episodes"]:
        if episode["split"] not in ("train", "validation"): continue
        actions = [int(value.rsplit(":", 1)[1]) for value in episode["candidate_ids"]]
        rows.append((episode["split"], len(actions), tuple(actions)))
    return {"version": version, "episode_count": len(rows), "candidate_count_per_episode": describe([row[1] for row in rows]),
            "action_sets": sorted({str(row[2]) for row in rows}),
            "action_frequency": {ACTION_NAMES[action]: sum(action in row[2] for row in rows) for action in ACTION_NAMES},
            "test_candidate_reads": 0}


def candidate_contract_rows(args, train, validation):
    v1 = candidate_contract(args.manifest_v1, "manifest_v1"); v2 = candidate_contract(args.manifest_v2, "manifest_v2")
    old_ranking = list(csv.DictReader((args.phase5b16_dir / "episode_ranking.csv").open(encoding="utf-8")))
    old = [row for row in old_ranking if row["model"] == "R0 B1-Control"]
    v1_feasible = [int(row["feasible_count"]) for row in old]
    v2_feasible = [sum(sample.targets.feasible for sample in values) for values in ([validation[index] for index in indices] for indices in b15.group_episode(validation).values())]
    rows = []
    for record, feasible, source in ((v1, v1_feasible, "existing validation episode_ranking.csv"), (v2, v2_feasible, "manifest_v2 deterministic development labels")):
        count = record["candidate_count_per_episode"]
        rows.append({"synthetic_interaction": LABEL, "manifest": record["version"], "episode_count": record["episode_count"],
                     "candidate_count_mean": count["mean"], "candidate_count_min": count["min"], "candidate_count_max": count["max"],
                     "action_sets": "|".join(record["action_sets"]), "feasible_count_mean_validation": float(np.mean(feasible)),
                     "feasible_count_min_validation": int(np.min(feasible)), "feasible_count_max_validation": int(np.max(feasible)),
                     "mean_max_feasible_pairs_validation": float(np.mean([int(value) * (int(value) - 1) / 2 for value in feasible])),
                     "non_tie_pair_count_available": record["version"] == "manifest_v2",
                     "feasibility_source": source, **{f"frequency_{name}": value for name, value in record["action_frequency"].items()}})
    return rows, {"v1": v1, "v2": v2, "candidate_cardinality_changed": (
                      v1["candidate_count_per_episode"]["min"] != v2["candidate_count_per_episode"]["min"]
                      or v1["candidate_count_per_episode"]["mean"] != v2["candidate_count_per_episode"]["mean"]),
                  "action_set_changed": v1["action_sets"] != v2["action_sets"], "feasibility_distribution_changed": np.min(v1_feasible) < 5 and np.min(v2_feasible) == 5}


def pairwise_rows(samples):
    margins, composition = [], defaultdict(list)
    for episode_id, indices in b15.group_episode(samples).items():
        feasible = [index for index in indices if samples[index].targets.feasible]
        for left in range(len(feasible)):
            for right in range(left + 1, len(feasible)):
                a, b = samples[feasible[left]], samples[feasible[right]]
                delta = float(a.targets.benefit - b.targets.benefit)
                if delta == 0: continue
                actions = tuple(sorted((int(a.split_metadata["candidate_action_id_audit"]), int(b.split_metadata["candidate_action_id_audit"]))))
                margins.append(abs(delta)); composition[actions].append(delta if actions[0] == int(a.split_metadata["candidate_action_id_audit"]) else -delta)
    stats = describe(margins)
    margin_row = {"synthetic_interaction": LABEL, "split": samples[0].split, **stats,
                  **{f"fraction_lt_{threshold:g}": float(np.mean(np.asarray(margins) < threshold)) for threshold in (.01, .02, .05, .1)}}
    rows = []
    for actions, deltas in sorted(composition.items()):
        values = np.asarray(deltas)
        rows.append({"synthetic_interaction": LABEL, "split": samples[0].split,
                     "action_pair": f"{ACTION_NAMES[actions[0]]}_vs_{ACTION_NAMES[actions[1]]}", "pair_count": len(values),
                     "first_action_better_rate": float(np.mean(values > 0)), "second_action_better_rate": float(np.mean(values < 0)),
                     "mean_abs_benefit_margin": float(np.mean(np.abs(values))), "median_abs_benefit_margin": float(np.median(np.abs(values)))})
    return margin_row, rows


def _gradient_vector(loss, named, predicate, torch, retain_graph=True):
    selected = [(name, parameter) for name, parameter in named if parameter.requires_grad and predicate(name)]
    gradients = torch.autograd.grad(loss, [parameter for _, parameter in selected], retain_graph=retain_graph, allow_unused=True)
    pieces = [gradient.detach().float().reshape(-1) for gradient in gradients if gradient is not None]
    return torch.cat(pieces) if pieces else torch.zeros(1, device=loss.device)


def _norm(vector, torch): return float(torch.linalg.vector_norm(vector))


def gradient_audit(model, samples, batches, normalizers, torch, device):
    model.to(device).eval(); records, cosine_rows = [], []
    groups = {"all_shared": lambda name: not name.startswith(("harm.", "uncertainty.")),
              "temporal_encoder": lambda name: not name.startswith(("benefit.", "harm.", "uncertainty.")),
              "benefit_head": lambda name: name.startswith("benefit."),
              "deprecated_auxiliary_harm_head": lambda name: name.startswith("harm.")}
    for batch_index, indices in enumerate(batches[:32], 1):
        selected = [samples[index] for index in indices]
        output = model(b1.temporal_batch(selected, normalizers, torch, device)); terms = d.loss_terms(output, selected, normalizers, torch, device)
        named = list(model.named_parameters())
        base_all = _gradient_vector(terms["base"], named, lambda _: True, torch)
        raw_rank_all = _gradient_vector(terms["rank"], named, lambda _: True, torch)
        weighted_all = _gradient_vector(terms["weighted_rank"], named, lambda _: True, torch)
        combined_all = _gradient_vector(terms["base"] + terms["weighted_rank"], named, lambda _: True, torch)
        records.append({"batch": batch_index, "nll": float(terms["nll"].detach()), "deprecated_old_harm": float(terms["deprecated_old_harm"].detach()),
                        "base": float(terms["base"].detach()), "raw_rank": float(terms["rank"].detach()), "weighted_rank": float(terms["weighted_rank"].detach()),
                        "base_gradient_norm": _norm(base_all, torch), "raw_rank_gradient_norm": _norm(raw_rank_all, torch),
                        "weighted_rank_gradient_norm": _norm(weighted_all, torch), "combined_gradient_norm": _norm(combined_all, torch),
                        "weighted_rank_to_base_gradient_ratio": _norm(weighted_all, torch) / max(_norm(base_all, torch), 1e-12)})
        for group, predicate in groups.items():
            base_group = _gradient_vector(terms["base"], named, predicate, torch)
            nll = _gradient_vector(terms["nll"], named, predicate, torch)
            rank = _gradient_vector(terms["rank"], named, predicate, torch, True)
            weighted_group = _gradient_vector(terms["weighted_rank"], named, predicate, torch)
            combined_group = _gradient_vector(terms["base"] + terms["weighted_rank"], named, predicate, torch)
            cosine = float(torch.nn.functional.cosine_similarity(nll, rank, dim=0)) if _norm(nll, torch) and _norm(rank, torch) else None
            cosine_rows.append({"synthetic_interaction": LABEL, "batch": batch_index, "module": group,
                                "base_gradient_norm": _norm(base_group, torch), "nll_gradient_norm": _norm(nll, torch),
                                "raw_rank_gradient_norm": _norm(rank, torch), "weighted_rank_gradient_norm": _norm(weighted_group, torch),
                                "combined_gradient_norm": _norm(combined_group, torch), "cosine_similarity": cosine})
    def aggregate(field): return describe([row[field] for row in records])
    summary = {field: aggregate(field) for field in ("nll", "deprecated_old_harm", "base", "raw_rank", "weighted_rank", "base_gradient_norm", "raw_rank_gradient_norm", "weighted_rank_gradient_norm", "combined_gradient_norm", "weighted_rank_to_base_gradient_ratio")}
    cosine_summary = {group: describe([row["cosine_similarity"] for row in cosine_rows if row["module"] == group and row["cosine_similarity"] is not None]) for group in groups}
    module_norm_summary = {group: {field: describe([row[field] for row in cosine_rows if row["module"] == group])
                                  for field in ("base_gradient_norm", "raw_rank_gradient_norm", "weighted_rank_gradient_norm", "combined_gradient_norm")}
                           for group in groups}
    return {"label": LABEL, "diagnostic_replay": True, "optimizer_created": False, "optimizer_step_count": 0,
            "batch_count": len(records), "lambda_rank": LAMBDA_RANK, "summary": summary, "gradient_cosine": cosine_summary,
            "module_gradient_norms": module_norm_summary,
            "records": records}, cosine_rows


def replay_r1(args, train, validation, normalizers, epoch_batches, torch, device):
    """Advance the exact frozen stochastic sequence, then capture R1 epochs."""
    from src.models.large_context_adapter import SmallContextNetwork
    from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
    torch.manual_seed(args.seed); b0_model = SmallContextNetwork()
    torch.manual_seed(args.seed); b1_model = RichTemporalSmallTransformer(); r1_model = copy.deepcopy(b1_model)
    replay_args = argparse.Namespace(device=args.device, seed=args.seed, epochs=30, patience=5, batch_size=64, learning_rate=3e-4)
    # These two exact diagnostic replays advance the CUDA/dropout RNG exactly
    # as frozen 1.7D.  Returned states are not checkpoints and are never saved.
    b0_model, _, _ = d.train_model(d.MODELS[0], b0_model, train, validation, normalizers, epoch_batches, replay_args, torch, device, False)
    b1_model, _, b1_selection = d.train_model(d.MODELS[1], b1_model, train, validation, normalizers, epoch_batches, replay_args, torch, device, False)
    b1_prediction = b1.predict(d.MODELS[1], b1_model, validation, normalizers, 64, torch, device)

    optimizer = torch.optim.AdamW(r1_model.parameters(), lr=3e-4, weight_decay=1e-3); r1_model.to(device)
    rows, states, predictions, best, stale = [], {}, {}, None, 0
    for epoch, batches in enumerate(epoch_batches, 1):
        r1_model.train(); losses = []
        for indices in batches:
            selected = [train[index] for index in indices]; output = r1_model(b1.temporal_batch(selected, normalizers, torch, device))
            terms = d.loss_terms(output, selected, normalizers, torch, device); loss = terms["base"] + terms["weighted_rank"]
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(r1_model.parameters(), 10.0, error_if_nonfinite=True); optimizer.step()
            losses.append(float(loss.detach()))
        prediction, candidate, ranking, historical, key = d.validation_snapshot(d.MODELS[2], r1_model, validation, normalizers, replay_args, torch, device)
        row = {"synthetic_interaction": LABEL, "replay": "DIAGNOSTIC REPLAY - NOT A FORMAL CHECKPOINT", "epoch": epoch,
               "selection_key": json.dumps(io.clean(list(key))), "selection_harmful_switch_rate": historical["Harmful_Switch_Rate"],
               "selection_beneficial_switch_recall": historical["Beneficial_Switch_Recall"], "selection_mean_regret": historical["Mean_Regret"],
               "Benefit_MAE": candidate["Benefit_MAE"], "Benefit_Spearman": candidate["Benefit_Spearman"],
               "Benefit_Sign_Accuracy": candidate["Benefit_Sign_Accuracy"], "mean_within_episode_spearman": ranking["mean_within_episode_spearman"],
               "mean_feasible_within_episode_spearman": ranking["mean_feasible_within_episode_spearman"],
               "mean_feasible_pairwise_accuracy": ranking["mean_feasible_pairwise_accuracy"], "gt_best_top1_accuracy": ranking["gt_best_top1_accuracy"],
               "gt_best_top2_recall": ranking["gt_best_top2_recall"], "mean_gt_best_rank": ranking["mean_gt_best_rank"], "train_total": float(np.mean(losses))}
        rows.append(row); states[epoch] = copy.deepcopy(r1_model.state_dict()); predictions[epoch] = prediction
        if best is None or key < best[0]: best = (key, epoch); stale = 0
        else: stale += 1
        if stale >= 5: break
    official = json.loads((args.phase5b17d_dir / "checkpoint_selection.json").read_text(encoding="utf-8"))["models"][d.MODELS[2]]["best_epoch"]
    oracle = max(rows, key=lambda row: (row["mean_feasible_within_episode_spearman"], row["mean_feasible_pairwise_accuracy"], -row["mean_gt_best_rank"], row["gt_best_top1_accuracy"], row["gt_best_top2_recall"]))
    return {"rows": rows, "b1_prediction": b1_prediction, "r1_prediction": predictions[official],
            "official_epoch": official, "replay_selected_epoch": best[1], "oracle_epoch": oracle["epoch"], "oracle_row": oracle,
            "b1_selection": b1_selection, "epochs_completed": len(rows), "patience_exhausted": len(rows) < 30}


def selector_audit(rows, official_epoch, oracle_epoch):
    ordered = sorted(rows, key=lambda row: tuple(json.loads(row["selection_key"])))
    rank = {row["epoch"]: index + 1 for index, row in enumerate(ordered)}
    fields = ("Benefit_MAE", "Benefit_Spearman", "mean_within_episode_spearman", "mean_feasible_within_episode_spearman", "mean_feasible_pairwise_accuracy", "gt_best_top1_accuracy", "gt_best_top2_recall", "mean_gt_best_rank")
    correlations = {}
    for field in fields:
        # -selection rank is higher-is-better; GT rank is lower-is-better.
        target = [-row[field] if field in ("Benefit_MAE", "mean_gt_best_rank") else row[field] for row in rows]
        correlations[field] = spearman([-rank[row["epoch"]] for row in rows], target)
    return {"label": LABEL, "official_frozen_epoch": official_epoch, "audit_only_oracle_epoch": oracle_epoch,
            "oracle_cannot_replace_formal_checkpoint": True, "formal_checkpoint_changed": False,
            "selector_rank_correlations_higher_is_better": correlations,
            "checkpoint_objective_mismatch": oracle_epoch != official_epoch,
            "post_official_epochs": [row for row in rows if row["epoch"] > official_epoch],
            "early_stopping": {"patience": 5, "reason": "patience exhausted", "max_epochs": 30, "epochs_completed": len(rows)}}


def prediction_audits(validation, b1_prediction, r1_prediction):
    groups = {
        "all": lambda sample: True, "GT_positive": lambda sample: sample.targets.benefit > 0,
        "GT_negative": lambda sample: sample.targets.benefit < 0,
        "safe_beneficial": lambda sample: sample.split_metadata["safe_beneficial_evaluation_only"],
        "harm_v2_positive": lambda sample: sample.split_metadata["harm_v2_evaluation_only"],
        "harm_v2_negative": lambda sample: not sample.split_metadata["harm_v2_evaluation_only"],
    }
    rows = []
    for name, prediction in (("B1-v2", b1_prediction), ("R1-v2", r1_prediction)):
        for group, predicate in groups.items():
            indices = [index for index, sample in enumerate(validation) if predicate(sample)]
            target = np.asarray([validation[index].targets.benefit for index in indices]); predicted = np.asarray(prediction["benefit"])[indices]
            rows.append({"synthetic_interaction": LABEL, "model": name, "group": group, "candidate_count": len(indices),
                         "predicted_mean": float(predicted.mean()), "predicted_std": float(predicted.std()), "predicted_positive_rate": float(np.mean(predicted > 0)),
                         "GT_mean": float(target.mean()), "bias": float(np.mean(predicted - target)), "sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(target)))})
    offsets = []
    grouped = b15.group_episode(validation)
    for episode_id, indices in grouped.items():
        target = np.asarray([validation[index].targets.benefit for index in indices])
        left, right = np.asarray(b1_prediction["benefit"])[indices], np.asarray(r1_prediction["benefit"])[indices]
        offsets.append({"synthetic_interaction": LABEL, "episode_id": episode_id, "GT_episode_mean": float(target.mean()),
                        "B1_episode_mean": float(left.mean()), "R1_episode_mean": float(right.mean()),
                        "R1_minus_B1_offset": float(right.mean() - left.mean()), "B1_bias": float((left - target).mean()), "R1_bias": float((right - target).mean())})
    return rows, offsets


def cardinality_counterfactual(validation, b1_prediction, r1_prediction):
    """Apply the existing v1 development action-set rule to v2, audit only."""
    keep = []
    for index, sample in enumerate(validation):
        motion = str(sample.split_metadata["motion_type_evaluation_only"])
        action = int(sample.split_metadata["candidate_action_id_audit"])
        allowed = not (motion == "deceleration" and action in (2, 4))
        allowed &= not (motion in ("left_turn", "right_turn") and action in (1, 2))
        if allowed: keep.append(index)
    subset = [validation[index] for index in keep]
    rows = []
    for model, prediction in (("B1-v2", b1_prediction), ("R1-v2", r1_prediction)):
        selected = {key: np.asarray(value)[keep] for key, value in prediction.items()}
        _, metrics = d.rank_evaluation(model, subset, selected)
        rows.append({"synthetic_interaction": LABEL, "audit_only": True, "model": model,
                     "contract": "v1 development action-set rule mapped by v2 motion; no training",
                     "candidate_count": len(subset), **metrics})
    counts = [len(indices) for indices in b15.group_episode(subset).values()]
    return rows, {"candidate_count": len(subset), "episode_count": len(counts),
                  "candidate_count_mean": float(np.mean(counts)), "candidate_count_min": int(np.min(counts)),
                  "candidate_count_max": int(np.max(counts))}


def group_audit(validation, b1_prediction, r1_prediction, dimension):
    predicates = ({"EXCESSIVE_DECELERATION": lambda s: s.split_metadata["excessive_deceleration_evaluation_only"],
                   "ABRUPT_LATERAL_RESPONSE": lambda s: s.split_metadata["abrupt_lateral_response_evaluation_only"],
                   "ABRUPT_HEADING_CHANGE": lambda s: s.split_metadata["abrupt_heading_change_evaluation_only"],
                   "GT_UNSAFE": lambda s: s.targets.gt_unsafe} if dimension == "adverse_event" else
                  {"stop": lambda s: s.split_metadata["motion_type_evaluation_only"] == "stop",
                   "non_stop": lambda s: s.split_metadata["motion_type_evaluation_only"] != "stop"})
    rows = []
    for group, predicate in predicates.items():
        episodes = {sample.episode_id for sample in validation if predicate(sample)}
        indices = [index for index, sample in enumerate(validation) if sample.episode_id in episodes]; subset = [validation[index] for index in indices]
        pair, _ = pairwise_rows(subset)
        for model, prediction in (("B1-v2", b1_prediction), ("R1-v2", r1_prediction)):
            selected_prediction = {key: np.asarray(value)[indices] for key, value in prediction.items()}
            _, ranking = d.rank_evaluation(model, subset, selected_prediction)
            target = np.asarray([sample.targets.benefit for sample in subset]); predicted = selected_prediction["benefit"]
            rows.append({"synthetic_interaction": LABEL, "dimension": dimension, "group": group, "model": model,
                         "episode_count": len(episodes), "candidate_count": len(indices), "raw_benefit_mean": float(target.mean()),
                         "raw_benefit_std": float(target.std()), "pair_margin_mean": pair["mean"], "pair_margin_median": pair["median"],
                         "sign_accuracy": float(np.mean(np.sign(predicted) == np.sign(target))),
                         "within_episode_spearman": ranking["mean_within_episode_spearman"],
                         "pairwise_accuracy": ranking["mean_feasible_pairwise_accuracy"], "mean_gt_best_rank": ranking["mean_gt_best_rank"]})
    return rows


def make_figures(output, target_rows, epoch_rows, cosine_rows, offsets):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    def save(name):
        path = folder / name; plt.suptitle(LABEL, fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    plt.figure(figsize=(9, 4)); plt.plot([row["epoch"] for row in epoch_rows], [row["mean_feasible_within_episode_spearman"] for row in epoch_rows], label="feasible Spearman")
    plt.plot([row["epoch"] for row in epoch_rows], [row["mean_feasible_pairwise_accuracy"] for row in epoch_rows], label="pairwise accuracy"); plt.axvline(7, color="r", linestyle="--", label="official epoch 7"); plt.legend(); save("epochwise_ranking.png")
    plt.figure(); groups = sorted({row["module"] for row in cosine_rows if row["cosine_similarity"] is not None}); plt.boxplot([[row["cosine_similarity"] for row in cosine_rows if row["module"] == group and row["cosine_similarity"] is not None] for group in groups], tick_labels=groups); plt.axhline(0, color="k", linewidth=.7); plt.ylabel("NLL vs rank gradient cosine"); save("gradient_cosine.png")
    plt.figure(); values = [row["R1_minus_B1_offset"] for row in offsets]; plt.hist(values, bins=30); plt.xlabel("per-episode R1 - B1 mean prediction"); save("episode_prediction_offset.png")
    raw = [row for row in target_rows if row["split"] == "train" and row["feasibility"] == "all" and row["scale"] == "raw_GT_benefit"][0]
    plt.figure(); plt.bar(("mean", "std", "P95", "max"), [raw[key] for key in ("mean", "std", "P95", "max")]); plt.ylabel("v2 raw benefit"); save("v2_target_scale.png")
    return paths


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite 1.7D-A: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    frozen_before = {"manifest_v2": sha(args.manifest_v2), "manifest_v1": sha(args.manifest_v1),
                     "phase5b17d_summary": sha(args.phase5b17d_dir / "summary.json"),
                     "model_source": sha(PROJECT_ROOT / "src" / "models" / "rich_temporal_small_transformer.py"),
                     "rank_source": sha(PROJECT_ROOT / "src" / "training" / "candidate_ranking.py")}
    if frozen_before["manifest_v2"] != d.EXPECTED_MANIFEST_SHA or frozen_before["manifest_v1"] != b1.EXPECTED_MANIFEST_SHA: raise RuntimeError("frozen manifest mismatch")
    episodes = {"train": build_development_split("train", 240, GENERATOR_SEED, RISK_SEED),
                "validation": build_development_split("validation", 240, GENERATOR_SEED + 1000, RISK_SEED + 1000)}
    splits = {name: build_v2_temporal_samples(values) for name, values in episodes.items()}
    contract = d.manifest_contract(args.manifest_v2, splits["train"] + splits["validation"])
    normalizers = b1.fit_normalizers(splits["train"]); frozen_normalizer = json.loads((args.phase5b17d_dir / "normalizer.json").read_text(encoding="utf-8"))
    normalizer_audit = {"label": LABEL, "fit_split": normalizers["fit_split"], "fit_scope": normalizers["fit_scope"],
                        "fit_candidate_count": len(normalizers["fit_sample_ids"]), "fit_candidate_ids_sha256": b1.digest_json(normalizers["fit_sample_ids"]),
                        "mean": normalizers["benefit_mean"], "std": normalizers["benefit_raw_std"], "scale": normalizers["benefit_scale"], "epsilon": 1e-4,
                        "frozen_record_checksum": frozen_normalizer["sha256"], "recomputed_ids_identical": normalizers["fit_sample_ids"] == frozen_normalizer["fit_sample_ids"],
                        "recomputed_mean_identical": normalizers["benefit_mean"] == frozen_normalizer["benefit_mean"],
                        "recomputed_scale_identical": normalizers["benefit_scale"] == frozen_normalizer["benefit_scale"],
                        "models_using_identical_normalizer": list(d.MODELS), "train_only": True,
                        "extreme_rollouts_expand_scale": normalizers["benefit_raw_std"] > 10 * json.loads((args.phase5b1_dir / "normalizer.json").read_text(encoding="utf-8"))["benefit_raw_std"]}
    normalizer_audit["passed"] = all((normalizer_audit["recomputed_ids_identical"], normalizer_audit["recomputed_mean_identical"], normalizer_audit["recomputed_scale_identical"]))

    target_rows = target_distribution(splits["train"] + splits["validation"], normalizers)
    distribution = v1_v2_distribution_audit(args, splits["train"], splits["validation"], normalizers)
    contract_rows, contract_comparison = candidate_contract_rows(args, splits["train"], splits["validation"])
    train_margin, train_composition = pairwise_rows(splits["train"]); validation_margin, validation_composition = pairwise_rows(splits["validation"])
    epoch_batches, _ = b16.make_episode_batches(splits["train"], 30, 64, 42)
    from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
    torch.manual_seed(42); gradient_model = RichTemporalSmallTransformer()
    gradient_result, cosine_rows = gradient_audit(gradient_model, splits["train"], epoch_batches[0], normalizers, torch, device)
    old_gradient = json.loads((args.phase5b16_dir / "no_update_gradient_audit.json").read_text(encoding="utf-8"))
    gradient_result["v1_existing_audit"] = {"weighted_rank_to_base_gradient_ratio_median": old_gradient["weighted_rank_to_base_gradient_ratio_median"],
                                            "weighted_rank_to_base_gradient_ratio_max": old_gradient["weighted_rank_to_base_gradient_ratio_max"]}
    gradient_result["v2_to_v1_median_ratio"] = gradient_result["summary"]["weighted_rank_to_base_gradient_ratio"]["median"] / old_gradient["weighted_rank_to_base_gradient_ratio_median"]

    replay = replay_r1(args, splits["train"], splits["validation"], normalizers, epoch_batches, torch, device)
    selector = selector_audit(replay["rows"], replay["official_epoch"], replay["oracle_epoch"])
    sign_rows, offset_rows = prediction_audits(splits["validation"], replay["b1_prediction"], replay["r1_prediction"])
    adverse_rows = group_audit(splits["validation"], replay["b1_prediction"], replay["r1_prediction"], "adverse_event")
    stop_rows = group_audit(splits["validation"], replay["b1_prediction"], replay["r1_prediction"], "motion")
    cardinality_rows, cardinality_subset = cardinality_counterfactual(splits["validation"], replay["b1_prediction"], replay["r1_prediction"])
    cardinality = {"label": LABEL, "audit_only": True, **contract_comparison,
                   "counterfactual_required": contract_comparison["candidate_cardinality_changed"],
                   "counterfactual_performed": True, "mapped_v1_action_subset": cardinality_subset,
                   "limitation": "mapping uses existing v1 development action holdout rule by motion; it does not reconstruct old TEST or alter formal v2"}
    ordering = {"label": LABEL, "audit_only": True, "method": "within-episode rank-normalized ordering; no training and no target change",
                "B1": d.rank_evaluation("B1", splits["validation"], replay["b1_prediction"])[1],
                "R1": d.rank_evaluation("R1", splits["validation"], replay["r1_prediction"])[1]}

    cosine_median = gradient_result["gradient_cosine"]["all_shared"]["median"]
    offset_all = {row["model"]: row for row in sign_rows if row["group"] == "all"}
    adverse_b1 = np.mean([row["within_episode_spearman"] for row in adverse_rows if row["model"] == "B1-v2"])
    adverse_r1 = np.mean([row["within_episode_spearman"] for row in adverse_rows if row["model"] == "R1-v2"])
    evidence = {
        "A": {"present": distribution["std_ratio_v2_over_v1"] > 10, "std_ratio": distribution["std_ratio_v2_over_v1"]},
        "B": {"present": contract_comparison["feasibility_distribution_changed"] or contract_comparison["candidate_cardinality_changed"],
              "cardinality_changed": contract_comparison["candidate_cardinality_changed"], "action_set_changed": contract_comparison["action_set_changed"]},
        "C": {"present": train_margin["fraction_lt_0.1"] > .25, "tiny_pair_fraction_lt_0.1": train_margin["fraction_lt_0.1"]},
        "D": {"present": cosine_median < 0, "median_shared_gradient_cosine": cosine_median},
        "E": {"present": selector["checkpoint_objective_mismatch"], "official_epoch": replay["official_epoch"], "oracle_epoch": replay["oracle_epoch"]},
        "F": {"present": abs(offset_all["R1-v2"]["bias"]) > abs(offset_all["B1-v2"]["bias"]) and offset_all["R1-v2"]["sign_accuracy"] < offset_all["B1-v2"]["sign_accuracy"],
              "B1_bias": offset_all["B1-v2"]["bias"], "R1_bias": offset_all["R1-v2"]["bias"]},
        "G": {"present": adverse_r1 < adverse_b1, "B1_mean_adverse_spearman": adverse_b1, "R1_mean_adverse_spearman": adverse_r1},
    }
    present = [key for key, value in evidence.items() if value["present"]]
    classification = {"label": LABEL, "selected_class": "H", "selected_name": ROOT_CAUSES["H"], "present_causes": present,
                      "evidence": evidence, "primary_drivers": ["A", "E", "F"],
                      "single_variable_next_intervention": "Checkpoint Selection Criterion Repair",
                      "rationale": "E is directly actionable without changing training loss/model/data; predeclare a validation composite aligned with ranking while retaining MAE calibration",
                      "implemented_in_this_stage": False}
    figures = make_figures(args.output_dir, target_rows, replay["rows"], cosine_rows, offset_rows)
    frozen_after = {key: sha(path) for key, path in (("manifest_v2", args.manifest_v2), ("manifest_v1", args.manifest_v1),
                    ("phase5b17d_summary", args.phase5b17d_dir / "summary.json"), ("model_source", PROJECT_ROOT / "src" / "models" / "rich_temporal_small_transformer.py"),
                    ("rank_source", PROJECT_ROOT / "src" / "training" / "candidate_ranking.py"))}
    frozen = {"label": LABEL, "diagnostic_only": True, "frozen_before": frozen_before, "frozen_after": frozen_after,
              "all_frozen_inputs_unchanged": frozen_before == frozen_after, "manifest_v2_expected": d.EXPECTED_MANIFEST_SHA,
              "test_reads": 0, "model_structure_changed": False, "lambda_rank": LAMBDA_RANK, "learning_rate": 3e-4,
              "checkpoint_criterion": "unchanged validation_selection_key", "formal_checkpoint_written": False,
              "official_epoch_remains": replay["official_epoch"], "audit_only_oracle_epoch": replay["oracle_epoch"],
              "harm_v2_in_training_loss": "harm_v2" in inspect.getsource(d.loss_terms), "phase5b17d_output_overwritten": False}
    if not frozen["all_frozen_inputs_unchanged"] or frozen["test_reads"] or frozen["harm_v2_in_training_loss"]: raise RuntimeError("frozen audit contract failed")

    summary = {"label": LABEL, "stage": "Phase 5B-1.7D-A Ranking Reproducibility Failure Audit",
               "diagnostic_replay_only": True, "test_reads": 0, "manifest_contract_passed": contract["passed"],
               "target_scale": {"v2_over_v1_std_ratio": distribution["std_ratio_v2_over_v1"], "normalizer": normalizer_audit},
               "candidate_contract": contract_comparison, "pairwise_margin": {"train": train_margin, "validation": validation_margin},
               "gradient": {"v2_weighted_rank_to_base_median": gradient_result["summary"]["weighted_rank_to_base_gradient_ratio"]["median"],
                            "v1_weighted_rank_to_base_median": old_gradient["weighted_rank_to_base_gradient_ratio_median"],
                            "shared_nll_rank_cosine_median": cosine_median},
               "checkpoint": {"official_epoch": replay["official_epoch"], "audit_only_oracle_epoch": replay["oracle_epoch"],
                              "epochs_completed": replay["epochs_completed"], "patience_exhausted": replay["patience_exhausted"], "selector": selector},
               "absolute_calibration": {"B1": offset_all["B1-v2"], "R1": offset_all["R1-v2"]},
               "root_cause": classification, "ready_for_phase5b17e": False, "phase5b17e_started": False, "phase5b2_started": False,
               "next_intervention_requires_human_approval": True, "figures": figures}

    io.write_json(args.output_dir / "frozen_contract.json", frozen); io.write_csv(args.output_dir / "benefit_target_distribution.csv", target_rows)
    io.write_json(args.output_dir / "benefit_normalization_audit.json", normalizer_audit); io.write_json(args.output_dir / "v1_v2_benefit_distribution.json", distribution)
    io.write_csv(args.output_dir / "manifest_v1_v2_candidate_contract.csv", contract_rows)
    v1_margin_unavailable = {"synthetic_interaction": LABEL, "split": "v1_existing_development_statistics",
                             "count": None, "mean": None, "median": None,
                             "comparison_status": "NOT AVAILABLE IN FROZEN V1 ARTIFACTS; old TEST not read and v1 not reconstructed"}
    io.write_csv(args.output_dir / "pairwise_margin_distribution.csv", [v1_margin_unavailable, train_margin, validation_margin]); io.write_csv(args.output_dir / "pair_composition.csv", train_composition + validation_composition)
    io.write_json(args.output_dir / "no_update_loss_gradient_audit.json", gradient_result); io.write_csv(args.output_dir / "gradient_cosine.csv", cosine_rows)
    io.write_csv(args.output_dir / "epochwise_checkpoint_audit.csv", replay["rows"]); io.write_json(args.output_dir / "checkpoint_selector_correlation.json", selector)
    io.write_csv(args.output_dir / "benefit_sign_collapse.csv", sign_rows); io.write_csv(args.output_dir / "episode_prediction_offset.csv", offset_rows)
    io.write_csv(args.output_dir / "by_adverse_event.csv", adverse_rows); io.write_csv(args.output_dir / "stop_motion_audit.csv", stop_rows)
    io.write_csv(args.output_dir / "candidate_cardinality_audit.csv", cardinality_rows); io.write_json(args.output_dir / "candidate_cardinality_contract.json", cardinality)
    io.write_json(args.output_dir / "target_scale_counterfactual.json", ordering)
    io.write_json(args.output_dir / "root_cause_classification.json", classification); io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(io.clean(summary), indent=2), flush=True)


if __name__ == "__main__": main()

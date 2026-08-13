"""Phase 5B-1.7D-B CRACS-v1 checkpoint-selector repair.

This is a synthetic DEVELOPMENT/VALIDATION replay.  TEST is sealed.  The
training data, model, loss, normalizer, optimizer, batch order and seed are
the frozen Phase 5B-1.7D protocol; only validation checkpoint selection is
changed from the historical selector to CRACS-v1.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import audit_phase5b17da_ranking_failure as da
from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b1_static_vs_temporal as b1
from scripts import run_phase5b16_candidate_ranking as b16
from scripts import run_phase5b17d_manifest_v2_rebaseline as d
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.evaluation.context_value_metrics import validation_selection_key
from src.evaluation.cracs_selector import (
    BIAS, EPOCH, MAE, PAIRWISE, SIGN, SPEARMAN, TOP1, TOP2,
    MAX_MAE_RATIO, MAX_SIGN_DROP, SCORE_TIE_TOLERANCE,
    annotate, calibration_limits, ranking_score, select_cracs,
)
from src.multimodal.phase5b_v2_dataset import DEPRECATED_HARM_TARGET, build_v2_temporal_samples
from src.multimodal.temporal_schema import LABEL
from src.training.candidate_ranking import LAMBDA_RANK

STAGE = "Phase 5B-1.7D-B Checkpoint Selection Criterion Repair"
MODEL_NAME = d.MODELS[2]
FORMAL_REPLAY = "FORMAL DIAGNOSTIC REPLAY - VALIDATION ONLY"
HISTORICAL_AUDIT = "HISTORICAL COUNTERFACTUAL ONLY - NOT A FORMAL CHECKPOINT"
TOP_ACCURACY_TOLERANCE = d.TOP_ACCURACY_TOLERANCE


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--epochs", type=int, choices=(30,), default=30)
    parser.add_argument("--patience", type=int, choices=(5,), default=5)
    parser.add_argument("--batch-size", type=int, choices=(64,), default=64)
    parser.add_argument("--learning-rate", type=float, choices=(3e-4,), default=3e-4)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17db_checkpoint_selector_repair")
    parser.add_argument("--phase5b17d-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17d_manifest_v2_rebaseline")
    parser.add_argument("--phase5b17da-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17da_ranking_failure_audit")
    parser.add_argument("--manifest-v2", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17c_adverse_response_expansion" / "phase5b_manifest_v2.json")
    return parser.parse_args()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric_history_row(row: dict) -> dict:
    fields = (EPOCH, MAE, SIGN, SPEARMAN, PAIRWISE, TOP1, TOP2, "mean_gt_best_rank")
    result = {name: int(row[name]) if name == EPOCH else float(row[name]) for name in fields}
    return result


def historical_counterfactual(args, b1_reference: dict) -> tuple[list[dict], dict]:
    """Apply CRACS to frozen epoch metrics without training or checkpointing."""
    source = args.phase5b17da_dir / "epochwise_checkpoint_audit.csv"
    source_rows = read_csv(source)
    metrics = [numeric_history_row(row) for row in source_rows]
    selected, audited = select_cracs(metrics, b1_reference[MAE], b1_reference[SIGN])
    frozen_old = json.loads((args.phase5b17d_dir / "checkpoint_selection.json").read_text(encoding="utf-8"))["models"][MODEL_NAME]["best_epoch"]
    rows = []
    for row in audited:
        clean_row = dict(row)
        clean_row[BIAS] = "UNAVAILABLE_IN_FROZEN_EPOCH_ARTIFACT"
        rows.append({"synthetic_interaction": LABEL, "provenance": HISTORICAL_AUDIT,
                     "source_sha256": sha(source), "old_selector_selected": row[EPOCH] == frozen_old,
                     "cracs_selected": row[EPOCH] == selected[EPOCH], **clean_row})
    summary = {"label": LABEL, "audit_only": True, "source": str(source), "source_sha256": sha(source),
               "old_selector_epoch": frozen_old, "cracs_counterfactual_epoch": selected[EPOCH],
               "cracs_counterfactual_ranking_score": selected["RankingScore"],
               "historical_epoch_cannot_be_promoted": True, "formal_checkpoint_written_from_historical_audit": False,
               "bias_tie_break_available": False, "bias_tie_break_needed": False}
    return rows, summary


def validation_row(epoch, candidate, ranking, historical, prediction, validation, train_rows, model, old_key):
    target = np.asarray([sample.targets.benefit for sample in validation], dtype=np.float64)
    predicted = np.asarray(prediction["benefit"], dtype=np.float64)
    bias = float(np.mean(predicted - target))
    row = {
        "synthetic_interaction": LABEL, "replay": FORMAL_REPLAY, "epoch": epoch,
        **{f"train_{name}": float(np.mean([item[name] for item in train_rows])) for name in train_rows[0]},
        **candidate, **ranking, "global_bias": bias, "absolute_global_bias": abs(bias),
        "positive_prediction_rate": float(np.mean(predicted > 0)),
        "old_selection_key": json.dumps(io.clean(list(old_key))),
        "old_selection_harmful_switch_rate_deprecated": historical["Harmful_Switch_Rate"],
        "parameter_checksum": d.model_sha(model),
    }
    return row


def advance_b1_full(model, train, validation, normalizers, epoch_batches, args, selected_epoch, torch, device):
    """Replay all frozen B1 updates and retain its frozen selected epoch state."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    model.to(device); selected_state = None
    for epoch, batches in enumerate(epoch_batches, 1):
        model.train(); totals = []
        for indices in batches:
            selected = [train[index] for index in indices]
            output = model(b1.temporal_batch(selected, normalizers, torch, device))
            terms = d.loss_terms(output, selected, normalizers, torch, device); loss = terms["base"]
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True); optimizer.step()
            totals.append(float(loss.detach()))
        _, metrics, ranking, _, _ = d.validation_snapshot(d.MODELS[1], model, validation, normalizers, args, torch, device)
        if epoch == selected_epoch:
            selected_state = copy.deepcopy(model.state_dict())
        print(f"{d.MODELS[1]} epoch={epoch:02d} total={np.mean(totals):.5f} val_mae={metrics[MAE]:.5f} rank={ranking['mean_gt_best_rank']:.3f} frozen_selected={epoch == selected_epoch}", flush=True)
    if selected_state is None: raise RuntimeError("frozen B1 selected epoch was not replayed")
    model.load_state_dict(selected_state); model.eval()
    return model


def replay(args, train, validation, normalizers, epoch_batches, b1_reference, torch, device):
    """Advance the exact frozen RNG trajectory and select R1 with CRACS-v1."""
    from src.models.large_context_adapter import SmallContextNetwork
    from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer

    torch.manual_seed(args.seed); b0_model = SmallContextNetwork()
    torch.manual_seed(args.seed); b1_model = RichTemporalSmallTransformer(); r1_model = copy.deepcopy(b1_model)
    initial_checksum = d.model_sha(r1_model)
    frozen_initial = json.loads((args.phase5b17d_dir / "r1_model_audit.json").read_text(encoding="utf-8"))["initial_checksum"]
    replay_args = argparse.Namespace(device=args.device, seed=args.seed, epochs=args.epochs, patience=args.patience,
                                     batch_size=args.batch_size, learning_rate=args.learning_rate)
    # B0 and B1 are advanced first solely to reproduce the frozen dropout RNG sequence.
    b0_model, _, _ = d.train_model(d.MODELS[0], b0_model, train, validation, normalizers, epoch_batches, replay_args, torch, device, False)
    frozen_b1_selection = json.loads((args.phase5b17d_dir / "checkpoint_selection.json").read_text(encoding="utf-8"))["models"][d.MODELS[1]]
    # The frozen B1 completed all 30 epochs and selected epoch 27. Tiny CUDA
    # variation can alter its discontinuous old key, so explicitly replay all
    # 30 frozen updates and retain the preregistered selected epoch. This keeps
    # the exact R1 RNG advancement and the correct frozen B1 subgroup reference.
    b1_model = advance_b1_full(b1_model, train, validation, normalizers, epoch_batches, replay_args,
                               frozen_b1_selection["best_epoch"], torch, device)
    b1_prediction = b1.predict(d.MODELS[1], b1_model, validation, normalizers, args.batch_size, torch, device)

    optimizer = torch.optim.AdamW(r1_model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    r1_model.to(device); rows, states, predictions = [], {}, {}
    old_best = None; cracs_best_epoch = None; cracs_stale = 0; start = time.perf_counter()
    for epoch, batches in enumerate(epoch_batches, 1):
        r1_model.train(); batch_rows = []
        for indices in batches:
            selected = [train[index] for index in indices]
            output = r1_model(b1.temporal_batch(selected, normalizers, torch, device))
            terms = d.loss_terms(output, selected, normalizers, torch, device)
            loss = terms["base"] + terms["weighted_rank"]
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(r1_model.parameters(), 10.0, error_if_nonfinite=True)); optimizer.step()
            if not bool(torch.isfinite(loss)) or not math.isfinite(gradient):
                raise FloatingPointError("non-finite Phase5B-1.7D-B replay training")
            batch_rows.append({name: float(terms[name].detach()) for name in ("nll", "deprecated_old_harm", "base", "rank", "weighted_rank")} | {"total": float(loss.detach()), "ranking_pairs": terms["rank_audit"].pair_count})
        prediction, candidate, ranking, historical, old_key = d.validation_snapshot(MODEL_NAME, r1_model, validation, normalizers, replay_args, torch, device)
        row = validation_row(epoch, candidate, ranking, historical, prediction, validation, batch_rows, r1_model, old_key)
        rows.append(row); states[epoch] = copy.deepcopy(r1_model.state_dict()); predictions[epoch] = prediction
        if old_best is None or old_key < old_best[0]:
            old_best = (old_key, epoch)
        audited = [annotate(item, b1_reference[MAE], b1_reference[SIGN]) for item in rows]
        for original, audit in zip(rows, audited):
            original.update({key: audit[key] for key in ("cracs_eligible", "cracs_ineligibility_reasons", "S_spearman", "S_pairwise", "S_top1", "S_top2", "RankingScore")})
        eligible_rows = [item for item in audited if item["cracs_eligible"]]
        if eligible_rows:
            current_cracs, _ = select_cracs(rows, b1_reference[MAE], b1_reference[SIGN])
            if current_cracs[EPOCH] != cracs_best_epoch:
                cracs_best_epoch = current_cracs[EPOCH]; cracs_stale = 0
            else:
                cracs_stale += 1
        row["cracs_current_best_epoch"] = cracs_best_epoch; row["cracs_stale_epochs"] = cracs_stale
        print(f"{MODEL_NAME} epoch={epoch:02d} total={row['train_total']:.5f} val_mae={row[MAE]:.5f} score={row['RankingScore'] if row['cracs_eligible'] else float('nan'):.5f} cracs_best={cracs_best_epoch} stale={cracs_stale}", flush=True)
        # Before the first eligible epoch there is no valid checkpoint, so the
        # patience clock cannot expire.  Afterwards CRACS alone drives stopping.
        if cracs_best_epoch is not None and cracs_stale >= args.patience:
            break

    selected, audited = select_cracs(rows, b1_reference[MAE], b1_reference[SIGN])
    for original, audit in zip(rows, audited):
        original.update({key: audit[key] for key in ("cracs_eligible", "cracs_ineligibility_reasons", "S_spearman", "S_pairwise", "S_top1", "S_top2", "RankingScore")})
        original["old_selector_final_selected"] = original[EPOCH] == old_best[1]
        original["cracs_final_selected"] = original[EPOCH] == selected[EPOCH]
    return {
        "rows": rows, "states": states, "predictions": predictions,
        "old_epoch": old_best[1], "cracs_epoch": selected[EPOCH], "selected_row": selected,
        "b1_model": b1_model, "b1_prediction": b1_prediction, "b1_selection": frozen_b1_selection,
        "initial_checksum": initial_checksum, "frozen_initial_checksum": frozen_initial,
        "epochs_completed": len(rows), "early_stopped": len(rows) < args.epochs,
        "training_time_s": time.perf_counter() - start,
    }


def metric_view(row):
    fields = (MAE, SIGN, "Benefit_Spearman", SPEARMAN, PAIRWISE, TOP1, TOP2, "mean_gt_best_rank", BIAS, "absolute_global_bias", "positive_prediction_rate", "RankingScore")
    return {name: row.get(name) for name in fields}


def ranking_comparison_rows(left_name, left, right_name, right):
    fields = (MAE, SIGN, SPEARMAN, PAIRWISE, TOP1, TOP2, "mean_gt_best_rank", "RankingScore")
    rows = []
    for field in fields:
        rows.append({"synthetic_interaction": LABEL, "metric": field, left_name: left.get(field), right_name: right.get(field),
                     "R1_minus_reference": None if left.get(field) is None or right.get(field) is None else float(right[field] - left[field])})
    return rows


def systematic_degradation(rows) -> dict:
    groups = sorted({(row["dimension"], row["group"]) for row in rows})
    details = []
    for dimension, group in groups:
        by_model = {row["model"]: row for row in rows if row["dimension"] == dimension and row["group"] == group}
        left, right = by_model["B1-v2"], by_model["R1-v2-CRACS"]
        checks = {
            "spearman_worse": right["mean_feasible_within_episode_spearman"] < left["mean_feasible_within_episode_spearman"],
            "pairwise_worse": right["mean_feasible_pairwise_accuracy"] < left["mean_feasible_pairwise_accuracy"],
            "top1_worse": right["gt_best_top1_accuracy"] < left["gt_best_top1_accuracy"],
            "top2_worse": right["gt_best_top2_recall"] < left["gt_best_top2_recall"],
            "mean_rank_worse": right["mean_gt_best_rank"] > left["mean_gt_best_rank"],
        }
        details.append({"dimension": dimension, "group": group, "all_five_ranking_metrics_worse": all(checks.values()), "checks": checks})
    bad = [row for row in details if row["all_five_ranking_metrics_worse"]]
    return {"definition": "systematic comprehensive degradation means all five reported ranking metrics worsen in the same subgroup",
            "subgroup_count": len(details), "fully_degraded_subgroups": bad, "systematic_comprehensive_degradation": bool(bad), "details": details}


def make_figures(output, rows, old_row, cracs_row, b1_metrics):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    def save(name):
        path = folder / name; plt.suptitle(LABEL, fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    plt.figure(figsize=(9, 4)); eligible = [row for row in rows if row["cracs_eligible"]]
    plt.plot([row[EPOCH] for row in eligible], [row["RankingScore"] for row in eligible], marker="o", label="eligible RankingScore")
    plt.axvline(old_row[EPOCH], color="tab:red", linestyle="--", label="old selector")
    plt.axvline(cracs_row[EPOCH], color="tab:green", linestyle="--", label="CRACS-v1"); plt.xlabel("epoch"); plt.legend(); save("cracs_epoch_selection.png")
    fields = (SPEARMAN, PAIRWISE, TOP1, TOP2); x = np.arange(len(fields)); width = .25
    plt.figure(figsize=(10, 4))
    for offset, (name, item) in enumerate((("B1-v2", b1_metrics), ("old R1", old_row), ("CRACS R1", cracs_row))):
        plt.bar(x + (offset - 1) * width, [item[field] for field in fields], width, label=name)
    plt.xticks(x, ("feasible Spearman", "pairwise", "Top1", "Top2"), rotation=10); plt.legend(); save("ranking_comparison.png")
    plt.figure(figsize=(8, 4)); plt.plot([row[EPOCH] for row in rows], [row[MAE] for row in rows], label="validation MAE")
    plt.axhline(MAX_MAE_RATIO * b1_metrics[MAE], color="tab:red", linestyle="--", label="fixed 1.25x B1 guard")
    plt.xlabel("epoch"); plt.legend(); save("calibration_guard.png")
    return paths


def save_formal_checkpoint(path, replay, normalizer, training_config, selector_config, gates, torch):
    """Save only a formal replay state and only after all three gates pass."""
    if not all(gates[name]["passed"] for name in ("Gate_A", "Gate_B", "Gate_C")):
        return False
    payload = {"label": LABEL, "stage": STAGE, "formal_replay": True, "validation_only": True, "test_reads": 0,
               "model_name": "R1-v2-CRACS", "epoch": replay["cracs_epoch"], "model_state_dict": replay["states"][replay["cracs_epoch"]],
               "manifest_sha256": d.EXPECTED_MANIFEST_SHA, "normalizer": normalizer,
               "training_config": training_config, "selector_config": selector_config}
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, path)
    return True


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite Phase5B-1.7D-B: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    frozen_selection = json.loads((args.phase5b17d_dir / "checkpoint_selection.json").read_text(encoding="utf-8"))
    frozen_b1 = frozen_selection["models"][d.MODELS[1]]
    b1_reference = {**frozen_b1["validation_candidate_metrics"], **frozen_b1["validation_ranking_metrics"]}
    b1_reference[BIAS] = None; b1_reference["positive_prediction_rate"] = None
    b1_reference["RankingScore"] = ranking_score(b1_reference)
    limits = calibration_limits(b1_reference[MAE], b1_reference[SIGN])

    # Required first step: no-training historical counterfactual sanity audit.
    historical_rows, historical_summary = historical_counterfactual(args, b1_reference)
    io.write_csv(args.output_dir / "historical_selector_counterfactual.csv", historical_rows)

    frozen_paths = {
        "manifest_v2": args.manifest_v2, "phase5b17d_summary": args.phase5b17d_dir / "summary.json",
        "phase5b17da_summary": args.phase5b17da_dir / "summary.json",
        "normalizer": args.phase5b17d_dir / "normalizer.json",
        "model_source": PROJECT_ROOT / "src" / "models" / "rich_temporal_small_transformer.py",
        "ranking_loss_source": PROJECT_ROOT / "src" / "training" / "candidate_ranking.py",
    }
    frozen_before = {name: sha(path) for name, path in frozen_paths.items()}
    if frozen_before["manifest_v2"] != d.EXPECTED_MANIFEST_SHA: raise RuntimeError("manifest_v2 checksum mismatch")

    episodes = {"train": build_development_split("train", 240, GENERATOR_SEED, RISK_SEED),
                "validation": build_development_split("validation", 240, GENERATOR_SEED + 1000, RISK_SEED + 1000)}
    splits = {name: build_v2_temporal_samples(values) for name, values in episodes.items()}
    contract = d.manifest_contract(args.manifest_v2, splits["train"] + splits["validation"])
    normalizers = b1.fit_normalizers(splits["train"]); normalizer = d.normalizer_record(normalizers)
    frozen_normalizer = json.loads((args.phase5b17d_dir / "normalizer.json").read_text(encoding="utf-8"))
    if normalizer["sha256"] != frozen_normalizer["sha256"]: raise RuntimeError("normalizer checksum changed")
    batches, batch_audit = b16.make_episode_batches(splits["train"], args.epochs, args.batch_size, args.seed)
    frozen_batch_audit = json.loads((args.phase5b17d_dir / "batch_order_audit.json").read_text(encoding="utf-8"))
    batch_audit_record = {"label": LABEL, "epochs": batch_audit}
    if batch_audit_record != frozen_batch_audit: raise RuntimeError("frozen data order changed")

    replay_result = replay(args, splits["train"], splits["validation"], normalizers, batches, b1_reference, torch, device)
    rows = replay_result["rows"]; old_row = next(row for row in rows if row[EPOCH] == replay_result["old_epoch"])
    cracs_row = next(row for row in rows if row[EPOCH] == replay_result["cracs_epoch"])
    b1_prediction = replay_result["b1_prediction"]; r1_prediction = replay_result["predictions"][replay_result["cracs_epoch"]]
    reconstructed_b1_eval = d.evaluate(d.MODELS[1], replay_result["b1_model"], splits["validation"], normalizers, args, torch, device)[2]
    b1_target = np.asarray([sample.targets.benefit for sample in splits["validation"]], float)
    b1_predicted = np.asarray(b1_prediction["benefit"], float)
    b1_eval = dict(b1_reference)
    b1_eval[BIAS] = float(np.mean(b1_predicted - b1_target)); b1_eval["absolute_global_bias"] = abs(b1_eval[BIAS])
    b1_eval["positive_prediction_rate"] = float(np.mean(b1_predicted > 0)); b1_eval["RankingScore"] = ranking_score(b1_eval)
    b1_eval["reconstructed_epoch27_max_reported_metric_delta"] = max(abs(float(reconstructed_b1_eval[name]) - float(b1_reference[name])) for name in (MAE, SIGN, SPEARMAN, PAIRWISE, TOP1, TOP2, "mean_gt_best_rank"))

    predictions = {"B1-v2": b1_prediction, "R1-v2-CRACS": r1_prediction}
    adverse_rows, context_rows, motion_rows = d.grouped_rows(splits["validation"], predictions)
    subgroup_audit = systematic_degradation(adverse_rows + context_rows + [row for row in motion_rows if row["group"] == "stop"])

    gate_a_checks = {
        "feasible_spearman_improved": cracs_row[SPEARMAN] > old_row[SPEARMAN],
        "pairwise_accuracy_improved": cracs_row[PAIRWISE] > old_row[PAIRWISE],
        "mean_gt_best_rank_improved": cracs_row["mean_gt_best_rank"] < old_row["mean_gt_best_rank"],
        "top1_not_clearly_worse": cracs_row[TOP1] + TOP_ACCURACY_TOLERANCE >= old_row[TOP1],
        "top2_not_clearly_worse": cracs_row[TOP2] + TOP_ACCURACY_TOLERANCE >= old_row[TOP2],
    }
    gate_b_checks = {"mae_within_fixed_guard": cracs_row[MAE] <= limits["max_mae"],
                     "sign_within_fixed_guard": cracs_row[SIGN] >= limits["min_sign_accuracy"],
                     "all_selector_metrics_finite": all(math.isfinite(float(cracs_row[name])) for name in (MAE, SIGN, SPEARMAN, PAIRWISE, TOP1, TOP2, BIAS))}
    # Gate C is preregistered here as a decision-ranking composite comparison,
    # not as seven independent wins: the fixed CRACS score must improve, mean
    # GT-best rank must improve, Top1/Top2 must not clearly worsen, and Gate B
    # must protect calibration. Component deltas remain fully reported.
    gate_c_checks = {"ranking_score_improved": cracs_row["RankingScore"] > b1_eval["RankingScore"],
                     "mean_gt_best_rank_improved": cracs_row["mean_gt_best_rank"] < b1_eval["mean_gt_best_rank"],
                     "top1_not_clearly_worse": cracs_row[TOP1] + TOP_ACCURACY_TOLERANCE >= b1_eval[TOP1],
                     "top2_not_clearly_worse": cracs_row[TOP2] + TOP_ACCURACY_TOLERANCE >= b1_eval[TOP2],
                     "calibration_protected": all(gate_b_checks.values())}
    gates = {"Gate_A": {"name": "Selector Mechanism Repair", "checks": gate_a_checks, "passed": all(gate_a_checks.values())},
             "Gate_B": {"name": "Absolute Calibration Protection", "checks": gate_b_checks, "passed": all(gate_b_checks.values())},
             "Gate_C": {"name": "Temporal/Ranking Reproducibility", "checks": gate_c_checks, "passed": all(gate_c_checks.values()),
                        "interpretation": "composite decision-ranking advantage; individual component deltas are not hidden"}}

    old_curve = read_csv(args.phase5b17d_dir / "r1_training_curve.csv")
    common = min(len(old_curve), len(rows)); comparison_fields = ("train_total", MAE, SIGN, SPEARMAN, PAIRWISE, TOP1, TOP2, "mean_gt_best_rank")
    trajectory_deltas = {field: max(abs(float(rows[index][field]) - float(old_curve[index][field])) for index in range(common)) for field in comparison_fields}
    trajectory_audit = {"common_epochs": common, "old_epochs": len(old_curve), "replay_epochs": len(rows), "max_absolute_delta_by_field": trajectory_deltas,
                        "max_absolute_delta": max(trajectory_deltas.values()), "cuda_nondeterminism_tolerance": 1e-5,
                        "common_trajectory_consistent": max(trajectory_deltas.values()) <= 1e-5,
                        "initial_checksum_identical": replay_result["initial_checksum"] == replay_result["frozen_initial_checksum"],
                        "historical_epoch_parameter_checksums_available": False,
                        "replay_epoch_parameter_checksums_recorded": True}

    selector_definition = {
        "label": LABEL, "name": "Constrained Ranking-Aware Checkpoint Selector", "short_name": "CRACS-v1", "validation_only": True, "test_reads": 0,
        "eligibility": {"reference": "frozen B1-v2 validation", "reference_mae": b1_reference[MAE], "reference_sign_accuracy": b1_reference[SIGN],
                        "MAX_MAE_RATIO": MAX_MAE_RATIO, "MAX_SIGN_DROP": MAX_SIGN_DROP, **limits, "all_metrics_must_be_finite": True},
        "ranking_score": {"S_spearman": "(feasible_within_episode_spearman + 1) / 2", "S_pairwise": PAIRWISE,
                          "S_top1": TOP1, "S_top2": TOP2, "weights": [0.25, 0.25, 0.25, 0.25], "mean_gt_best_rank_in_score": False},
        "tie_break": {"score_tolerance": SCORE_TIE_TOLERANCE, "order": ["lower Benefit_MAE", "lower absolute global bias", "earlier epoch"]},
        "guard_rationale": "The fixed 25% MAE guard prevents a ranking objective from selecting an absolutely uncalibrated checkpoint. It was fixed before replay and must not be tuned from these results.",
        "gate_c_predeclared_rule": "CRACS RankingScore improves over B1, mean GT-best rank improves, Top1/Top2 do not clearly worsen (frozen 0.02 tolerance), and Gate B passes.",
    }
    training_config = {"label": LABEL, "stage": STAGE, "development_validation_only": True, "seed": args.seed, "optimizer": "AdamW",
                       "learning_rate": args.learning_rate, "weight_decay": 1e-3, "batch_size": args.batch_size, "max_epochs": args.epochs,
                       "patience": args.patience, "gradient_clip": 10.0, "lambda_rank": LAMBDA_RANK,
                       "base_loss": "heteroscedastic Gaussian NLL + BCE deprecated old-harm auxiliary",
                       "deprecated_harm_target": DEPRECATED_HARM_TARGET, "deprecated_harm_used_for_safety_conclusion": False,
                       "ranking_loss_definition_changed": False, "data_order_changed": False, "normalizer_changed": False,
                       "model_architecture_changed": False, "only_protocol_change": "validation checkpoint selector OLD -> CRACS-v1", "test_reads": 0}

    selected_metrics = {"label": LABEL, "model": "R1-v2-CRACS", "formal_replay": True, "selected_epoch": replay_result["cracs_epoch"],
                        "old_selector_would_select_epoch": replay_result["old_epoch"], "metrics": metric_view(cracs_row),
                        "calibration_limits": limits, "epochs_completed": replay_result["epochs_completed"], "early_stopped": replay_result["early_stopped"]}
    comparison_rows = ranking_comparison_rows("old_R1", old_row, "R1_CRACS", cracs_row)
    b1_comparison_rows = ranking_comparison_rows("B1_v2", b1_eval, "R1_CRACS", cracs_row)
    calibration_audit = {"label": LABEL, "reference": metric_view(b1_eval), "selected": metric_view(cracs_row), "limits": limits,
                         "mae_ratio_to_B1": cracs_row[MAE] / b1_reference[MAE], "sign_drop_from_B1": b1_reference[SIGN] - cracs_row[SIGN],
                         "global_bias": cracs_row[BIAS], "absolute_global_bias": cracs_row["absolute_global_bias"],
                         "positive_prediction_rate": cracs_row["positive_prediction_rate"], "passed": gates["Gate_B"]["passed"],
                         "target_scale_risk_unmodified": {"v1_std_approx": .1981, "v2_std": normalizers["benefit_raw_std"],
                                                           "ratio_approx": normalizers["benefit_raw_std"] / .1981}}

    frozen_after = {name: sha(path) for name, path in frozen_paths.items()}
    frozen_contract = {"label": LABEL, "stage": STAGE, "frozen_before": frozen_before, "frozen_after": frozen_after,
                       "all_frozen_artifacts_unchanged": frozen_before == frozen_after, "manifest_sha256": frozen_before["manifest_v2"],
                       "expected_manifest_sha256": d.EXPECTED_MANIFEST_SHA, "normalizer_sha256": normalizer["sha256"],
                       "architecture_initial_checksum": replay_result["initial_checksum"], "frozen_initial_checksum": replay_result["frozen_initial_checksum"],
                       "initial_checksum_identical": replay_result["initial_checksum"] == replay_result["frozen_initial_checksum"],
                       "lambda_rank": LAMBDA_RANK, "ranking_loss_source_unchanged": frozen_before["ranking_loss_source"] == frozen_after["ranking_loss_source"],
                       "data_order_identical": batch_audit_record == frozen_batch_audit, "test_reads": 0, "phase5b17d_output_overwritten": False,
                       "phase5b17da_output_overwritten": False, "training_trajectory": trajectory_audit}
    if not (contract["passed"] and frozen_contract["all_frozen_artifacts_unchanged"] and frozen_contract["initial_checksum_identical"] and frozen_contract["data_order_identical"] and frozen_contract["test_reads"] == 0):
        raise RuntimeError("frozen CRACS replay contract failed")

    figures = make_figures(args.output_dir, rows, old_row, cracs_row, b1_eval)
    all_gates_pass = all(gate["passed"] for gate in gates.values())
    checkpoint_path = args.output_dir / "checkpoints" / "r1_v2_cracs_best.pt"
    checkpoint_written = save_formal_checkpoint(checkpoint_path, replay_result, normalizer, training_config, selector_definition, gates, torch)
    if checkpoint_written != all_gates_pass: raise RuntimeError("checkpoint freeze gate mismatch")
    if checkpoint_written:
        io.write_json(args.output_dir / "normalizer.json", normalizer)

    gates["all_passed"] = all_gates_pass; gates["checkpoint_frozen"] = checkpoint_written
    summary = {"label": LABEL, "stage": STAGE, "development_validation_experiment": True, "test_reads": 0,
               "historical_counterfactual": historical_summary,
               "formal_replay": {"old_selector_epoch": replay_result["old_epoch"], "cracs_epoch": replay_result["cracs_epoch"],
                                 "same_training_trajectory": trajectory_audit["common_trajectory_consistent"], "epochs_completed": replay_result["epochs_completed"],
                                 "selected": metric_view(cracs_row), "old": metric_view(old_row), "B1": metric_view(b1_eval)},
               "subgroups": subgroup_audit, "gates": gates, "R1_v2_CRACS_frozen": checkpoint_written,
               "representation_benefit_ranking_reliably_freezable": checkpoint_written,
               "formal_safety_decision_conclusion_allowed": False,
               "ready_for_phase5b17e": all_gates_pass, "phase5b17e_started": False, "phase5b2_started": False,
               "figures": figures}

    io.write_json(args.output_dir / "frozen_contract.json", frozen_contract)
    io.write_json(args.output_dir / "selector_definition.json", selector_definition)
    io.write_json(args.output_dir / "training_config.json", training_config)
    io.write_csv(args.output_dir / "replay_training_curve.csv", rows)
    io.write_csv(args.output_dir / "old_vs_cracs_epoch_selection.csv", comparison_rows)
    io.write_json(args.output_dir / "selected_checkpoint_metrics.json", selected_metrics)
    io.write_csv(args.output_dir / "b1_vs_r1_cracs.csv", b1_comparison_rows)
    io.write_csv(args.output_dir / "by_adverse_event.csv", adverse_rows)
    io.write_csv(args.output_dir / "by_context.csv", context_rows)
    io.write_csv(args.output_dir / "by_motion.csv", motion_rows)
    io.write_json(args.output_dir / "calibration_audit.json", calibration_audit)
    io.write_json(args.output_dir / "gate_results.json", gates)
    io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(io.clean(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()

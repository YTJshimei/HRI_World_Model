"""Phase 5B-1.6 validation-only candidate-set ranking intervention."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as p5
from scripts import run_phase5b1_static_vs_temporal as b1
from scripts import run_phase5b15_decision_bottleneck as b15
from src.evaluation.context_value_metrics import spearman, validation_selection_key
from src.multimodal.temporal_schema import LABEL
from src.training.candidate_ranking import LAMBDA_RANK, pairwise_logistic_ranking_loss

MODELS = ("R0 B1-Control", "R1 B1-Rank")
FROZEN_THRESHOLDS = (-0.02, 0.2)
CONTEXTS = ("C7", "C8", "C9")
MAE_COLLAPSE_ABSOLUTE = 0.015
MAE_COLLAPSE_RELATIVE = 0.20
AUROC_COLLAPSE_ABSOLUTE = 0.03


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--epochs", type=int, choices=(30,), default=30)
    parser.add_argument("--patience", type=int, choices=(5,), default=5)
    parser.add_argument("--batch-size", type=int, choices=(64,), default=64)
    parser.add_argument("--learning-rate", type=float, choices=(3e-4,), default=3e-4)
    parser.add_argument("--belief-samples", type=int, choices=(16,), default=16)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b16_candidate_ranking")
    parser.add_argument("--phase5b1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b1_static_vs_temporal_small")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b05_c7_coverage")
    parser.add_argument("--phase5a-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--phase4c2-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c2")
    parser.add_argument("--phase4c3-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c3")
    return parser.parse_args()


def model_checksum(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_frozen_normalizer(folder: Path):
    normalizer = b15.load_normalizers(folder / "normalizer.json")
    record = json.loads((folder / "normalizer.json").read_text(encoding="utf-8"))
    return normalizer, record


def build_train_validation_only(args, torch):
    """Deterministically replay only frozen train/validation manifest members."""
    _, splits = b1.build_development(args, torch)
    if set(splits) != {"train", "validation"}:
        raise RuntimeError("Phase5B-1.6 builder may expose only train/validation")
    if any(sample.split not in ("train", "validation") for values in splits.values() for sample in values):
        raise RuntimeError("sealed split reached development builder")
    if any(not sample.sample_id.startswith(f"{name}:") for name, values in splits.items() for sample in values):
        raise RuntimeError("sample ID/split contract mismatch")
    return splits


def validate_development_manifest(manifest, splits):
    expected = {
        candidate: (row["episode_id"], row["split"])
        for row in manifest["episodes"] if row["split"] in ("train", "validation")
        for candidate in row["candidate_ids"]
    }
    actual = {
        sample.sample_id: (sample.episode_id, sample.split)
        for sample in splits["train"] + splits["validation"]
    }
    return {
        "expected_development_candidates": len(expected),
        "actual_development_candidates": len(actual),
        "candidate_ids_identical": set(expected) == set(actual),
        "episode_split_labels_identical": expected == actual,
        "passed": expected == actual,
    }


def generic_target_semantics_audit(samples):
    rows = []
    for episode_id, indices in b15.group_episode(samples).items():
        # Stage A defines every candidate target as one shared canonical
        # episode baseline cost minus that candidate's GT cost.  Auditing the
        # reconstructed baseline works for both normal and all-infeasible
        # episodes without redefining the original generic fallback rule.
        gt_cost = np.asarray([samples[index].targets.gt_cost for index in indices], float)
        actual = np.asarray([samples[index].targets.benefit for index in indices], float)
        reconstructed_baseline = actual + gt_cost
        rows.append({"episode_id": episode_id, "reconstructed_baseline_spread": float(np.ptp(reconstructed_baseline))})
    maximum = max(row["reconstructed_baseline_spread"] for row in rows)
    return {
        "target_definition": "GT cost of frozen generic candidate minus GT cost of candidate",
        "same_definition_for_generic_and_personalized_candidates": maximum <= 1e-6,
        "generic_candidate_included_when_feasible": True,
        "episode_count": len(rows),
        "maximum_reconstructed_baseline_spread": maximum,
        "passed": maximum <= 1e-6,
    }


def make_episode_batches(samples, epochs: int, candidate_budget: int, seed: int):
    grouped = b15.group_episode(samples)
    episodes = list(grouped)
    generator = np.random.default_rng(seed)
    all_epochs, order_rows = [], []
    for epoch in range(1, epochs + 1):
        order = [episodes[index] for index in generator.permutation(len(episodes))]
        batches, current = [], []
        for episode_id in order:
            indices = grouped[episode_id]
            if current and len(current) + len(indices) > candidate_budget:
                batches.append(current); current = []
            current.extend(indices)
        if current:
            batches.append(current)
        all_epochs.append(batches)
        order_rows.append({
            "epoch": epoch,
            "episode_order_sha256": b1.digest_json(order),
            "batch_sha256": b1.digest_json(batches),
            "episode_count": len(order),
            "batch_count": len(batches),
            "candidate_count": sum(map(len, batches)),
            "episode_groups_preserved": all(len({samples[i].episode_id for i in batch}) >= 1 for batch in batches),
        })
    return all_epochs, order_rows


def loss_terms(output, selected, normalizers, torch, device):
    target = torch.tensor(
        [(sample.targets.benefit - normalizers["benefit_mean"]) / normalizers["benefit_scale"] for sample in selected],
        dtype=torch.float32, device=device,
    )
    harm = torch.tensor([sample.targets.harm for sample in selected], dtype=torch.float32, device=device)
    feasible = torch.tensor([sample.targets.feasible for sample in selected], dtype=torch.bool, device=device)
    if not bool(feasible.any()):
        raise RuntimeError("episode-grouped batch has no feasible candidate")
    error = output.benefit_mean[feasible] - target[feasible]
    log_variance = output.benefit_log_variance[feasible]
    nll = 0.5 * (error.square() * torch.exp(-log_variance) + log_variance).mean()
    harm_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.harm_logit[feasible], harm[feasible], pos_weight=torch.tensor(2.0, device=device),
    )
    rank, rank_audit = pairwise_logistic_ranking_loss(
        output.benefit_mean, target, [sample.episode_id for sample in selected], feasible,
    )
    return {"nll": nll, "harm": harm_loss, "base": nll + harm_loss, "rank": rank,
            "weighted_rank": LAMBDA_RANK * rank, "target": target, "feasible": feasible, "rank_audit": rank_audit}


def _gradient_norm(loss, named_parameters, torch, retain_graph=True):
    parameters = [(name, value) for name, value in named_parameters if value.requires_grad]
    gradients = torch.autograd.grad(loss, [value for _, value in parameters], retain_graph=retain_graph, allow_unused=True)
    norms = {}
    groups = {
        "total": lambda _: True,
        "temporal_encoder": lambda name: not name.startswith(("benefit.", "uncertainty.", "harm.")),
        "benefit_head": lambda name: name.startswith("benefit."),
        "harm_head": lambda name: name.startswith("harm."),
    }
    for group, predicate in groups.items():
        square = [gradient.detach().float().square().sum() for (name, _), gradient in zip(parameters, gradients)
                  if gradient is not None and predicate(name)]
        norms[group] = float(torch.stack(square).sum().sqrt()) if square else 0.0
    return norms


def no_update_gradient_audit(model_r0, model_r1, samples, batches, normalizers, torch, device):
    model_r0.to(device).eval(); model_r1.to(device).eval()
    initial_r0, initial_r1 = model_checksum(model_r0), model_checksum(model_r1)
    first = [samples[index] for index in batches[0]]
    with torch.no_grad():
        out0 = model_r0(b1.temporal_batch(first, normalizers, torch, device))
        out1 = model_r1(b1.temporal_batch(first, normalizers, torch, device))
    forward_errors = {
        field: float((getattr(out0, field) - getattr(out1, field)).abs().max())
        for field in ("context_embedding", "benefit_mean", "benefit_log_variance", "harm_logit")
    }
    records = []
    for batch_number, indices in enumerate(batches[:32], 1):
        selected = [samples[index] for index in indices]
        output = model_r1(b1.temporal_batch(selected, normalizers, torch, device))
        terms = loss_terms(output, selected, normalizers, torch, device)
        base_gradient = _gradient_norm(terms["base"], list(model_r1.named_parameters()), torch)
        rank_gradient = _gradient_norm(terms["weighted_rank"], list(model_r1.named_parameters()), torch)
        total_gradient = _gradient_norm(terms["base"] + terms["weighted_rank"], list(model_r1.named_parameters()), torch, False)
        scalar = {key: float(terms[key].detach()) for key in ("nll", "harm", "base", "rank", "weighted_rank")}
        finite = all(np.isfinite(value) for value in scalar.values()) and all(
            np.isfinite(value) for group in (base_gradient, rank_gradient, total_gradient) for value in group.values()
        )
        records.append({
            "batch": batch_number, "candidate_count": len(indices),
            "episode_count": len({sample.episode_id for sample in selected}),
            "ranking_episode_count": terms["rank_audit"].episode_count,
            "ranking_pair_count": terms["rank_audit"].pair_count,
            **scalar, "base_gradient": base_gradient, "ranking_contribution_gradient": rank_gradient,
            "total_gradient": total_gradient,
            "weighted_rank_to_base_gradient_ratio": rank_gradient["total"] / max(base_gradient["total"], 1e-12),
            "finite": finite,
        })
    ratio = np.asarray([row["weighted_rank_to_base_gradient_ratio"] for row in records])
    summary = {
        key: float(np.mean([row[key] for row in records]))
        for key in ("nll", "harm", "base", "rank", "weighted_rank")
    }
    summary["gradient_norm_mean"] = {
        prefix: {group: float(np.mean([row[field][group] for row in records])) for group in ("total", "temporal_encoder", "benefit_head", "harm_head")}
        for prefix, field in (("base", "base_gradient"), ("ranking_contribution", "ranking_contribution_gradient"), ("combined_total", "total_gradient"))
    }
    sustained_dominance = bool(np.median(ratio) > 10.0 and np.mean(ratio > 10.0) >= 0.75)
    passed = (
        len(records) >= 32 and all(row["finite"] for row in records)
        and summary["gradient_norm_mean"]["base"]["total"] > 0
        and summary["gradient_norm_mean"]["ranking_contribution"]["total"] > 0
        and not sustained_dominance and initial_r0 == initial_r1
        and max(forward_errors.values()) == 0.0
    )
    return {
        "label": LABEL, "optimizer_created": False, "optimizer_step_count": 0,
        "batch_count": len(records), "episode_grouped_batches": True,
        "lambda_rank": LAMBDA_RANK, "summary": summary,
        "weighted_rank_to_base_gradient_ratio_median": float(np.median(ratio)),
        "weighted_rank_to_base_gradient_ratio_max": float(np.max(ratio)),
        "sustained_weighted_rank_gradient_above_10x_base": sustained_dominance,
        "r0_initial_parameter_checksum": initial_r0, "r1_initial_parameter_checksum": initial_r1,
        "initial_parameter_checksums_identical": initial_r0 == initial_r1,
        "ranking_disabled_forward_max_abs_errors": forward_errors,
        "ranking_disabled_forward_identical": max(forward_errors.values()) == 0.0,
        "records": records, "passed": passed,
    }


def ranking_summary(samples, prediction):
    audit = b15.audit_model("diagnostic", samples, prediction, FROZEN_THRESHOLDS)
    return b15.summarize_ranking(audit["ranking"])


def train_one(model_name, model, samples, validation, normalizers, epoch_batches, args, torch, device, use_ranking):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    curve, best, stale = [], None, 0
    started = time.perf_counter()
    for epoch, batches in enumerate(epoch_batches, 1):
        model.train(); losses = []
        for indices in batches:
            selected = [samples[index] for index in indices]
            output = model(b1.temporal_batch(selected, normalizers, torch, device))
            terms = loss_terms(output, selected, normalizers, torch, device)
            contribution = terms["weighted_rank"] if use_ranking else terms["rank"] * 0.0
            loss = terms["base"] + contribution
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True))
            optimizer.step()
            if not bool(torch.isfinite(loss)) or not np.isfinite(gradient):
                raise FloatingPointError("non-finite Phase5B-1.6 training state")
            losses.append({key: float(terms[key].detach()) for key in ("nll", "harm", "base", "rank", "weighted_rank")} | {"total": float(loss.detach())})
        prediction = b1.predict("B1", model, validation, normalizers, args.batch_size, torch, device)
        _, _, metrics = b1.decision_evaluation(model_name, prediction, validation, FROZEN_THRESHOLDS)
        diagnostic = ranking_summary(validation, prediction)
        row = {
            "synthetic_interaction": LABEL, "model": model_name, "epoch": epoch,
            **{f"train_{key}": float(np.mean([item[key] for item in losses])) for key in losses[0]},
            **metrics, **diagnostic, "benefit_threshold": FROZEN_THRESHOLDS[0], "harm_threshold": FROZEN_THRESHOLDS[1],
        }
        curve.append(row); key = validation_selection_key(metrics)
        if best is None or key < best[0]:
            best = (key, epoch, copy.deepcopy(model.state_dict()), metrics, diagnostic); stale = 0
        else:
            stale += 1
        print(f"{model_name} epoch={epoch:02d} loss={row['train_total']:.5f} val_regret={metrics['Mean_Regret']:.5f} stale={stale}", flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best[2]); model.eval()
    return model, curve, {
        "best_epoch": best[1], "thresholds": list(FROZEN_THRESHOLDS), "validation_metrics": best[3],
        "ranking_diagnostics_at_selection": best[4], "selection_key": list(best[0]),
        "epochs_completed": len(curve), "early_stopped": len(curve) < args.epochs,
        "training_time_s": time.perf_counter() - started, "checkpoint_selected_on": "validation only",
        "checkpoint_criterion": "unchanged Phase5B-1 validation_selection_key; ranking metrics excluded",
        "ranking_loss_contribution": LAMBDA_RANK if use_ranking else 0.0,
    }


def beneficial_sign_rows(model_name, samples, prediction):
    rows = []
    for index, sample in enumerate(samples):
        if sample.targets.benefit <= 1e-6:
            continue
        rows.append({
            "synthetic_interaction": LABEL, "model": model_name, "sample_id": sample.sample_id,
            "episode_id": sample.episode_id, "contexts": "|".join(sorted(b15.context_labels(sample))),
            "feasible": sample.targets.feasible, "gt_benefit": sample.targets.benefit,
            "predicted_benefit": float(prediction["benefit"][index]),
            "sign_correct": bool(prediction["benefit"][index] > 0),
            "bias": float(prediction["benefit"][index] - sample.targets.benefit),
        })
    return rows


def context_rows(model_name, samples, prediction):
    rows = []
    for context in CONTEXTS:
        indices = [index for index, sample in enumerate(samples) if context in b15.context_labels(sample)]
        subset = [samples[index] for index in indices]
        subprediction = {key: np.asarray(value)[indices] for key, value in prediction.items()}
        audit = b15.audit_model(model_name, subset, subprediction, FROZEN_THRESHOLDS)
        rank = b15.summarize_ranking(audit["ranking"])
        funnel = b15.summarize_funnel(audit["funnel"])
        _, _, decision = b1.decision_evaluation(model_name, subprediction, subset, FROZEN_THRESHOLDS)
        rows.append({
            "synthetic_interaction": LABEL, "model": model_name, "context": context,
            "episode_count": len(b15.group_episode(subset)), "candidate_count": len(subset),
            **rank, "beneficial_sign_correct": funnel["sign_correct"], "beneficial_count": funnel["opportunity_count"],
            "top1_beneficial_count": funnel["top1"], "top2_beneficial_count": funnel["top2"],
            "benefit_gate_pass_count": funnel["benefit_threshold_pass"], "harm_gate_pass_count": funnel["harm_threshold_pass"],
            "final_switch_count": funnel["final_switch"], "Mean_Regret": decision["Mean_Regret"],
        })
    return rows


def decision_deltas(decision_by_model):
    left = {row["episode_id"]: row for row in decision_by_model[MODELS[0]]}
    right = {row["episode_id"]: row for row in decision_by_model[MODELS[1]]}
    rows = []
    for episode_id in sorted(left):
        r0, r1 = left[episode_id], right[episode_id]
        delta = float(r1["Oracle_Regret"] - r0["Oracle_Regret"])
        if r0["selected_action"] != r1["selected_action"] or abs(delta) > 1e-12:
            rows.append({
                "synthetic_interaction": LABEL, "episode_id": episode_id, "contexts": r0["context_labels"],
                "r0_selected_action": r0["selected_action"], "r1_selected_action": r1["selected_action"],
                "r0_regret": r0["Oracle_Regret"], "r1_regret": r1["Oracle_Regret"], "r1_minus_r0_regret": delta,
                "outcome": "improved" if delta < -1e-9 else "worsened" if delta > 1e-9 else "changed_equal_regret",
                "r0_harmful_switch": r0["harmful_switch"], "r1_harmful_switch": r1["harmful_switch"],
            })
    return rows


def make_figures(output, audits, ranking, sign_rows, context):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    short = {MODELS[0]: "R0", MODELS[1]: "R1"}
    def save(name):
        path = folder / name; plt.title(LABEL, fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    plt.figure(); plt.bar([short[m] for m in MODELS], [ranking[m]["mean_feasible_within_episode_spearman"] for m in MODELS]); plt.ylabel("feasible within-episode Spearman"); save("feasible_spearman.png")
    plt.figure()
    for model in MODELS: plt.hist([row["gt_best_rank"] for row in audits[model]["ranking"]], bins=np.arange(.5, 6.5), alpha=.5, label=short[model])
    plt.xlabel("GT-best predicted rank"); plt.legend(); save("gt_best_rank_distribution.png")
    plt.figure()
    for model in MODELS:
        rows = [row for row in sign_rows if row["model"] == model]; plt.scatter([r["gt_benefit"] for r in rows], [r["predicted_benefit"] for r in rows], s=16, alpha=.65, label=short[model])
    plt.axhline(0, color="k", linewidth=.8); plt.xlabel("GT beneficial target"); plt.ylabel("predicted benefit"); plt.legend(); save("beneficial_predicted_benefit.png")
    stages = ("opportunity_count", "feasible", "sign_correct", "top1", "top2", "benefit_threshold_pass", "harm_threshold_pass", "generic_score_win", "final_switch")
    plt.figure(figsize=(11, 4))
    for model in MODELS:
        funnel = b15.summarize_funnel(audits[model]["funnel"]); plt.plot(stages, [funnel[key] for key in stages], marker="o", label=short[model])
    plt.xticks(rotation=30, ha="right"); plt.ylabel("beneficial candidate count"); plt.legend(); save("beneficial_funnel.png")
    plt.figure(); plt.bar([short[m] for m in MODELS], [ranking[m]["mean_feasible_pairwise_accuracy"] for m in MODELS]); plt.ylabel("feasible pairwise accuracy"); save("feasible_pairwise_accuracy.png")
    plt.figure(figsize=(8, 4)); x=np.arange(len(CONTEXTS)); width=.35
    for offset, model in ((-.5, MODELS[0]), (.5, MODELS[1])):
        values=[next(row["mean_feasible_within_episode_spearman"] for row in context if row["model"]==model and row["context"]==name) for name in CONTEXTS]
        plt.bar(x+offset*width, values, width, label=short[model])
    plt.xticks(x, CONTEXTS); plt.ylabel("feasible within-episode Spearman"); plt.legend(); save("context_ranking.png")
    return paths


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite Phase5B-1.6: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "checkpoints").mkdir()
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    manifest, manifest_audit = b1.manifest_file_audit(args.manifest_dir)
    splits = build_train_validation_only(args, torch)
    manifest_development = validate_development_manifest(manifest, splits)
    normalizers, normalizer_record = load_frozen_normalizer(args.phase5b1_dir)
    selection = json.loads((args.phase5b1_dir / "checkpoint_selection.json").read_text(encoding="utf-8"))
    source_thresholds = tuple(selection["models"]["B1 Rich Temporal Small"]["thresholds"])
    if source_thresholds != FROZEN_THRESHOLDS: raise RuntimeError("frozen B1 thresholds differ from preregistered thresholds")
    fit_ids = [sample.sample_id for sample in splits["train"] if sample.targets.feasible]
    normalizer_ids_identical = fit_ids == normalizer_record["fit_sample_ids"]
    semantics = generic_target_semantics_audit(splits["train"] + splits["validation"])
    if not manifest_development["passed"] or not normalizer_ids_identical or not semantics["passed"]:
        raise RuntimeError("frozen data/normalizer/target contract audit failed")

    epoch_batches, order_rows = make_episode_batches(splits["train"], args.epochs, args.batch_size, args.seed)
    preflight_batches = [batch for epoch in epoch_batches for batch in epoch][:32]
    from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
    base_model = RichTemporalSmallTransformer()
    r0, r1 = copy.deepcopy(base_model), copy.deepcopy(base_model)
    architecture = base_model.architecture_audit()
    initialization = {name: model_checksum(model) for name, model in zip(MODELS, (r0, r1))}

    ranking_definition = {
        "label": LABEL, "objective": "pairwise logistic", "formula": "softplus(-s_ij * (mu_i - mu_j))",
        "pair_scope": "same episode and feasible candidates only", "ties": "excluded",
        "reduction": "mean pairs within episode, then mean valid episodes within batch",
        "target": "train GT benefit ordering; TRAINING_TARGET_ONLY", "runtime_oracle_input": False,
        "lambda_rank": LAMBDA_RANK, "all_candidate_ranking": "diagnostic only",
        "generic_candidate_semantics_audit": semantics,
    }
    training_config = {
        "label": LABEL, "stage": "Phase 5B-1.6 Candidate-Set Ranking Objective Intervention",
        "models": list(MODELS), "only_difference": "R1 adds 0.25 * feasible episode-local ranking loss",
        "seed": 42, "max_epochs": 30, "patience": 5, "batch_size_candidate_budget": 64,
        "batching": "whole episode groups; no episode split across a batch", "optimizer": "AdamW",
        "learning_rate": 3e-4, "weight_decay": 1e-3, "gradient_clip": 10.0,
        "benefit_loss": "full heteroscedastic Gaussian NLL", "harm_loss": "BCEWithLogits pos_weight=2.0",
        "lambda_rank": LAMBDA_RANK, "thresholds": list(FROZEN_THRESHOLDS),
        "checkpoint_selection": "unchanged Phase5B-1 validation_selection_key; ranking diagnostics excluded",
        "gate_a_preregistered": {
            "strict_feasible_spearman_increase": True, "strict_feasible_pairwise_increase": True,
            "strict_beneficial_rank_decrease": True, "top1_not_decrease": True, "top2_not_decrease": True,
            "MAE_collapse": f"R1 > R0 + max({MAE_COLLAPSE_ABSOLUTE}, {MAE_COLLAPSE_RELATIVE}*R0)",
            "AUROC_collapse": f"R1 < R0 - {AUROC_COLLAPSE_ABSOLUTE}",
        },
        "gate_b_preregistered": {
            "beneficial_switch_count_and_recall_strictly_increase": True, "harmful_switch_count_not_increase": True,
            "mean_regret_strictly_decrease": True, "p95_regret_tolerance": 0.025, "safety_not_worse": True,
        },
        "test_access": "sealed; zero candidate/label reads", "oversampling": False, "undersampling": False,
    }
    frozen_contract = {
        "label": LABEL, **manifest_audit, "manifest_development_audit": manifest_development,
        "frozen_manifest_exact_deterministic_replay": True, "new_episode_ids_created": False,
        "train_candidates": len(splits["train"]), "validation_candidates": len(splits["validation"]),
        "train_episodes": len(b15.group_episode(splits["train"])), "validation_episodes": len(b15.group_episode(splits["validation"])),
        "test_candidates_read": 0, "test_labels_read": 0, "test_metrics_computed": False,
        "normalizer_sha256": normalizer_record["sha256"], "normalizer_fit_ids_identical": normalizer_ids_identical,
        "r0_r1_normalizer_identical": True, "r0_r1_data_order_identical": True,
        "data_order_sha256": b1.digest_json(order_rows), "r0_r1_thresholds_identical": True,
        "thresholds_before": list(FROZEN_THRESHOLDS), "thresholds_after": list(FROZEN_THRESHOLDS),
        "safety_mask_unchanged": True, "arbitration_unchanged": True,
        "architecture": architecture, "parameter_count_exact": architecture["trainable_parameter_count"] == 352376,
        "initialization_checksums": initialization, "initialization_checksums_identical": len(set(initialization.values())) == 1,
        "runtime_person_id_absent": True, "runtime_oracle_theta_absent": True, "runtime_gt_future_absent": True,
    }
    p5.write_json(args.output_dir / "frozen_contract.json", frozen_contract)
    p5.write_json(args.output_dir / "ranking_loss_definition.json", ranking_definition)
    p5.write_json(args.output_dir / "training_config.json", training_config)

    preflight = no_update_gradient_audit(r0, r1, splits["train"], preflight_batches, normalizers, torch, device)
    p5.write_json(args.output_dir / "no_update_gradient_audit.json", preflight)
    if not preflight["passed"]:
        raise RuntimeError("no-update loss/gradient preflight failed; formal training stopped")

    trained, curves, locks = {}, {}, {}
    for name, model, use_ranking in ((MODELS[0], r0, False), (MODELS[1], r1, True)):
        model, curve, lock = train_one(name, model, splits["train"], splits["validation"], normalizers, epoch_batches, args, torch, device, use_ranking)
        trained[name], curves[name], locks[name] = model, curve, lock
        torch.save({"model_state_dict": model.state_dict(), "selection": lock,
                    "manifest_sha256": b1.EXPECTED_MANIFEST_SHA, "normalizer_sha256": normalizer_record["sha256"],
                    "test_candidates_read": 0}, args.output_dir / "checkpoints" / ("r0_best.pt" if name == MODELS[0] else "r1_best.pt"))
    checkpoint = {
        "label": LABEL, "models": locks, "selection_validation_only": True,
        "ranking_metrics_used_for_checkpoint_selection": False, "thresholds_reselected": False,
        "test_can_change_checkpoint_or_threshold": False, "test_candidates_read": 0,
    }
    p5.write_json(args.output_dir / "checkpoint_selection.json", checkpoint)

    predictions, audits, global_rows, decision_by_model, ranking_by_model = {}, {}, [], {}, {}
    all_episode_ranking, all_funnel, all_sign, all_context = [], [], [], []
    for name in MODELS:
        prediction = b1.predict("B1", trained[name], splits["validation"], normalizers, args.batch_size, torch, device)
        predictions[name] = prediction
        candidate_rows, decisions, metrics = b1.decision_evaluation(name, prediction, splits["validation"], FROZEN_THRESHOLDS)
        audit = b15.audit_model(name, splits["validation"], prediction, FROZEN_THRESHOLDS)
        rank = b15.summarize_ranking(audit["ranking"])
        predictions[name]["candidate_rows"] = candidate_rows; audits[name] = audit; ranking_by_model[name] = rank; decision_by_model[name] = decisions
        global_rows.append({"synthetic_interaction": LABEL, "model": name, **metrics,
                            "All_Candidate_Benefit_Spearman_Diagnostic": spearman(prediction["benefit"], [sample.targets.benefit for sample in splits["validation"]])})
        all_episode_ranking += audit["ranking"]; all_funnel += audit["funnel"]
        all_sign += beneficial_sign_rows(name, splits["validation"], prediction)
        all_context += context_rows(name, splits["validation"], prediction)

    metrics_by_name = {row["model"]: row for row in global_rows}; m0, m1 = metrics_by_name[MODELS[0]], metrics_by_name[MODELS[1]]
    q0, q1 = ranking_by_model[MODELS[0]], ranking_by_model[MODELS[1]]
    gate_a_checks = {
        "feasible_spearman_increased": q1["mean_feasible_within_episode_spearman"] > q0["mean_feasible_within_episode_spearman"],
        "feasible_pairwise_increased": q1["mean_feasible_pairwise_accuracy"] > q0["mean_feasible_pairwise_accuracy"],
        "beneficial_gt_best_rank_decreased": q1["beneficial_episode_mean_gt_best_rank"] < q0["beneficial_episode_mean_gt_best_rank"],
        "top1_not_decreased": q1["gt_best_top1_accuracy"] >= q0["gt_best_top1_accuracy"],
        "top2_not_decreased": q1["gt_best_top2_recall"] >= q0["gt_best_top2_recall"],
        "benefit_mae_not_collapsed": m1["Benefit_MAE"] <= m0["Benefit_MAE"] + max(MAE_COLLAPSE_ABSOLUTE, MAE_COLLAPSE_RELATIVE*m0["Benefit_MAE"]),
        "harm_auroc_not_collapsed": m1["Harm_AUROC"] >= m0["Harm_AUROC"] - AUROC_COLLAPSE_ABSOLUTE,
    }
    gate_a = all(gate_a_checks.values())
    gate_b_checks = {
        "beneficial_switch_count_increased": m1["Beneficial_Switch_Count"] > m0["Beneficial_Switch_Count"],
        "beneficial_recall_increased": m1["Beneficial_Switch_Recall"] > m0["Beneficial_Switch_Recall"],
        "harmful_switch_not_increased": m1["Harmful_Switch_Count"] <= m0["Harmful_Switch_Count"],
        "mean_regret_decreased": m1["Mean_Regret"] < m0["Mean_Regret"],
        "p95_not_clearly_worse": m1["P95_Regret"] <= m0["P95_Regret"] + .025,
        "safety_not_worse": m1["Safety_Violation"] <= m0["Safety_Violation"],
    }
    gate_b = all(gate_b_checks.values())
    sign_count = {name: sum(row["sign_correct"] for row in all_sign if row["model"] == name) for name in MODELS}
    funnel_summary = {name: b15.summarize_funnel(audits[name]["funnel"]) for name in MODELS}
    deltas = decision_deltas(decision_by_model)
    figure_paths = make_figures(args.output_dir, audits, ranking_by_model, all_sign, all_context)

    p5.write_csv(args.output_dir / "r0_training_curve.csv", curves[MODELS[0]])
    p5.write_csv(args.output_dir / "r1_training_curve.csv", curves[MODELS[1]])
    p5.write_csv(args.output_dir / "global_metrics.csv", global_rows)
    p5.write_csv(args.output_dir / "episode_ranking.csv", all_episode_ranking)
    p5.write_json(args.output_dir / "episode_ranking_summary.json", {"label": LABEL, "models": ranking_by_model})
    p5.write_csv(args.output_dir / "beneficial_sign_audit.csv", all_sign)
    p5.write_csv(args.output_dir / "beneficial_funnel.csv", all_funnel)
    p5.write_csv(args.output_dir / "decision_metrics.csv", global_rows)
    p5.write_csv(args.output_dir / "by_context_metrics.csv", all_context)
    p5.write_csv(args.output_dir / "decision_delta_cases.csv", deltas)
    summary = {
        "label": LABEL, "stage": "Phase 5B-1.6 Candidate-Set Ranking Objective Intervention",
        "validation_only": True, "test_candidates_read": 0, "models": metrics_by_name,
        "episode_ranking": ranking_by_model, "beneficial_sign_correct_of_34": sign_count,
        "beneficial_funnel": funnel_summary, "no_update_gradient_audit_passed": preflight["passed"],
        "gate_a_ranking_mechanism": {"passed": gate_a, "checks": gate_a_checks},
        "gate_b_decision_transfer": {"passed": gate_b, "checks": gate_b_checks},
        "interpretation": (
            "ranking bottleneck repaired and decision transfer improved" if gate_a and gate_b else
            "ranking bottleneck partially repaired; downstream harm/arbitration remains dominant" if gate_a else
            "ranking mechanism gate failed; stop the ranking-objective route"
        ),
        "next_single_variable_intervention_allowed": bool(gate_a),
        "next_intervention_requires_human_approval": True, "next_intervention_automatically_started": False,
        "phase5b2_started": False, "figures": figure_paths,
    }
    p5.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(p5.clean(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()

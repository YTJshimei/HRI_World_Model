"""Phase 5B-v3-R1B GARA-v2 fair test on frozen synthetic representations.

Only the capacity-matched 129-parameter Benefit mean readout is trained.  The
R1-v3 backbone, uncertainty output and HARM-v3 risk path remain frozen.  TEST
is neither materialized nor read, and no threshold or decision chain is run.
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
from scripts import run_phase5b15_decision_bottleneck as b15
from scripts import run_phase5b16_candidate_ranking as b16
from scripts import run_phase5b17ed_risk_preserving_bypass as bypass
from scripts import run_phase5b_v3_r0_fair_rebaseline as r0
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.data.robot_action_schema import HOLD_ACTION_ID
from src.evaluation.context_value_metrics import pearson, spearman
from src.evaluation.cracs_selector import (
    BIAS, EPOCH, MAE, PAIRWISE, SIGN, SPEARMAN, TOP1, TOP2,
    annotate, calibration_limits, ranking_score, select_cracs,
)
from src.models.generic_anchored_benefit import AbsoluteBenefitReadout, GenericAnchoredBenefitReadout
from src.models.independent_harm_head import RiskPreservingBypassHead
from src.multimodal.phase5b_v3_dataset import build_v3_temporal_samples
from src.multimodal.temporal_schema import LABEL
from src.training.candidate_ranking import LAMBDA_RANK, pairwise_logistic_ranking_loss

MECHANISM = "DEVELOPMENT MECHANISM RESULT"
STAGE = "Phase 5B-v3-R1B Generic-Anchored Relative Advantage FAIR TEST"
TARGET_VERSION = "BENEFIT_TARGET_V2_RUNTIME_ANCHORED"
EXPECTED_TARGET_SHA = "ae8dc040459d9dcfaabdd480309ea10ea535ed44643484e12626ad93c24939f1"
EXPECTED_MANIFEST_SHA = "ac3a620b1fb254f1a89114f3b0daa1e3ed7dc6a570cfcf083c23355a70ee542a"
EXPECTED_R1_SHA = "dbeacc97cba71515591586ce8343a5747dbb8c7034d4dbd250c3b4a58318a0ff"
EXPECTED_HARM_SHA = "2dea4137e30324daaa0344f5c5770d4758bcb563f3016ccb4d61bfa38c355b6d"
EXPECTED_ANCHOR_SHA = "88bf047e0a26a9e79ca946dbe9ec277b24ce1e84fbaf956c4378826c29d3eb0f"
TEST_READS = 0
TOLERANCE = 1e-6


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
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev/phase5b_v3_r1b_gara_fair_test")
    return parser.parse_args()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_sha(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def array_sha(value) -> str:
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_contract(args):
    actual = {
        "manifest_v3": file_sha(args.manifest_v3), "Benefit_Target_v2": file_sha(args.target_v2),
        "runtime_anchor_map": file_sha(args.anchor_map), "R1_v3_BASE": file_sha(args.r1_checkpoint),
        "HARM_v3_BASE": file_sha(args.harm_checkpoint),
    }
    expected = {
        "manifest_v3": EXPECTED_MANIFEST_SHA, "Benefit_Target_v2": EXPECTED_TARGET_SHA,
        "runtime_anchor_map": EXPECTED_ANCHOR_SHA, "R1_v3_BASE": EXPECTED_R1_SHA,
        "HARM_v3_BASE": EXPECTED_HARM_SHA,
    }
    if actual != expected:
        raise RuntimeError(f"frozen contract checksum mismatch: {actual}")
    labels = {row["candidate_id"]: row for row in read_rows(args.target_v2)}
    anchors = {row["episode_id"]: row for row in read_rows(args.anchor_map)}
    if len(labels) != 2880 or len(anchors) != 480:
        raise RuntimeError("R1A derived contract row counts changed")
    if any(int(row["runtime_anchor_action_id"]) not in range(5) or row["HOLD_excluded"] != "True" for row in anchors.values()):
        raise RuntimeError("runtime anchor A0-A4/HOLD contract changed")
    return actual, labels, anchors


def apply_target_v2(samples, labels):
    for sample in samples:
        row = labels.get(sample.sample_id)
        if row is None or row["split"] != sample.split:
            raise RuntimeError(f"Target-v2 candidate mismatch: {sample.sample_id}")
        if bool(sample.targets.feasible) != (row["feasible_unchanged"] == "True"):
            raise RuntimeError("feasibility changed from frozen R1A layer")
    return np.asarray([float(labels[sample.sample_id]["benefit_v2_runtime_anchor"]) for sample in samples], np.float64)


def extract_frozen(model, samples, normalizers, batch_size, torch, device):
    context, log_variance, old_benefit, bypass_values = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = b1.temporal_batch(samples[start:start + batch_size], normalizers, torch, device)
            output = model(batch)
            audit = model.audit_representations(batch)
            context.append(output.context_embedding.cpu())
            log_variance.append(output.benefit_log_variance.cpu())
            old_benefit.append((output.benefit_mean * normalizers["benefit_scale"] + normalizers["benefit_mean"]).cpu())
            bypass_values.append(bypass.bypass_input(audit, torch).cpu())
    return {
        "context": torch.cat(context), "log_variance": torch.cat(log_variance),
        "old_benefit": torch.cat(old_benefit).numpy().astype(np.float64),
        "bypass": torch.cat(bypass_values),
    }


def generic_indices(samples, anchors):
    by_episode = b15.group_episode(samples); result = np.empty(len(samples), np.int64)
    identity = []
    for episode_id, indices in by_episode.items():
        action = int(anchors[episode_id]["runtime_anchor_action_id"])
        match = [index for index in indices if int(samples[index].split_metadata["candidate_action_id_audit"]) == action]
        if len(match) != 1:
            raise RuntimeError(f"runtime anchor representation missing for {episode_id}")
        result[indices] = match[0]
        identity.append((episode_id, action))
    return result, identity


def rank_metrics(samples, prediction, target, model_name):
    rows = []
    for episode_id, indices in b15.group_episode(samples).items():
        predicted = np.asarray(prediction)[indices]; truth = np.asarray(target)[indices]
        feasible = np.asarray([samples[i].targets.feasible for i in indices], bool)
        actions = np.asarray([samples[i].split_metadata["candidate_action_id_audit"] for i in indices], int)
        valid = np.flatnonzero(feasible)
        best = int(np.lexsort((actions, -truth))[0]); ranks = b15.ranks_desc(predicted)
        rows.append({
            "episode_id": episode_id, "within_episode_spearman": b15.spearman(predicted, truth),
            "feasible_within_episode_spearman": b15.spearman(predicted[valid], truth[valid]),
            "feasible_pairwise_accuracy": b15.pairwise_accuracy(predicted[valid], truth[valid]),
            "gt_best_top1": int(ranks[best] == 1), "gt_best_top2": int(ranks[best] <= 2),
            "gt_best_rank": int(ranks[best]), "model": model_name,
        })
    return rows, {
        "episode_count": len(rows), "mean_within_episode_spearman": float(np.mean([r["within_episode_spearman"] for r in rows])),
        SPEARMAN: float(np.mean([r["feasible_within_episode_spearman"] for r in rows])),
        PAIRWISE: float(np.mean([r["feasible_pairwise_accuracy"] for r in rows])),
        TOP1: float(np.mean([r["gt_best_top1"] for r in rows])), TOP2: float(np.mean([r["gt_best_top2"] for r in rows])),
        "mean_gt_best_rank": float(np.mean([r["gt_best_rank"] for r in rows])),
    }


def metrics(samples, prediction, sigma, target, model_name):
    prediction, target, sigma = map(lambda x: np.asarray(x, np.float64), (prediction, target, sigma))
    feasible = np.asarray([sample.targets.feasible for sample in samples], bool)
    error = prediction - target
    rank_rows, ranking = rank_metrics(samples, prediction, target, model_name)
    selected_error = error[feasible]; selected_prediction = prediction[feasible]; selected_target = target[feasible]
    result = {
        "model": model_name, MAE: float(np.mean(np.abs(selected_error))),
        "Benefit_Pearson": pearson(selected_prediction, selected_target),
        "Benefit_Spearman": spearman(selected_prediction, selected_target),
        SIGN: float(np.mean(np.sign(selected_prediction) == np.sign(selected_target))),
        "Benefit_Uncertainty_NLL": float(np.mean(.5 * (selected_error / np.maximum(sigma[feasible], 1e-6)) ** 2 + np.log(np.maximum(sigma[feasible], 1e-6)) + .5 * np.log(2*np.pi))),
        BIAS: float(np.mean(selected_error)), "median_error": float(np.median(selected_error)),
        "positive_class_bias": float(np.mean(error[target > TOLERANCE])),
        "negative_class_bias": float(np.mean(error[target < -TOLERANCE])),
        "positive_prediction_rate": float(np.mean(prediction > 0)),
        **ranking,
    }
    result["RankingScore"] = ranking_score(result)
    return result, rank_rows


def sign_summary(samples, prediction, target, model_name, predicate=lambda _: True):
    prediction, target = np.asarray(prediction), np.asarray(target)
    base = np.asarray([predicate(sample) for sample in samples], bool)
    feasible = np.asarray([sample.targets.feasible for sample in samples], bool)
    harm = np.asarray([sample.split_metadata["harm_v2_evaluation_only"] for sample in samples], bool)
    beneficial = base & (target > TOLERANCE)
    safe = base & feasible & ~harm & (target > TOLERANCE)
    negative = base & (target < -TOLERANCE); neutral = base & (np.abs(target) <= TOLERANCE)
    predicted_positive = prediction > 0
    true_positive = predicted_positive & safe
    return {
        "synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "model": model_name,
        "safe_beneficial_count": int(safe.sum()), "safe_beneficial_episode_count": len({samples[i].episode_id for i in np.flatnonzero(safe)}),
        "predicted_positive_count": int(true_positive.sum()), "sign_failure_count": int((safe & ~predicted_positive).sum()),
        "safe_beneficial_sign_accuracy": float(np.mean(predicted_positive[safe])) if safe.any() else None,
        "GT_negative_count": int(negative.sum()), "GT_negative_false_positive_count": int((negative & predicted_positive).sum()),
        "GT_negative_false_positive_rate": float(np.mean(predicted_positive[negative])) if negative.any() else None,
        "overall_predicted_positive_rate": float(np.mean(predicted_positive[base])) if base.any() else None,
        "positive_precision": float(np.sum(predicted_positive & beneficial) / max(np.sum(predicted_positive & base), 1)),
        "safe_beneficial_precision": float(np.sum(true_positive) / max(np.sum(predicted_positive & base), 1)),
        "GT_negative_specificity": float(np.mean(~predicted_positive[negative])) if negative.any() else None,
        "neutral_count": int(neutral.sum()), "neutral_positive_rate": float(np.mean(predicted_positive[neutral])) if neutral.any() else None,
        "prediction_mean": float(np.mean(prediction[base])) if base.any() else None,
        "prediction_median": float(np.median(prediction[base])) if base.any() else None,
        "GT_benefit_mean": float(np.mean(target[base])) if base.any() else None,
        "GT_benefit_median": float(np.median(target[base])) if base.any() else None,
    }


def loss_terms(readout, context, generic_context, target, log_variance, samples, indices, scale, torch, device):
    local = torch.as_tensor(indices, dtype=torch.long)
    candidate = context[local].to(device); generic = generic_context[local].to(device)
    normalized_target = torch.as_tensor(target[indices] / scale, dtype=torch.float32, device=device)
    fixed_log_variance = log_variance[local].to(device)
    feasible = torch.tensor([samples[index].targets.feasible for index in indices], dtype=torch.bool, device=device)
    prediction = readout(candidate, generic)
    error = prediction[feasible] - normalized_target[feasible]
    nll = .5 * (error.square() * torch.exp(-fixed_log_variance[feasible]) + fixed_log_variance[feasible]).mean()
    rank, audit = pairwise_logistic_ranking_loss(prediction, normalized_target, [samples[index].episode_id for index in indices], feasible)
    return prediction, {"nll": nll, "rank": rank, "weighted_rank": LAMBDA_RANK * rank, "total": nll + LAMBDA_RANK * rank, "rank_audit": audit}


def predict(readout, context, generic_context, log_variance, scale, batch_size, torch, device):
    values = []; readout.eval()
    with torch.inference_mode():
        for start in range(0, len(context), batch_size):
            values.append(readout(context[start:start+batch_size].to(device), generic_context[start:start+batch_size].to(device)).cpu())
    benefit = torch.cat(values).numpy().astype(np.float64) * scale
    sigma = np.exp(.5 * log_variance.numpy().astype(np.float64)) * scale
    return benefit, sigma


def train_readout(name, readout, train_data, validation_data, batches, args, torch, device, reference):
    readout.to(device); optimizer = torch.optim.AdamW(readout.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    rows, states, selected_epoch, stale = [], {}, None, 0; started = time.perf_counter()
    for epoch, epoch_batches in enumerate(batches, 1):
        readout.train(); train_rows = []
        for indices in epoch_batches:
            _, terms = loss_terms(readout, train_data["context"], train_data["generic"], train_data["target"], train_data["log_variance"], train_data["samples"], indices, train_data["scale"], torch, device)
            optimizer.zero_grad(set_to_none=True); terms["total"].backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(readout.parameters(), 10.0, error_if_nonfinite=True)); optimizer.step()
            train_rows.append({"nll": float(terms["nll"].detach()), "rank": float(terms["rank"].detach()), "weighted_rank": float(terms["weighted_rank"].detach()), "total": float(terms["total"].detach()), "gradient_norm": gradient, "pairs": terms["rank_audit"].pair_count})
        prediction, sigma = predict(readout, validation_data["context"], validation_data["generic"], validation_data["log_variance"], validation_data["scale"], args.batch_size, torch, device)
        result, _ = metrics(validation_data["samples"], prediction, sigma, validation_data["target"], name)
        row = {"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "epoch": epoch, "model": name, **{f"train_{key}": float(np.mean([item[key] for item in train_rows])) for key in train_rows[0]}, **result, "parameter_checksum": state_sha(readout.state_dict())}
        audited = annotate(row, reference[MAE], reference[SIGN]); row.update({key: audited[key] for key in ("cracs_eligible", "cracs_ineligibility_reasons", "S_spearman", "S_pairwise", "S_top1", "S_top2", "RankingScore")})
        rows.append(row); states[epoch] = copy.deepcopy(readout.state_dict())
        eligible = [item for item in rows if item["cracs_eligible"]]
        if eligible:
            current, _ = select_cracs(rows, reference[MAE], reference[SIGN])
            if current[EPOCH] != selected_epoch: selected_epoch, stale = current[EPOCH], 0
            else: stale += 1
        print(f"{name} epoch={epoch:02d} loss={row['train_total']:.5f} MAE={row[MAE]:.5f} score={row['RankingScore']} best={selected_epoch} stale={stale}", flush=True)
        if selected_epoch is not None and stale >= args.patience: break
    selected, audited = select_cracs(rows, reference[MAE], reference[SIGN])
    for row, audit in zip(rows, audited):
        row.update({key: audit[key] for key in ("cracs_eligible", "cracs_ineligibility_reasons", "S_spearman", "S_pairwise", "S_top1", "S_top2", "RankingScore")})
        row["cracs_final_selected"] = row[EPOCH] == selected[EPOCH]
        if not row["cracs_eligible"]: row["RankingScore"] = None
    readout.load_state_dict(states[selected[EPOCH]]); readout.eval()
    return {"model": readout, "rows": rows, "selected": selected, "training_time_s": time.perf_counter()-started, "epochs_completed": len(rows)}


def distribution(values):
    values = np.asarray(values, np.float64)
    return {"count": int(len(values)), "mean": float(values.mean()), "std": float(values.std()), **{f"P{p}": float(np.percentile(values, p)) for p in (10,25,50,75,90)}, "min": float(values.min()), "max": float(values.max())}


def subgroup_rows(samples, predictions, target, name, predicate):
    return [{"group": name, **sign_summary(samples, value, target, model, predicate)} for model, value in predictions.items()]


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite R1B result: {args.output_dir}")
    args.output_dir.mkdir(parents=True); (args.output_dir / "checkpoints").mkdir()
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    checksums_before, labels, anchors = load_contract(args)
    episodes = {"train": build_development_split("train", 240, GENERATOR_SEED, RISK_SEED), "validation": build_development_split("validation", 240, GENERATOR_SEED+1000, RISK_SEED+1000)}
    samples = {split: build_v3_temporal_samples(value) for split, value in episodes.items()}
    targets = {split: apply_target_v2(value, labels) for split, value in samples.items()}

    checkpoint = torch.load(args.r1_checkpoint, map_location=device, weights_only=False)
    from src.models.rich_temporal_small_transformer_v3 import RichTemporalSmallTransformerV3
    backbone = RichTemporalSmallTransformerV3().to(device); backbone.load_state_dict(checkpoint["model_state_dict"]); backbone.eval()
    backbone_state_before = state_sha(backbone.state_dict())
    for parameter in backbone.parameters(): parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in backbone.parameters()): raise RuntimeError("backbone freeze failed")
    frozen = {split: extract_frozen(backbone, value, checkpoint["normalizer"], args.batch_size, torch, device) for split, value in samples.items()}
    identity = {}
    for split in samples:
        indices, identity[split] = generic_indices(samples[split], anchors)
        frozen[split].update({"generic": frozen[split]["context"][indices], "generic_indices": indices, "target": targets[split], "samples": samples[split], "scale": float(checkpoint["normalizer"]["benefit_scale"])})
    if any(int(anchors[episode]["runtime_anchor_action_id"]) != action for split in identity for episode, action in identity[split]):
        raise RuntimeError("runtime anchor identity differs from R1A")

    harm_payload = torch.load(args.harm_checkpoint, map_location=device, weights_only=False)
    harm_head = RiskPreservingBypassHead().to(device); harm_head.load_state_dict(harm_payload["model_state_dict"]); harm_head.eval()
    for parameter in harm_head.parameters(): parameter.requires_grad_(False)
    with torch.inference_mode(): harm_logits_before = harm_head(frozen["validation"]["bypass"].to(device)).cpu().numpy()

    scale = frozen["train"]["scale"]
    reference_prediction = frozen["validation"]["old_benefit"]
    reference_sigma = np.exp(.5*frozen["validation"]["log_variance"].numpy())*scale
    b0_metrics, b0_rank_rows = metrics(samples["validation"], reference_prediction, reference_sigma, targets["validation"], "B0_FROZEN_R1_V3_BASE")
    # CRACS is a frozen selector contract, including its preregistered v2 B1
    # calibration reference.  B0/Target-v2 remains a diagnostic reference and
    # must not silently redefine checkpoint eligibility.
    reference = {MAE: r0.V2_B1_REFERENCE[MAE], SIGN: r0.V2_B1_REFERENCE[SIGN]}
    batches, batch_audit = b16.make_episode_batches(samples["train"], args.epochs, args.batch_size, args.seed)

    torch.manual_seed(args.seed); h0 = AbsoluteBenefitReadout(); h0_initial = copy.deepcopy(h0.state_dict())
    torch.manual_seed(args.seed); h1 = GenericAnchoredBenefitReadout(); h1_initial = copy.deepcopy(h1.state_dict())
    if not all(torch.equal(h0_initial[key], h1_initial[key]) for key in h0_initial): raise RuntimeError("H0/H1 initialization differs")
    first_batch = batches[0][0]
    step0 = {"label": LABEL, "mechanism_result": MECHANISM, "same_data": True, "same_representation": True, "same_Target_v2": True, "same_loss_weights": True, "same_training_protocol": True, "initial_state_sha": state_sha(h0_initial), "models": {}}
    for name, model in (("H0", h0), ("H1_GARA", h1)):
        model.to(device); _, terms = loss_terms(model, frozen["train"]["context"], frozen["train"]["generic"], targets["train"], frozen["train"]["log_variance"], samples["train"], first_batch, scale, torch, device)
        model.zero_grad(set_to_none=True); terms["total"].backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        step0["models"][name] = {"weight_mean": float(model.scorer.weight.detach().mean()), "weight_std": float(model.scorer.weight.detach().std(unbiased=False)), "bias": float(model.scorer.bias.detach()), "nll": float(terms["nll"].detach()), "rank": float(terms["rank"].detach()), "weighted_rank": float(terms["weighted_rank"].detach()), "total": float(terms["total"].detach()), "gradient_norm": float(torch.sqrt(sum((value.float().square().sum() for value in gradients if value is not None))).detach()), "bias_gradient": float(model.scorer.bias.grad.detach()), "finite": all(bool(torch.isfinite(value).all()) for value in (*[terms[k] for k in ("nll","rank","total")], *[g for g in gradients if g is not None]))}
        model.load_state_dict(h0_initial if name == "H0" else h1_initial); model.zero_grad(set_to_none=True); model.cpu()

    h0_result = train_readout("H0_ABSOLUTE", h0, frozen["train"], frozen["validation"], batches, args, torch, device, reference)
    h1_result = train_readout("H1_GARA", h1, frozen["train"], frozen["validation"], batches, args, torch, device, reference)
    predictions = {"B0_FROZEN_R1_V3_BASE": reference_prediction}
    sigmas = {"B0_FROZEN_R1_V3_BASE": reference_sigma}
    for name, result in (("H0_ABSOLUTE", h0_result), ("H1_GARA", h1_result)):
        predictions[name], sigmas[name] = predict(result["model"], frozen["validation"]["context"], frozen["validation"]["generic"], frozen["validation"]["log_variance"], scale, args.batch_size, torch, device)

    comparison, rank_rows = [], {}
    for name in predictions:
        row, rank_rows[name] = metrics(samples["validation"], predictions[name], sigmas[name], targets["validation"], name); comparison.append({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, **row})
    metrics_by_name = {row["model"]: row for row in comparison}
    sign_rows = [sign_summary(samples["validation"], predictions[name], targets["validation"], name) for name in predictions]
    sign_by_name = {row["model"]: row for row in sign_rows}

    safe = np.asarray([s.targets.feasible and not s.split_metadata["harm_v2_evaluation_only"] for s in samples["validation"]], bool) & (targets["validation"] > TOLERANCE)
    h0_positive, h1_positive = predictions["H0_ABSOLUTE"] > 0, predictions["H1_GARA"] > 0
    recovered_mask = safe & ~h0_positive & h1_positive; regression_mask = safe & h0_positive & ~h1_positive
    recovery = [{"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "recovered_sign_failures": int(recovered_mask.sum()), "regressions": int(regression_mask.sum()), "net_recovery": int(recovered_mask.sum()-regression_mask.sum())}]

    margin_rows = []
    for name, value in predictions.items():
        margin_rows.append({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "model": name, "population": "safe_beneficial", **distribution(value[safe]), **{f"distance_to_zero_{k}": v for k,v in distribution(np.abs(value[safe])).items()}})

    changed_episodes = {episode for episode,row in anchors.items() if row["split"] == "validation" and row["anchor_agrees"] == "False"}
    anchor_rows = []
    for group, predicate in (("ANCHOR_SAME", lambda e:e not in changed_episodes), ("ANCHOR_CHANGED", lambda e:e in changed_episodes)):
        mask = np.asarray([predicate(s.episode_id) for s in samples["validation"]], bool)
        for name in ("H0_ABSOLUTE","H1_GARA"):
            subgroup_samples = [s for s,keep in zip(samples["validation"],mask) if keep]
            m, ranks = metrics(subgroup_samples, predictions[name][mask], sigmas[name][mask], targets["validation"][mask], name)
            sign = sign_summary(subgroup_samples, predictions[name][mask], targets["validation"][mask], name)
            anchor_rows.append({"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "group": group, **m, **{k:v for k,v in sign.items() if k.startswith("safe_beneficial") or k in ("predicted_positive_count","sign_failure_count")}})

    historical = [row for row in read_rows(PROJECT_ROOT / "results_dev/phase5b_v3_r1a_runtime_anchor_realign/historical_sign_failure_reclassification.csv") if row["category"] == "A_STILL_NEW_SAFE_BENEFICIAL_AND_PREDICTED_NONPOSITIVE"]
    index_by_id = {s.sample_id:i for i,s in enumerate(samples["validation"])}
    true_indices = np.asarray([index_by_id[row["candidate_id"]] for row in historical], int)
    history_rows = [{"synthetic_interaction": LABEL, "mechanism_result": MECHANISM, "model": name, "historical_true_failure_count": len(true_indices), "recovered_positive_count": int(np.sum(value[true_indices] > 0)), "remaining_failure_count": int(np.sum(value[true_indices] <= 0))} for name,value in predictions.items()]

    c7_rows = subgroup_rows(samples["validation"], predictions, targets["validation"], "C7", lambda s:any(str(x).startswith("C7") for x in s.split_metadata["contexts_evaluation_only"]))
    stop_rows = subgroup_rows(samples["validation"], predictions, targets["validation"], "STOP", lambda s:s.split_metadata["motion_type_evaluation_only"] == "stop")
    hold_mask = np.asarray([s.split_metadata["candidate_action_id_audit"] == HOLD_ACTION_ID for s in samples["validation"]], bool)
    beneficial_hold = hold_mask & (targets["validation"] > TOLERANCE); safe_hold = beneficial_hold & np.asarray([s.targets.feasible and not s.split_metadata["harm_v2_evaluation_only"] for s in samples["validation"]],bool)
    hold_rows, hold_negative_rows = [], []
    for name,value in predictions.items():
        ranks = {row["episode_id"]: row["gt_best_rank"] for row in rank_rows[name]}
        hold_ranks = []
        for i in np.flatnonzero(hold_mask): hold_ranks.append(int(b15.ranks_desc(value[b15.group_episode(samples["validation"])[samples["validation"][i].episode_id]])[b15.group_episode(samples["validation"])[samples["validation"][i].episode_id].index(i)]))
        hold_rows.append({"synthetic_interaction": LABEL,"mechanism_result":MECHANISM,"model":name,"beneficial_HOLD_count":int(beneficial_hold.sum()),"beneficial_HOLD_predicted_positive":int(np.sum(value[beneficial_hold]>0)),"safe_beneficial_HOLD_count":int(safe_hold.sum()),"safe_beneficial_HOLD_predicted_positive":int(np.sum(value[safe_hold]>0)),"mean_rank":float(np.mean(hold_ranks)),"median_rank":float(np.median(hold_ranks)),**{f"rank_P{p}":float(np.percentile(hold_ranks,p)) for p in (10,25,50,75,90)}})
        nonbeneficial = hold_mask & (targets["validation"] <= TOLERANCE)
        hold_negative_rows.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"model":name,"GT_nonbeneficial_HOLD_count":int(nonbeneficial.sum()),"predicted_positive_count":int(np.sum(value[nonbeneficial]>0)),"false_positive_rate":float(np.mean(value[nonbeneficial]>0))})

    h1_model = h1_result["model"]
    with torch.inference_mode():
        s_i, s_g, difference = h1_model.score_components(frozen["validation"]["context"].to(device), frozen["validation"]["generic"].to(device))
    score_rows = []
    for name,value in (("s_i",s_i.cpu().numpy()),("s_g",s_g.cpu().numpy()),("s_i_minus_s_g",difference.cpu().numpy())):
        score_rows.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"quantity":name,"interpretation":"latent score only; not physical cost" if name != "s_i_minus_s_g" else "formal normalized Benefit prediction",**distribution(value)})

    shortcut_rows = []
    getters = {"action":lambda s:str(s.split_metadata["candidate_action_id_audit"]),"HOLD":lambda s:str(s.split_metadata["candidate_action_id_audit"]==HOLD_ACTION_ID),"motion":lambda s:str(s.split_metadata["motion_type_evaluation_only"]),"context":lambda s:"|".join(map(str,s.split_metadata["contexts_evaluation_only"])) or "NONE","profile_audit":lambda s:str(s.split_metadata["person_profile_id"])}
    for dimension,getter in getters.items():
        groups=defaultdict(list)
        for sample,value in zip(samples["validation"],predictions["H1_GARA"]): groups[getter(sample)].append(float(value))
        total_var=float(np.var(predictions["H1_GARA"])); overall=float(np.mean(predictions["H1_GARA"])); between=float(sum(len(v)*(np.mean(v)-overall)**2 for v in groups.values())/len(predictions["H1_GARA"]))
        ratio=between/max(total_var,1e-12); shortcut_rows.append({"synthetic_interaction":LABEL,"mechanism_result":MECHANISM,"dimension":dimension,"group_count":len(groups),"between_variance_ratio":ratio,"near_deterministic_shortcut":ratio>=.95,"profile_ID_runtime_input":False if dimension=="profile_audit" else ""})

    with torch.inference_mode(): harm_logits_after = harm_head(frozen["validation"]["bypass"].to(device)).cpu().numpy()
    checksums_after = {"manifest_v3":file_sha(args.manifest_v3),"Benefit_Target_v2":file_sha(args.target_v2),"runtime_anchor_map":file_sha(args.anchor_map),"R1_v3_BASE":file_sha(args.r1_checkpoint),"HARM_v3_BASE":file_sha(args.harm_checkpoint)}
    backbone_state_after = state_sha(backbone.state_dict())
    harm_audit = {"label":LABEL,"mechanism_result":MECHANISM,"R1_checkpoint_sha_before":checksums_before["R1_v3_BASE"],"R1_checkpoint_sha_after":checksums_after["R1_v3_BASE"],"R1_state_sha_before":backbone_state_before,"R1_state_sha_after":backbone_state_after,"R1_backbone_unchanged":backbone_state_before==backbone_state_after,"HARM_checkpoint_sha_before":checksums_before["HARM_v3_BASE"],"HARM_checkpoint_sha_after":checksums_after["HARM_v3_BASE"],"HARM_checkpoint_unchanged":checksums_before["HARM_v3_BASE"]==checksums_after["HARM_v3_BASE"],"validation_harm_logits_before_sha":array_sha(harm_logits_before),"validation_harm_logits_after_sha":array_sha(harm_logits_after),"validation_harm_logits_max_abs_diff":float(np.max(np.abs(harm_logits_after-harm_logits_before))),"harm_outputs_unchanged":np.array_equal(harm_logits_before,harm_logits_after),"harm_optimizer_created":False}

    h0m,h1m=metrics_by_name["H0_ABSOLUTE"],metrics_by_name["H1_GARA"]; h0s,h1s=sign_by_name["H0_ABSOLUTE"],sign_by_name["H1_GARA"]
    generic_mask=np.asarray([int(s.split_metadata["candidate_action_id_audit"])==int(anchors[s.episode_id]["runtime_anchor_action_id"]) for s in samples["validation"]],bool)
    generic_max=float(np.max(np.abs(predictions["H1_GARA"][generic_mask])))
    gates={
        "Gate_A":{"name":"Contract & Isolation","checks":{"Benefit_Target_v2_SHA_correct":checksums_before["Benefit_Target_v2"]==EXPECTED_TARGET_SHA,"runtime_anchor_map_identical":checksums_before["runtime_anchor_map"]==EXPECTED_ANCHOR_SHA,"TEST_reads_zero":TEST_READS==0,"backbone_frozen":not any(p.requires_grad for p in backbone.parameters()),"harm_frozen":not any(p.requires_grad for p in harm_head.parameters()),"only_mean_parameterization_differs":True}},
        "Gate_B":{"name":"Safe-Beneficial Sign","checks":{"H1_minus_H0_at_least_0_10":h1s["safe_beneficial_sign_accuracy"]-h0s["safe_beneficial_sign_accuracy"]>=.10,"net_recovery_positive":recovery[0]["net_recovery"]>0}},
        "Gate_C":{"name":"Ranking Preservation","checks":{"spearman_drop_at_most_0_02":h1m[SPEARMAN]>=h0m[SPEARMAN]-.02,"pairwise_drop_at_most_0_02":h1m[PAIRWISE]>=h0m[PAIRWISE]-.02,"Top1_drop_at_most_0_02":h1m[TOP1]>=h0m[TOP1]-.02,"Top2_drop_at_most_0_02":h1m[TOP2]>=h0m[TOP2]-.02,"mean_GT_best_rank_worsening_at_most_0_10":h1m["mean_gt_best_rank"]<=h0m["mean_gt_best_rank"]+.10}},
        "Gate_D":{"name":"Calibration / MAE","checks":{"MAE_worsening_at_most_10_percent":h1m[MAE]<=h0m[MAE]*1.10,"finite":all(math.isfinite(float(x)) for x in (h1m[MAE],h1m[BIAS],generic_max)),"generic_zero_exact":generic_max<=TOLERANCE,"uncertainty_contract_unchanged":True}},
        "Gate_E":{"name":"Harm Isolation","checks":{"risk_outputs_unchanged":harm_audit["harm_outputs_unchanged"],"R1_backbone_unchanged":harm_audit["R1_backbone_unchanged"],"HARM_checkpoint_unchanged":harm_audit["HARM_checkpoint_unchanged"]}},
        "Gate_F":{"name":"No Degenerate Shift","checks":{"GT_negative_FPR_increase_at_most_0_05":h1s["GT_negative_false_positive_rate"]<=h0s["GT_negative_false_positive_rate"]+.05,"safe_beneficial_precision_drop_at_most_0_10":h1s["safe_beneficial_precision"]>=h0s["safe_beneficial_precision"]-.10}},
    }
    for gate in gates.values(): gate["passed"]=all(gate["checks"].values())
    gates["all_passed"]=all(gate["passed"] for gate in gates.values())

    def selection_record(name,result):
        selected=result["selected"]; limits=calibration_limits(reference[MAE],reference[SIGN])
        return {"label":LABEL,"mechanism_result":MECHANISM,"selector":"CRACS-v1 unchanged","model":name,"reference":"frozen formal v2 B1 validation reference (unchanged from R0); B0/Target-v2 is diagnostic only","reference_metrics":reference,"eligible_epochs":[r[EPOCH] for r in result["rows"] if r["cracs_eligible"]],"selected_epoch":int(selected[EPOCH]),"RankingScore":float(selected["RankingScore"]),"MAE":float(selected[MAE]),"sign_accuracy":float(selected[SIGN]),"global_bias":float(selected[BIAS]),"MAE_guard":float(selected[MAE])<=limits["max_mae"],"sign_guard":float(selected[SIGN])>=limits["min_sign_accuracy"],"epochs_completed":result["epochs_completed"]}

    h0_selection=selection_record("H0_ABSOLUTE",h0_result); h1_selection=selection_record("H1_GARA",h1_result)
    torch.save({"label":LABEL,"mechanism_result":MECHANISM,"model":"H0_ABSOLUTE","state_dict":h0_result["model"].state_dict(),"selector":h0_selection,"Benefit_Target_v2_SHA":EXPECTED_TARGET_SHA,"test_reads":0},args.output_dir/"checkpoints/h0_absolute_cracs.pt")
    torch.save({"label":LABEL,"mechanism_result":MECHANISM,"model":"H1_GARA","state_dict":h1_result["model"].state_dict(),"selector":h1_selection,"Benefit_Target_v2_SHA":EXPECTED_TARGET_SHA,"test_reads":0},args.output_dir/"checkpoints/h1_gara_cracs.pt")

    frozen_contract={"label":LABEL,"mechanism_result":MECHANISM,"stage":STAGE,"test_reads":0,"checksums_before":checksums_before,"checksums_after":checksums_after,"seed":args.seed,"R1_backbone_frozen":True,"HARM_v3_frozen":True,"trainable_parameters":{"H0":129,"H1":129},"threshold_calibration":False,"decision_chain":False,"profile_ID_runtime_input":False,"GT_runtime_inputs":False,"batch_order_audit":batch_audit}
    target_contract={"label":LABEL,"mechanism_result":MECHANISM,"target_version":TARGET_VERSION,"SHA256":EXPECTED_TARGET_SHA,"runtime_anchor_family":"A0-A4","HOLD_excluded":True,"anchor_identity_matches_R1A":True,"GT_reads_for_anchor_selection":0}
    training_config={"label":LABEL,"mechanism_result":MECHANISM,"seed":42,"optimizer":"AdamW","learning_rate":args.learning_rate,"weight_decay":.001,"batch_size":args.batch_size,"max_epochs":args.epochs,"patience":args.patience,"gradient_clip":10.0,"lambda_rank":LAMBDA_RANK,"ranking_loss":"same-episode feasible pairwise logistic; ties excluded","uncertainty":"frozen R1 per-candidate log variance; unchanged heteroscedastic NLL coupling","hyperparameter_search":False}
    stop_by_model={row["model"]:row for row in stop_rows}
    stop_regression=(stop_by_model["H1_GARA"]["safe_beneficial_sign_accuracy"] < stop_by_model["H0_ABSOLUTE"]["safe_beneficial_sign_accuracy"])
    if gates["Gate_B"]["passed"] and not gates["Gate_C"]["passed"]: outcome="ANCHOR / RANKING TRADEOFF"
    elif not gates["Gate_B"]["passed"] and all(gates[name]["passed"] for name in ("Gate_C","Gate_D","Gate_F")): outcome="GENERIC ZERO ANCHOR ALONE INSUFFICIENT"
    elif not gates["Gate_F"]["passed"]: outcome="DEGENERATE POSITIVE SHIFT"
    elif gates["all_passed"]: outcome="GARA SUCCESS"
    else: outcome="GARA FAILED OTHER PREREGISTERED GATE"
    summary={"label":LABEL,"mechanism_result":MECHANISM,"stage":STAGE,"test_reads":0,"B0":metrics_by_name["B0_FROZEN_R1_V3_BASE"],"H0":h0m,"H1":h1m,"safe_beneficial":{"B0":sign_by_name["B0_FROZEN_R1_V3_BASE"],"H0":h0s,"H1":h1s},"sign_recovery":recovery[0],"generic_prediction":{"H0":distribution(predictions["H0_ABSOLUTE"][generic_mask]),"H1":distribution(predictions["H1_GARA"][generic_mask]),"H1_max_abs":generic_max},"historical_true_failure_recovery":{row["model"]:row for row in history_rows},"score_geometry":{"common_score_drift_or_explosion":bool(max(float(s_i.abs().max()),float(s_g.abs().max()))>=10.0),"maximum_absolute_latent_score":max(float(s_i.abs().max()),float(s_g.abs().max()))},"STOP_REGRESSION_WARNING":stop_regression,"outcome_classification":outcome,"gates":gates,"GARA_formally_successful":gates["all_passed"],"ready_for_v3_safe_decision_chain_reconstruction":gates["all_passed"],"next_stage_started":False}

    io.write_json(args.output_dir/"frozen_contract.json",frozen_contract); io.write_json(args.output_dir/"target_v2_contract.json",target_contract)
    io.write_csv(args.output_dir/"b0_target_v2_reference.csv",[comparison[0]])
    io.write_json(args.output_dir/"h0_architecture.json",{"label":LABEL,"mechanism_result":MECHANISM,**h0_result["model"].architecture_audit()})
    io.write_json(args.output_dir/"h1_gara_architecture.json",{"label":LABEL,"mechanism_result":MECHANISM,**h1_result["model"].architecture_audit(),"bias_gradient_step0":step0["models"]["H1_GARA"]["bias_gradient"],"initial_bias":step0["models"]["H1_GARA"]["bias"],"selected_bias":float(h1_result["model"].scorer.bias.detach()),"bias_note":"bias cancels from every prediction; AdamW weight decay may still change its unused value"})
    io.write_json(args.output_dir/"training_config.json",training_config); io.write_json(args.output_dir/"step0_fairness.json",step0)
    io.write_csv(args.output_dir/"h0_training_curve.csv",h0_result["rows"]); io.write_csv(args.output_dir/"h1_training_curve.csv",h1_result["rows"])
    io.write_json(args.output_dir/"h0_cracs_selection.json",h0_selection); io.write_json(args.output_dir/"h1_cracs_selection.json",h1_selection)
    io.write_csv(args.output_dir/"overall_comparison.csv",comparison); io.write_csv(args.output_dir/"safe_beneficial_sign.csv",sign_rows)
    io.write_csv(args.output_dir/"sign_recovery.csv",recovery); io.write_csv(args.output_dir/"safe_beneficial_margin.csv",margin_rows)
    io.write_csv(args.output_dir/"anchor_same_changed.csv",anchor_rows); io.write_csv(args.output_dir/"historical_true_failure_recovery.csv",history_rows)
    io.write_csv(args.output_dir/"c7_audit.csv",c7_rows); io.write_csv(args.output_dir/"stop_audit.csv",stop_rows)
    io.write_csv(args.output_dir/"hold_beneficial_audit.csv",hold_rows); io.write_csv(args.output_dir/"hold_negative_protection.csv",hold_negative_rows)
    io.write_csv(args.output_dir/"score_geometry.csv",score_rows); io.write_csv(args.output_dir/"shortcut_audit.csv",shortcut_rows)
    io.write_json(args.output_dir/"harm_isolation_audit.json",harm_audit); io.write_json(args.output_dir/"gate_results.json",gates); io.write_json(args.output_dir/"summary.json",summary)
    print(json.dumps(io.clean(summary),indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()

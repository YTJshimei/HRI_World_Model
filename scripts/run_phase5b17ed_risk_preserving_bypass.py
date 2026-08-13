"""Phase 5B-1.7E-D minimal risk-preserving fusion bypass.

Synthetic TRAIN/VALIDATION only. The R1-v2-CRACS model is frozen; H1 changes
only the independent harm-v2 head input and trains one Linear(1408,1).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import audit_phase5b17ea_harm_readout_capacity as ea
from scripts import audit_phase5b17ec_representation_risk as ec
from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b17d_manifest_v2_rebaseline as d
from scripts import run_phase5b17e_independent_harm_v2 as e
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.evaluation.probabilistic_harm import harm_metrics, prevalence_baseline
from src.evaluation.representation_risk_audit import candidate_conditioning_distances, pairwise_discrimination
from src.models.independent_harm_head import IndependentHarmV2Head, RiskPreservingBypassHead
from src.multimodal.phase5b_v2_dataset import build_v2_temporal_samples
from src.multimodal.temporal_schema import LABEL

STAGE = "Phase 5B-1.7E-D Minimal Risk-Preserving Fusion Bypass"
EXPECTED_CHECKPOINT_SHA256 = ea.EXPECTED_CHECKPOINT_SHA256
EXPECTED_NORMALIZER_SHA256 = ec.EXPECTED_NORMALIZER_SHA256
H0_EXPECTED = ea.EXPECTED_LINEAR
H0_TOLERANCE = ea.REPRODUCTION_TOLERANCE


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--epochs", type=int, choices=(30,), default=30)
    parser.add_argument("--patience", type=int, choices=(5,), default=5)
    parser.add_argument("--batch-size", type=int, choices=(64,), default=64)
    parser.add_argument("--learning-rate", type=float, choices=(3e-4,), default=3e-4)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17db_checkpoint_selector_repair" / "checkpoints" / "r1_v2_cracs_best.pt")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17c_adverse_response_expansion" / "phase5b_manifest_v2.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17ed_risk_preserving_bypass")
    return parser.parse_args()


def bypass_input(stages, torch):
    required = ("R0_FINAL_FUSED", "R1_HISTORY_CONTEXT_PREFUSION", "R2_CANDIDATE_PREFUSION")
    if any(name not in stages for name in required): raise ValueError("missing frozen bypass source stage")
    value = torch.cat(tuple(stages[name].detach() for name in required), dim=-1)
    if value.ndim != 2 or value.shape[1] != RiskPreservingBypassHead.INPUT_DIM:
        raise ValueError("risk bypass must concatenate [128,1024,256] into [B,1408]")
    return value


def extract_inputs(model, samples, normalizers, batch_size, torch, device):
    h0, h1 = [], []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = e.b1.temporal_batch(samples[start:start + batch_size], normalizers, torch, device)
            normal = model.encode(batch); stages = model.audit_representations(batch)
            if not torch.equal(normal, stages["R0_FINAL_FUSED"]): raise RuntimeError("audit extraction changed normal encode output")
            h0.append(normal.detach().cpu()); h1.append(bypass_input(stages, torch).detach().cpu())
    return torch.cat(h0), torch.cat(h1)


def init_stats(head):
    weight = head.linear.weight.detach().cpu().numpy(); bias = head.linear.bias.detach().cpu().numpy()
    return {"scheme": "PyTorch nn.Linear default reset_parameters", "weight_mean": float(weight.mean()), "weight_std": float(weight.std()),
            "weight_min": float(weight.min()), "weight_max": float(weight.max()), "bias_mean": float(bias.mean()),
            "seed": 42, "full_head_checksum": e.state_sha(head.state_dict())}


def selected_result(training, validation_x, validation_y, args, torch, device):
    probability = e.probabilities(training["head"], validation_x, args.batch_size, torch, device)
    metrics = {**harm_metrics(probability, validation_y.numpy().astype(bool)), "selected_epoch": int(training["selected"]["epoch"]),
               "parameter_count": sum(parameter.numel() for parameter in training["head"].parameters())}
    return probability, metrics


def subtype_rows(samples, predictions):
    rows = []
    for model, probability in predictions.items():
        for name, predicate in ec.SUBTYPE_PREDICATES.items():
            positive = np.asarray([predicate(sample) for sample in samples], bool)
            target = np.asarray([e.harm_v2_target(sample) for sample in samples], bool); keep = positive | ~target
            metrics = harm_metrics(probability[keep], positive[keep])
            rows.append({"synthetic_interaction": LABEL, "model": model, "subtype": name,
                         "positive_probability_mean": float(probability[positive].mean()), **metrics})
    return rows


def safe_rows(samples, predictions):
    safe = np.asarray([sample.split_metadata["safe_beneficial_evaluation_only"] for sample in samples], bool)
    episodes = len({sample.episode_id for sample, keep in zip(samples, safe) if keep}); rows = []
    for model, probability in predictions.items():
        rows.append({"synthetic_interaction": LABEL, "model": model, "episode_count": episodes,
                     **ea.probability_describe(probability[safe]), "FP_at_0_5_diagnostic": float(np.mean(probability[safe] >= .5))})
    return rows


def tradeoff_rows(samples, predictions):
    mask = np.asarray([sample.split_metadata["benefit_risk_tradeoff_evaluation_only"] for sample in samples], bool); indices = np.flatnonzero(mask); rows = []
    for index in indices:
        h0, h1 = predictions["H0"][index], predictions["H1"][index]
        rows.append({"synthetic_interaction": LABEL, "candidate_id": samples[index].sample_id, "episode_id": samples[index].episode_id,
                     "action": e.ACTION_NAMES[int(samples[index].split_metadata["candidate_action_id_audit"])], "benefit": samples[index].targets.benefit,
                     "H0_harm_probability": float(h0), "H1_harm_probability": float(h1), "H1_minus_H0": float(h1 - h0),
                     "H1_score_higher": bool(h1 > h0)})
    return rows


def context_rows(samples, predictions): return ea.context_comparison(samples, predictions)


def categorical_rows(samples, predictions, dimension, getter):
    values = sorted({getter(sample) for sample in samples})
    return ea.comparison_rows(samples, predictions, dimension, values, getter)


def stop_rows(samples, predictions): return ea.stop_rows(samples, predictions)


def shortcut_record(rows_by_dimension, predictions, samples, getters):
    result = {}
    for dimension, rows in rows_by_dimension.items():
        result[dimension] = ea.shortcut_audit(rows, predictions, samples, getters[dimension])
    return result


def training_curve_rows(trainings):
    rows = []
    for name, training in trainings.items():
        for row in training["rows"]: rows.append({"model": name, **row})
    return rows


def evaluate_gates(frozen, h0, h1, baseline, safe_by_model, subtype_by_model):
    gate_a_checks = {"shared_backbone_checksum_unchanged": frozen["temporal_backbone_checksum_unchanged"],
                     "benefit_output_checksum_unchanged": frozen["benefit_output_checksum_unchanged"],
                     "ranking_behavior_checksum_unchanged": frozen["ranking_behavior_checksum_unchanged"],
                     "only_harm_head_parameters_optimized": frozen["optimizer_only_harm_heads"],
                     "H0_strict_reproduction": frozen["H0_strict_reproduction"]}
    gate_b_checks = {"AUROC_at_least_0_80": h1["AUROC"] >= .80,
                     "AUPRC_at_least_prevalence_plus_0_20": h1["AUPRC"] >= h1["prevalence"] + .20,
                     "NLL_better_than_constant": h1["NLL"] < baseline["NLL"], "Brier_better_than_constant": h1["Brier"] < baseline["Brier"]}
    gate_c_checks = {"AUROC_improvement_at_least_0_02": h1["AUROC"] - h0["AUROC"] >= .02,
                     "AUPRC_improved": h1["AUPRC"] > h0["AUPRC"], "NLL_not_worse": h1["NLL"] <= h0["NLL"],
                     "Brier_not_worse": h1["Brier"] <= h0["Brier"]}
    gate_d_checks = {"safe_mean_increase_at_most_0_05": safe_by_model["H1"]["mean"] - safe_by_model["H0"]["mean"] <= .05,
                     "safe_P90_not_higher_than_H0": safe_by_model["H1"]["P90"] <= safe_by_model["H0"]["P90"],
                     "safe_P95_not_higher_than_H0": safe_by_model["H1"]["P95"] <= safe_by_model["H0"]["P95"],
                     "GT_unsafe_AUROC_drop_at_most_0_03": subtype_by_model["H1"]["GT_UNSAFE"]["AUROC"] + .03 >= subtype_by_model["H0"]["GT_UNSAFE"]["AUROC"]}
    result = {"Gate_A": {"name": "Isolation", "checks": gate_a_checks, "passed": all(gate_a_checks.values())},
              "Gate_B": {"name": "Independent Harm Learnability", "checks": gate_b_checks, "passed": all(gate_b_checks.values())},
              "Gate_C": {"name": "Intervention Mechanism", "checks": gate_c_checks, "passed": all(gate_c_checks.values())},
              "Gate_D": {"name": "Safe-Beneficial Protection", "checks": gate_d_checks, "passed": all(gate_d_checks.values())}}
    result["all_passed"] = all(gate["passed"] for gate in result.values()); return result


def conditional_checkpoint(path, head, gates, contract, selector, torch):
    if not gates["all_passed"]: return False
    torch.save({"label": LABEL, "stage": STAGE, "model_state_dict": head.state_dict(), "architecture": head.architecture_audit(),
                "risk_bypass_input_contract": contract, "selector": selector, "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "manifest_sha256": d.EXPECTED_MANIFEST_SHA, "normalizer_sha256": EXPECTED_NORMALIZER_SHA256, "test_reads": 0}, path)
    return True


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite Phase5B-1.7E-D: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True); (args.output_dir / "checkpoints").mkdir()
    random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint_sha = e.file_sha(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256: raise RuntimeError("source checkpoint SHA mismatch")
    model, payload = e.load_frozen(args.checkpoint, torch, device); before = d.model_sha(model); state_before = copy.deepcopy(model.state_dict())
    backbone_before = e.state_sha(state_before, exclude=("benefit.", "uncertainty.", "harm.")); benefit_before = e.state_sha(state_before, prefixes=("benefit.", "uncertainty.")); ranking_before = e.state_sha(state_before, exclude=("harm.",))
    samples = {"train": build_v2_temporal_samples(build_development_split("train", 240, GENERATOR_SEED, RISK_SEED)),
               "validation": build_v2_temporal_samples(build_development_split("validation", 240, GENERATOR_SEED + 1000, RISK_SEED + 1000))}
    manifest_contract = d.manifest_contract(args.manifest, samples["train"] + samples["validation"]); normalizers = payload["normalizer"]
    if normalizers["sha256"] != EXPECTED_NORMALIZER_SHA256: raise RuntimeError("normalizer SHA mismatch")
    inputs = {split: extract_inputs(model, values, normalizers, args.batch_size, torch, device) for split, values in samples.items()}
    train_y = torch.tensor([e.harm_v2_target(sample) for sample in samples["train"]], dtype=torch.float32)
    validation_y = torch.tensor([e.harm_v2_target(sample) for sample in samples["validation"]], dtype=torch.float32)
    with torch.inference_mode():
        benefit_before_behavior = model.benefit(inputs["validation"][0].to(device)).cpu().numpy()
        uncertainty_before_behavior = model.uncertainty(inputs["validation"][0].to(device)).cpu().numpy()

    torch.manual_seed(args.seed); h0_head = IndependentHarmV2Head().to(device); h0_init = init_stats(h0_head)
    h0_training = e.train_head(h0_head, inputs["train"][0], train_y, inputs["validation"][0], validation_y, args, torch, device)
    h0_probability, h0_metrics = selected_result(h0_training, inputs["validation"][0], validation_y, args, torch, device)
    h0_deltas = {name: abs(float(h0_metrics[name]) - H0_EXPECTED[name]) for name in ("AUROC", "AUPRC", "NLL", "Brier", "ECE")}
    h0_reproduced = h0_metrics["selected_epoch"] == H0_EXPECTED["selected_epoch"] and max(h0_deltas.values()) <= H0_TOLERANCE
    if not h0_reproduced: raise RuntimeError("H0 failed strict Phase5B-1.7E reproduction")

    torch.manual_seed(args.seed); h1_head = RiskPreservingBypassHead().to(device); h1_init = init_stats(h1_head)
    h1_training = e.train_head(h1_head, inputs["train"][1], train_y, inputs["validation"][1], validation_y, args, torch, device)
    h1_probability, h1_metrics = selected_result(h1_training, inputs["validation"][1], validation_y, args, torch, device)
    predictions = {"H0": h0_probability, "H1": h1_probability}; global_rows = [{"synthetic_interaction": LABEL, "model": name, **metrics} for name, metrics in (("H0", h0_metrics), ("H1", h1_metrics))]
    safe = safe_rows(samples["validation"], predictions); tradeoff = tradeoff_rows(samples["validation"], predictions); subtype = subtype_rows(samples["validation"], predictions)
    contexts = context_rows(samples["validation"], predictions)
    getters = {"motion": lambda sample: sample.split_metadata["motion_type_evaluation_only"], "action": lambda sample: int(sample.split_metadata["candidate_action_id_audit"]),
               "profile": lambda sample: int(sample.split_metadata["person_profile_id"]), "context": lambda sample: "|".join(map(str, sample.split_metadata["contexts_evaluation_only"]))}
    grouped = {name: categorical_rows(samples["validation"], predictions, name, getter) for name, getter in getters.items()}
    stop = stop_rows(samples["validation"], predictions)
    target = validation_y.numpy().astype(bool); episodes = np.asarray([sample.episode_id for sample in samples["validation"]])
    within = [{"synthetic_interaction": LABEL, "model": name, **pairwise_discrimination(score, target, episodes)} for name, score in predictions.items()]
    geometry = {"label": LABEL, "input": "z_harm=[z_final;z_human;z_candidate]", **candidate_conditioning_distances(inputs["validation"][1], target, episodes),
                "reference_R0": candidate_conditioning_distances(inputs["validation"][0], target, episodes)}
    shortcuts = shortcut_record(grouped, predictions, samples["validation"], getters)
    baseline = prevalence_baseline(train_y.numpy().astype(bool), target)

    with torch.inference_mode():
        benefit_after_behavior = model.benefit(inputs["validation"][0].to(device)).cpu().numpy(); uncertainty_after_behavior = model.uncertainty(inputs["validation"][0].to(device)).cpu().numpy()
    ranking_behavior_before = np.column_stack((benefit_before_behavior, uncertainty_before_behavior))
    ranking_behavior_after = np.column_stack((benefit_after_behavior, uncertainty_after_behavior))
    after = d.model_sha(model); state_after = model.state_dict()
    frozen = {"label": LABEL, "test_reads": 0, "source_checkpoint_sha256": checkpoint_sha, "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
              "manifest_sha256": e.file_sha(args.manifest), "expected_manifest_sha256": d.EXPECTED_MANIFEST_SHA,
              "normalizer_sha256": normalizers["sha256"], "expected_normalizer_sha256": EXPECTED_NORMALIZER_SHA256,
              "full_model_checksum_before": before, "full_model_checksum_after": after, "full_model_unchanged": before == after,
              "temporal_backbone_checksum_unchanged": backbone_before == e.state_sha(state_after, exclude=("benefit.", "uncertainty.", "harm.")),
              "benefit_head_checksum_unchanged": benefit_before == e.state_sha(state_after, prefixes=("benefit.", "uncertainty.")),
              "ranking_parameter_checksum_unchanged": ranking_before == e.state_sha(state_after, exclude=("harm.",)),
              "ranking_behavior_checksum_before": e.array_sha(ranking_behavior_before),
              "ranking_behavior_checksum_after": e.array_sha(ranking_behavior_after),
              "ranking_behavior_checksum_unchanged": e.array_sha(ranking_behavior_before) == e.array_sha(ranking_behavior_after),
              "benefit_output_checksum_before": e.array_sha(benefit_before_behavior), "benefit_output_checksum_after": e.array_sha(benefit_after_behavior),
              "benefit_output_checksum_unchanged": e.array_sha(benefit_before_behavior) == e.array_sha(benefit_after_behavior),
              "uncertainty_output_unchanged": e.array_sha(uncertainty_before_behavior) == e.array_sha(uncertainty_after_behavior),
              "all_shared_parameters_require_grad_false": not any(parameter.requires_grad for parameter in model.parameters()),
              "optimizer_only_harm_heads": h0_training["optimizer_exactly_head"] and h1_training["optimizer_exactly_head"],
              "H0_strict_reproduction": h0_reproduced, "manifest_contract_passed": manifest_contract["passed"]}
    contract = {"label": LABEL, "input_name": "z_harm", "shape": ["B", 1408], "stop_gradient": True,
                "components": [
                    {"name": "z_final", "source_layer": "frozen model.fusion output / R0_FINAL_FUSED", "shape": ["B", 128], "runtime_sources": "all canonical runtime streams", "contains_GT_only_information": False},
                    {"name": "z_human", "source_layer": "existing per-history projections pooled plus scene / R1_HISTORY_CONTEXT_PREFUSION", "shape": ["B", 1024], "runtime_sources": "skeleton,human motion,robot,functional,visibility,WM diagnostic,interaction history,scene", "contains_GT_only_information": False},
                    {"name": "z_candidate", "source_layer": "candidate future/action projections / R2_CANDIDATE_PREFUSION", "shape": ["B", 256], "runtime_sources": "deterministic candidate robot future and structured candidate action", "contains_GT_only_information": False}],
                "forbidden_fields_absent": ["GT future human trajectory", "GT harm", "GT benefit", "profile ID", "oracle action"],
                "runtime_valid": True, "profile_id_in_input": False, "GT_future_in_input": False}
    safe_by_model = {row["model"]: row for row in safe}; subtype_by_model = {name: {row["subtype"]: row for row in subtype if row["model"] == name} for name in predictions}
    gates = evaluate_gates(frozen, h0_metrics, h1_metrics, baseline, safe_by_model, subtype_by_model)
    checkpoint_path = args.output_dir / "checkpoints" / "harm_v2_risk_bypass_head.pt"
    checkpoint_written = conditional_checkpoint(checkpoint_path, h1_training["head"], gates, contract, h1_training["selected"], torch)
    architecture = {"label": LABEL, **h1_training["head"].architecture_audit(), "initialization": h1_init, "stop_gradient_input": True}
    config = {"label": LABEL, "seed": args.seed, "learning_rate": args.learning_rate, "batch_size": args.batch_size, "max_epochs": args.epochs, "patience": args.patience,
              "loss": "unweighted BCEWithLogitsLoss", "selector": "PHS-v1: NLL -> Brier -> AUROC -> earlier epoch", "class_weighting": False, "oversampling": False,
              "focal_loss": False, "threshold_tuning": False, "same_batch_order_seed": True, "H0_initialization": h0_init, "H1_initialization": h1_init}
    reproduction = {"label": LABEL, "expected": H0_EXPECTED, "actual": h0_metrics, "metric_deltas": h0_deltas, "strict_tolerance": H0_TOLERANCE, "reproduced": h0_reproduced}
    summary = {"label": LABEL, "stage": STAGE, "test_reads": 0, "H0": h0_metrics, "H1": h1_metrics,
               "H1_minus_H0": {key: h1_metrics[key] - h0_metrics[key] for key in ("AUROC", "AUPRC", "NLL", "Brier", "ECE")},
               "constant_prevalence_baseline": baseline, "gates": gates, "formal_checkpoint_written": checkpoint_written,
               "ready_for_phase5b17f": bool(gates["all_passed"] and checkpoint_written), "phase5b17f_started": False,
               "threshold_calibration_performed": False, "shortcut_audit": shortcuts}
    io.write_json(args.output_dir / "frozen_contract.json", frozen); io.write_json(args.output_dir / "risk_bypass_input_contract.json", contract)
    io.write_json(args.output_dir / "h0_reproduction.json", reproduction); io.write_json(args.output_dir / "h1_architecture.json", architecture)
    io.write_json(args.output_dir / "training_config.json", config); io.write_csv(args.output_dir / "training_curve.csv", training_curve_rows({"H0": h0_training, "H1": h1_training}))
    io.write_json(args.output_dir / "phs_selection.json", {"label": LABEL, "H0": h0_training["selected"], "H1": h1_training["selected"]})
    io.write_csv(args.output_dir / "global_metrics.csv", global_rows); io.write_csv(args.output_dir / "safe_beneficial_comparison.csv", safe)
    io.write_csv(args.output_dir / "benefit_risk_tradeoff.csv", tradeoff); io.write_csv(args.output_dir / "by_harm_subtype.csv", subtype)
    io.write_csv(args.output_dir / "by_context.csv", contexts); io.write_csv(args.output_dir / "by_motion.csv", grouped["motion"])
    io.write_csv(args.output_dir / "by_action.csv", grouped["action"]); io.write_csv(args.output_dir / "by_profile.csv", grouped["profile"])
    io.write_csv(args.output_dir / "stop_audit.csv", stop); io.write_csv(args.output_dir / "within_episode_risk.csv", within)
    io.write_json(args.output_dir / "candidate_conditioning_geometry.json", geometry); io.write_json(args.output_dir / "parameter_freeze_audit.json", frozen)
    io.write_json(args.output_dir / "gate_results.json", gates); io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()

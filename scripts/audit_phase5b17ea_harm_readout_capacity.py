"""Phase 5B-1.7E-A frozen-representation harm readout capacity audit."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b17d_manifest_v2_rebaseline as d
from scripts import run_phase5b17e_independent_harm_v2 as e
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.evaluation.probabilistic_harm import harm_metrics
from src.models.independent_harm_head import IndependentHarmV2Head, MinimalNonlinearHarmV2Probe
from src.multimodal.phase5b_v2_dataset import build_v2_temporal_samples
from src.multimodal.temporal_schema import LABEL
from src.training.independent_harm import harm_v2_target

STAGE = "Phase 5B-1.7E-A Frozen Representation Harm-v2 Readout Capacity Audit"
EXPECTED_LINEAR = {"AUROC": .7750483796995425, "AUPRC": .5706201254893742,
                   "NLL": .46160859510436714, "Brier": .14966288243132198,
                   "ECE": .03965453734621406, "selected_epoch": 30}
REPRODUCTION_TOLERANCE = 1e-12
EXPECTED_CHECKPOINT_SHA256 = "eb8321e9b4f3cd7213ec52c48169857d37980f5457568fc832ff486157914ce8"


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
    parser.add_argument("--phase5b17e-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17e_independent_harm_v2")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17ea_harm_readout_capacity_audit")
    return parser.parse_args()


def train_probe(probe, train_x, train_y, validation_x, validation_y, args, torch, device):
    return e.train_head(probe.to(device), train_x, train_y, validation_x, validation_y, args, torch, device)


def metadata_arrays(samples):
    return {
        "candidate_id": np.asarray([sample.sample_id for sample in samples]),
        "episode_id": np.asarray([sample.episode_id for sample in samples]),
        "split": np.asarray([sample.split for sample in samples]),
        "harm_v2": np.asarray([harm_v2_target(sample) for sample in samples], bool),
        "gt_unsafe": np.asarray([sample.targets.gt_unsafe for sample in samples], bool),
        "excessive_deceleration": np.asarray([sample.split_metadata["excessive_deceleration_evaluation_only"] for sample in samples], bool),
        "abrupt_lateral_response": np.asarray([sample.split_metadata["abrupt_lateral_response_evaluation_only"] for sample in samples], bool),
        "abrupt_heading_change": np.asarray([sample.split_metadata["abrupt_heading_change_evaluation_only"] for sample in samples], bool),
        "safe_beneficial": np.asarray([sample.split_metadata["safe_beneficial_evaluation_only"] for sample in samples], bool),
        "benefit_risk_tradeoff": np.asarray([sample.split_metadata["benefit_risk_tradeoff_evaluation_only"] for sample in samples], bool),
        "motion": np.asarray([sample.split_metadata["motion_type_evaluation_only"] for sample in samples]),
        "contexts": np.asarray(["|".join(map(str, sample.split_metadata["contexts_evaluation_only"])) for sample in samples]),
        "action": np.asarray([sample.split_metadata["candidate_action_id_audit"] for sample in samples], np.int64),
        "profile_audit_only": np.asarray([sample.split_metadata["person_profile_id"] for sample in samples], np.int64),
    }


def save_embedding_cache(path, train_x, validation_x, train_samples, validation_samples):
    train_meta, validation_meta = metadata_arrays(train_samples), metadata_arrays(validation_samples)
    payload = {"train_embedding": train_x.numpy(), "validation_embedding": validation_x.numpy()}
    payload.update({f"train_{key}": value for key, value in train_meta.items()})
    payload.update({f"validation_{key}": value for key, value in validation_meta.items()})
    np.savez_compressed(path, **payload)
    return {"label": LABEL, "cache_file": str(path), "cache_sha256": e.file_sha(path),
            "embedding_dimension": 128, "train_candidates": len(train_x), "validation_candidates": len(validation_x),
            "splits": ["train", "validation"], "test_embeddings": 0, "test_reads": 0,
            "runtime_probe_input": ["128D frozen representation only"],
            "audit_only_fields": sorted(train_meta), "profile_id_in_probe_input": False}


def selected_metrics(training, validation_x, validation_y, args, torch, device):
    probability = e.probabilities(training["head"], validation_x, args.batch_size, torch, device)
    return probability, {**harm_metrics(probability, validation_y.numpy().astype(bool)),
                         "selected_epoch": training["selected"]["epoch"], "parameter_count": sum(parameter.numel() for parameter in training["head"].parameters())}


def probe_subtypes(train_x, validation_x, train_samples, validation_samples, args, torch, device):
    definitions = {
        "GT_UNSAFE": lambda sample: sample.targets.gt_unsafe,
        "EXCESSIVE_DECELERATION": lambda sample: sample.split_metadata["excessive_deceleration_evaluation_only"],
        "ABRUPT_LATERAL_RESPONSE": lambda sample: sample.split_metadata["abrupt_lateral_response_evaluation_only"],
        "ABRUPT_HEADING_CHANGE": lambda sample: sample.split_metadata["abrupt_heading_change_evaluation_only"],
    }
    rows = []
    for name, predicate in definitions.items():
        train_positive = np.asarray([predicate(sample) for sample in train_samples], bool)
        train_harm = np.asarray([harm_v2_target(sample) for sample in train_samples], bool)
        validation_positive = np.asarray([predicate(sample) for sample in validation_samples], bool)
        validation_harm = np.asarray([harm_v2_target(sample) for sample in validation_samples], bool)
        train_keep, validation_keep = train_positive | ~train_harm, validation_positive | ~validation_harm
        torch.manual_seed(args.seed); probe = IndependentHarmV2Head()
        trained = train_probe(probe, train_x[train_keep], torch.tensor(train_positive[train_keep], dtype=torch.float32),
                              validation_x[validation_keep], torch.tensor(validation_positive[validation_keep], dtype=torch.float32), args, torch, device)
        probability, metrics = selected_metrics(trained, validation_x[validation_keep], torch.tensor(validation_positive[validation_keep], dtype=torch.float32), args, torch, device)
        rows.append({"synthetic_interaction": LABEL, "diagnostic_only": True, "subtype": name,
                     "train_positive_count": int(train_positive.sum()), "train_negative_count": int((~train_harm).sum()),
                     "validation_positive_count": int(validation_positive.sum()), "validation_negative_count": int((~validation_harm).sum()),
                     **metrics, "formal_checkpoint_allowed": False})
    return rows


def probability_describe(probability):
    values = np.asarray(probability, np.float64)
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            **{f"P{p}": float(np.percentile(values, p)) for p in (10, 50, 90, 95)}}


def comparison_rows(samples, predictions, dimension, values, getter):
    target = np.asarray([harm_v2_target(sample) for sample in samples], bool); rows = []
    for model, probability in predictions.items():
        for value in values:
            mask = np.asarray([getter(sample) == value for sample in samples])
            metrics = harm_metrics(probability[mask], target[mask])
            rows.append({"synthetic_interaction": LABEL, "model": model, "dimension": dimension, "group": value,
                         "episode_count": len({sample.episode_id for sample, keep in zip(samples, mask) if keep}), **metrics})
    return rows


def context_comparison(samples, predictions):
    target = np.asarray([harm_v2_target(sample) for sample in samples], bool); rows = []
    for model, probability in predictions.items():
        for context in ("C7", "C8", "C9"):
            mask = np.asarray([any(str(value).startswith(context) for value in sample.split_metadata["contexts_evaluation_only"]) for sample in samples])
            rows.append({"synthetic_interaction": LABEL, "model": model, "dimension": "context", "group": context,
                         "episode_count": len({sample.episode_id for sample, keep in zip(samples, mask) if keep}), **harm_metrics(probability[mask], target[mask])})
    return rows


def semantic_separation(samples, probability):
    target = np.asarray([harm_v2_target(sample) for sample in samples], bool)
    safe = np.asarray([sample.split_metadata["safe_beneficial_evaluation_only"] for sample in samples], bool)
    return e.separation(samples, probability) | {"safe_beneficial_false_positive_rate_at_0_5_diagnostic": float(np.mean(probability[safe] >= .5))}


def safe_rows(samples, predictions):
    safe = np.asarray([sample.split_metadata["safe_beneficial_evaluation_only"] for sample in samples], bool); rows = []
    episode_count = len({sample.episode_id for sample, keep in zip(samples, safe) if keep})
    for model, probability in predictions.items():
        rows.append({"synthetic_interaction": LABEL, "model": model, "group": "safe_beneficial", "episode_count": episode_count,
                     **probability_describe(probability[safe]), "false_positive_at_0_5_diagnostic": float(np.mean(probability[safe] >= .5))})
    return rows


def tradeoff_rows(samples, predictions):
    mask = np.asarray([sample.split_metadata["benefit_risk_tradeoff_evaluation_only"] for sample in samples], bool)
    indices = np.flatnonzero(mask); rows = []
    for index in indices:
        base = {"synthetic_interaction": LABEL, "candidate_id": samples[index].sample_id, "episode_id": samples[index].episode_id,
                "action": e.ACTION_NAMES[int(samples[index].split_metadata["candidate_action_id_audit"])], "benefit": samples[index].targets.benefit}
        for model, probability in predictions.items():
            rank = np.argsort(np.argsort(-probability, kind="stable"), kind="stable") + 1
            rows.append({**base, "model": model, "harm_probability": float(probability[index]), "global_probability_rank": int(rank[index]),
                         "above_0_5_diagnostic": bool(probability[index] >= .5)})
    return rows


def subtype_comparison(samples, predictions):
    rows = []
    for model, probability in predictions.items():
        for name, predicate in e.SUBTYPES.items():
            positive = np.asarray([predicate(sample) for sample in samples], bool); target = np.asarray([harm_v2_target(sample) for sample in samples], bool)
            comparison = positive | ~target
            metrics = harm_metrics(probability[comparison], positive[comparison])
            rows.append({"synthetic_interaction": LABEL, "model": model, "subtype": name,
                         "positive_count": int(positive.sum()), **metrics, "positive_probability_mean": float(probability[positive].mean())})
    return rows


def stop_rows(samples, predictions):
    target = np.asarray([harm_v2_target(sample) for sample in samples], bool)
    mask = np.asarray([sample.split_metadata["motion_type_evaluation_only"] == "stop" for sample in samples]); rows = []
    for model, probability in predictions.items():
        metrics = harm_metrics(probability[mask], target[mask])
        rows.append({"synthetic_interaction": LABEL, "model": model, "true_harm_rate": float(target[mask].mean()),
                     "mean_predicted_probability": float(probability[mask].mean()), "underconfidence_gap": float(probability[mask].mean() - target[mask].mean()), **metrics})
    return rows


def shortcut_audit(rows, predictions, samples, getter):
    result = {}
    for model, probability in predictions.items(): result[model] = e.variance_shortcut(rows, samples, probability, getter)
    return result


def geometry(embedding, samples):
    x = embedding.numpy().astype(np.float64); target = np.asarray([harm_v2_target(sample) for sample in samples], bool)
    groups = {"harm_v2_positive": target,
              "safe_beneficial": np.asarray([sample.split_metadata["safe_beneficial_evaluation_only"] for sample in samples], bool),
              **{name: np.asarray([predicate(sample) for sample in samples], bool) for name, predicate in e.SUBTYPES.items()}}
    centroids, variances = {}, {}
    for name, mask in groups.items():
        centroids[name] = x[mask].mean(0); variances[name] = float(np.mean(np.sum((x[mask] - centroids[name]) ** 2, axis=1)))
    pairs = []
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = centroids[left], centroids[right]
            pairs.append({"left": left, "right": right, "euclidean_distance": float(np.linalg.norm(a - b)),
                          "cosine_distance": float(1 - np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))})
    return {"label": LABEL, "split": "validation", "embedding_dim": 128,
            "groups": {name: {"count": int(groups[name].sum()), "within_group_squared_euclidean_variance": variances[name]} for name in names},
            "centroid_distances": pairs, "t_sne_used_for_quantitative_conclusion": False}


def make_figure(output, embedding, samples):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True)
    x = embedding.numpy().astype(np.float64); centered = x - x.mean(0); _, _, vt = np.linalg.svd(centered, full_matrices=False); pca = centered @ vt[:2].T
    target = np.asarray([harm_v2_target(sample) for sample in samples], bool)
    plt.figure(figsize=(7, 5)); plt.scatter(pca[~target, 0], pca[~target, 1], s=7, alpha=.35, label="harm-v2 negative")
    plt.scatter(pca[target, 0], pca[target, 1], s=8, alpha=.5, label="harm-v2 positive"); plt.legend(); plt.title(f"{LABEL}\nPCA diagnostic only")
    path = folder / "embedding_pca.png"; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); return [str(path)]


def gate_results(linear_metrics, mlp_metrics, linear_semantic, mlp_semantic, linear_subtype, mlp_subtype):
    reproduction_deltas = {name: abs(float(linear_metrics[name]) - EXPECTED_LINEAR[name]) for name in ("AUROC", "AUPRC", "NLL", "Brier", "ECE")}
    gate_a_checks = {"selected_epoch_identical": linear_metrics["selected_epoch"] == EXPECTED_LINEAR["selected_epoch"],
                     "all_metric_deltas_within_tolerance": max(reproduction_deltas.values()) <= REPRODUCTION_TOLERANCE}
    gate_b_checks = {"MLP_AUROC_at_least_0_80": mlp_metrics["AUROC"] >= .80,
                     "AUROC_improvement_at_least_0_02": mlp_metrics["AUROC"] - linear_metrics["AUROC"] >= .02,
                     "NLL_not_worse": mlp_metrics["NLL"] <= linear_metrics["NLL"], "Brier_not_worse": mlp_metrics["Brier"] <= linear_metrics["Brier"]}
    gate_c_checks = {"MLP_semantic_AUROC_at_least_0_85": mlp_semantic["harm_positive_vs_safe_beneficial_AUROC"] >= .85,
                     "safe_beneficial_mean_increase_at_most_0_10": mlp_semantic["safe_beneficial"]["mean"] - linear_semantic["safe_beneficial"]["mean"] <= .10,
                     "GT_unsafe_AUROC_drop_at_most_0_03": mlp_subtype["GT_UNSAFE"] + .03 >= linear_subtype["GT_UNSAFE"]}
    return {"Gate_A": {"name": "Reproduction", "checks": gate_a_checks, "metric_deltas": reproduction_deltas, "passed": all(gate_a_checks.values())},
            "Gate_B": {"name": "Readout Capacity Evidence", "checks": gate_b_checks, "passed": all(gate_b_checks.values())},
            "Gate_C": {"name": "Semantic Protection", "checks": gate_c_checks, "passed": all(gate_c_checks.values())}}


def classify_root_cause(gates, linear_metrics, mlp_metrics, subtype_probe_rows):
    weak = [row["subtype"] for row in subtype_probe_rows if row["AUROC"] < .80]
    improvement = mlp_metrics["AUROC"] - linear_metrics["AUROC"]
    if gates["Gate_B"]["passed"] and not weak:
        selected = "C"; name = "HETEROGENEOUS UNION / NONLINEAR READOUT BOTTLENECK"
    elif gates["Gate_B"]["passed"]:
        selected = "A"; name = "LINEAR READOUT CAPACITY BOTTLENECK"
    elif len(weak) == 1:
        selected = "D"; name = "SUBTYPE-SPECIFIC REPRESENTATION FAILURE"
    elif len(weak) >= 2 and (mlp_metrics["AUROC"] < .80 or improvement < .02):
        selected = "B"; name = "FROZEN REPRESENTATION BOTTLENECK"
    else:
        selected = "F"; name = "MULTIPLE INTERACTING CAUSES"
    return {"label": LABEL, "selected_class": selected, "selected_name": name,
            "linear_AUROC": linear_metrics["AUROC"], "MLP_AUROC": mlp_metrics["AUROC"], "AUROC_improvement": improvement,
            "subtype_linear_probes_below_0_80": weak, "calibration_only_excluded": mlp_metrics["AUROC"] < .80,
            "formal_intervention_implemented": False}


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite Phase5B-1.7E-A: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True); random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    checkpoint_sha256 = e.file_sha(args.checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"source checkpoint SHA256 mismatch: {checkpoint_sha256}")
    model, checkpoint_payload = e.load_frozen(args.checkpoint, torch, device); before = d.model_sha(model)
    state = model.state_dict(); backbone_before = e.state_sha(state, exclude=("benefit.", "uncertainty.", "harm.")); benefit_before = e.state_sha(state, prefixes=("benefit.", "uncertainty.")); ranking_before = e.state_sha(state, exclude=("harm.",))
    episodes = {"train": build_development_split("train", 240, GENERATOR_SEED, RISK_SEED),
                "validation": build_development_split("validation", 240, GENERATOR_SEED + 1000, RISK_SEED + 1000)}
    samples = {name: build_v2_temporal_samples(values) for name, values in episodes.items()}
    contract = d.manifest_contract(args.manifest, samples["train"] + samples["validation"]); normalizers = checkpoint_payload["normalizer"]
    train_x, train_y = e.encode_samples(model, samples["train"], normalizers, args.batch_size, torch, device)
    validation_x, validation_y = e.encode_samples(model, samples["validation"], normalizers, args.batch_size, torch, device)
    cache_contract = save_embedding_cache(args.output_dir / "embedding_cache.npz", train_x, validation_x, samples["train"], samples["validation"])

    torch.manual_seed(args.seed); linear_training = train_probe(IndependentHarmV2Head(), train_x, train_y, validation_x, validation_y, args, torch, device)
    linear_probability, linear_metrics = selected_metrics(linear_training, validation_x, validation_y, args, torch, device)
    if max(abs(float(linear_metrics[name]) - EXPECTED_LINEAR[name]) for name in ("AUROC", "AUPRC", "NLL", "Brier", "ECE")) > REPRODUCTION_TOLERANCE:
        raise RuntimeError("Linear probe failed strict Phase5B-1.7E reproduction")
    torch.manual_seed(args.seed); mlp_training = train_probe(MinimalNonlinearHarmV2Probe(), train_x, train_y, validation_x, validation_y, args, torch, device)
    mlp_probability, mlp_metrics = selected_metrics(mlp_training, validation_x, validation_y, args, torch, device)
    predictions = {"Linear": linear_probability, "MinimalMLP": mlp_probability}
    subtype_probes = probe_subtypes(train_x, validation_x, samples["train"], samples["validation"], args, torch, device)
    subtype_rows = subtype_comparison(samples["validation"], predictions)
    context_rows = context_comparison(samples["validation"], predictions)
    motions = sorted({sample.split_metadata["motion_type_evaluation_only"] for sample in samples["validation"]})
    motion_rows = comparison_rows(samples["validation"], predictions, "motion", motions, lambda sample: sample.split_metadata["motion_type_evaluation_only"])
    actions = sorted({int(sample.split_metadata["candidate_action_id_audit"]) for sample in samples["validation"]})
    action_rows = comparison_rows(samples["validation"], predictions, "action", actions, lambda sample: int(sample.split_metadata["candidate_action_id_audit"]))
    for row in action_rows: row["action_name"] = e.ACTION_NAMES[int(row["group"])]
    profiles = sorted({int(sample.split_metadata["person_profile_id"]) for sample in samples["validation"]})
    profile_rows = comparison_rows(samples["validation"], predictions, "profile_audit_only", profiles, lambda sample: int(sample.split_metadata["person_profile_id"]))
    safe = safe_rows(samples["validation"], predictions); tradeoff = tradeoff_rows(samples["validation"], predictions); stop = stop_rows(samples["validation"], predictions)
    semantics = {name: semantic_separation(samples["validation"], probability) for name, probability in predictions.items()}
    subtype_auc = {model: {row["subtype"]: row["AUROC"] for row in subtype_rows if row["model"] == model} for model in predictions}
    gates = gate_results(linear_metrics, mlp_metrics, semantics["Linear"], semantics["MinimalMLP"], subtype_auc["Linear"], subtype_auc["MinimalMLP"])
    gates["all_passed"] = all(item["passed"] for item in gates.values())
    root_cause = classify_root_cause(gates, linear_metrics, mlp_metrics, subtype_probes)
    geometry_audit = geometry(validation_x, samples["validation"]); figures = make_figure(args.output_dir, validation_x, samples["validation"])
    shortcuts = {"action": shortcut_audit(action_rows, predictions, samples["validation"], lambda sample: int(sample.split_metadata["candidate_action_id_audit"])),
                 "motion": shortcut_audit(motion_rows, predictions, samples["validation"], lambda sample: sample.split_metadata["motion_type_evaluation_only"]),
                 "profile": shortcut_audit(profile_rows, predictions, samples["validation"], lambda sample: int(sample.split_metadata["person_profile_id"])),
                 "profile_id_in_probe_input": False}
    after = d.model_sha(model); state_after = model.state_dict()
    frozen = {"label": LABEL, "manifest_sha256": e.file_sha(args.manifest), "expected_manifest_sha256": d.EXPECTED_MANIFEST_SHA,
              "normalizer_sha256": normalizers["sha256"], "source_checkpoint_sha256": checkpoint_sha256,
              "source_checkpoint_expected_sha256": EXPECTED_CHECKPOINT_SHA256, "test_reads": 0,
              "backbone_requires_grad_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
              "entire_model_checksum_before": before, "entire_model_checksum_after": after, "entire_model_unchanged": before == after,
              "backbone_checksum_unchanged": backbone_before == e.state_sha(state_after, exclude=("benefit.", "uncertainty.", "harm.")),
              "benefit_checksum_unchanged": benefit_before == e.state_sha(state_after, prefixes=("benefit.", "uncertainty.")),
              "ranking_checksum_unchanged": ranking_before == e.state_sha(state_after, exclude=("harm.",)),
              "manifest_contract_passed": contract["passed"], "formal_harm_checkpoint_written": False}
    linear_record = {"label": LABEL, "probe": "Linear(128,1)", "reproduced": gates["Gate_A"]["passed"], "expected": EXPECTED_LINEAR,
                     "actual": linear_metrics, "training_curve": linear_training["rows"]}
    mlp_record = {"label": LABEL, "probe": "Linear(128,32)->GELU->Linear(32,1)", "architecture": mlp_training["head"].architecture_audit(),
                  "metrics": mlp_metrics, "training_curve": mlp_training["rows"], "diagnostic_only": True, "formal_checkpoint_written": False}
    global_rows = [{"synthetic_interaction": LABEL, "model": name, **metrics} for name, metrics in (("Linear", linear_metrics), ("MinimalMLP", mlp_metrics))]
    summary = {"label": LABEL, "stage": STAGE, "diagnostic_only": True, "test_reads": 0, "linear_reproduction": linear_record,
               "minimal_mlp": mlp_record, "subtype_linear_probes": subtype_probes, "semantic_separation": semantics,
               "shortcuts": shortcuts, "gates": gates, "root_cause": root_cause,
               "ready_for_phase5b17eb": gates["all_passed"], "phase5b17eb_started": False,
               "formal_harm_checkpoint_written": False, "threshold_calibration_performed": False, "figures": figures}

    io.write_json(args.output_dir / "frozen_contract.json", frozen); io.write_json(args.output_dir / "embedding_cache_contract.json", cache_contract)
    io.write_json(args.output_dir / "linear_reproduction.json", linear_record); io.write_json(args.output_dir / "minimal_mlp_probe.json", mlp_record)
    io.write_csv(args.output_dir / "subtype_linear_probes.csv", subtype_probes); io.write_csv(args.output_dir / "global_probe_comparison.csv", global_rows)
    io.write_csv(args.output_dir / "safe_beneficial_comparison.csv", safe); io.write_csv(args.output_dir / "benefit_risk_tradeoff_comparison.csv", tradeoff)
    io.write_csv(args.output_dir / "by_subtype.csv", subtype_rows); io.write_csv(args.output_dir / "by_context.csv", context_rows)
    io.write_csv(args.output_dir / "by_motion.csv", motion_rows); io.write_csv(args.output_dir / "by_action.csv", action_rows)
    io.write_csv(args.output_dir / "by_profile.csv", profile_rows); io.write_csv(args.output_dir / "stop_audit.csv", stop)
    io.write_json(args.output_dir / "embedding_geometry.json", geometry_audit); io.write_json(args.output_dir / "root_cause_classification.json", root_cause)
    io.write_json(args.output_dir / "gate_results.json", gates); io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(io.clean(summary), indent=2), flush=True)


if __name__ == "__main__": main()

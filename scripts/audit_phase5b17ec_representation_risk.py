"""Phase 5B-1.7E-C frozen representation risk-information audit.

Only synthetic TRAIN/VALIDATION samples are accessed. TEST stays sealed. Every
probe is a diagnostic-only linear readout and no model checkpoint is written.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import audit_phase5b17ea_harm_readout_capacity as ea
from scripts import run_phase5a_frozen3b as io
from scripts import run_phase5b17d_manifest_v2_rebaseline as d
from scripts import run_phase5b17e_independent_harm_v2 as e
from src.data.adverse_response_dataset import GENERATOR_SEED, RISK_SEED, build_development_split
from src.evaluation.probabilistic_harm import harm_metrics
from src.evaluation.representation_risk_audit import (SUBTYPE_ORDER, candidate_conditioning_distances,
    cosine_similarity_rows, fixed_noisy_or, group_geometry, pairwise_discrimination)
from src.multimodal.phase5b_v2_dataset import build_v2_temporal_samples
from src.multimodal.temporal_schema import LABEL

STAGE = "Phase 5B-1.7E-C Representation Risk Information Audit"
EXPECTED_CHECKPOINT_SHA256 = ea.EXPECTED_CHECKPOINT_SHA256
EXPECTED_NORMALIZER_SHA256 = "dc4e412b5313d5b8d96b7ad6521b03e0a7672419c5ba50076ecf002740011d2c"
SUBTYPE_PREDICATES = {
    "GT_UNSAFE": lambda sample: sample.targets.gt_unsafe,
    "EXCESSIVE_DECELERATION": lambda sample: sample.split_metadata["excessive_deceleration_evaluation_only"],
    "ABRUPT_LATERAL_RESPONSE": lambda sample: sample.split_metadata["abrupt_lateral_response_evaluation_only"],
    "ABRUPT_HEADING_CHANGE": lambda sample: sample.split_metadata["abrupt_heading_change_evaluation_only"],
}


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
    parser.add_argument("--phase5b17ea-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17ea_harm_readout_capacity_audit")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17ec_representation_risk_audit")
    return parser.parse_args()


def stage_registry(dimensions):
    rows = [
        ("R0_FINAL_FUSED", True, "model fusion output; current harm-head input"),
        ("R1_HISTORY_CONTEXT_PREFUSION", True, "existing projected history-stream pools plus projected scene; candidate-free audit concatenation"),
        ("R2_CANDIDATE_PREFUSION", True, "existing candidate-future projection pool plus action projection"),
        ("R3_SKELETON_PROJECTED_POOL", True, "existing skeleton projection, masked temporal mean"),
        ("R4_MOTION_PROJECTED_POOL", True, "existing human-motion projection, masked temporal mean"),
        ("R5_FUNCTIONAL_INTERACTION_DIAGNOSTIC", True, "existing functional, interaction and world-model diagnostic projection pools"),
        ("R6_PREFINAL_FUSION_CONCAT", True, "actual input to frozen final fusion: joint temporal pool, action, scene"),
        ("R6_JOINT_TEMPORAL_POOL", True, "masked mean of joint history plus candidate-future Transformer tokens"),
        ("AUDIT_HUMAN_CANDIDATE_CONCAT", True, "fixed concat of R1 and R2; no learned audit fusion"),
        ("POST_TRANSFORMER_HUMAN_BEFORE_CANDIDATE_FUSION", False, "NOT AVAILABLE: candidate future and history tokens share the frozen Transformer before pooling"),
    ]
    return [{"label": LABEL, "representation_stage": name, "available": available,
             "embedding_dimension": dimensions.get(name), "provenance": provenance,
             "runtime_valid_inputs_only": True, "GT_future_used": False} for name, available, provenance in rows]


def extract_representations(model, samples, normalizers, batch_size, torch, device):
    chunks = defaultdict(list); model.eval()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = e.b1.temporal_batch(samples[start:start + batch_size], normalizers, torch, device)
            normal = model.encode(batch); audit = model.audit_representations(batch)
            if not torch.equal(normal, audit["R0_FINAL_FUSED"]):
                raise RuntimeError("audit extraction changed normal encode output")
            audit["AUDIT_HUMAN_CANDIDATE_CONCAT"] = torch.cat((audit["R1_HISTORY_CONTEXT_PREFUSION"], audit["R2_CANDIDATE_PREFUSION"]), -1)
            for name, value in audit.items(): chunks[name].append(value.detach().cpu())
    return {name: torch.cat(values) for name, values in chunks.items()}


def metadata(samples):
    return {
        "candidate_id": np.asarray([sample.sample_id for sample in samples]),
        "episode_id": np.asarray([sample.episode_id for sample in samples]),
        "split": np.asarray([sample.split for sample in samples]),
        "harm_v2": np.asarray([e.harm_v2_target(sample) for sample in samples], bool),
        **{name: np.asarray([predicate(sample) for sample in samples], bool) for name, predicate in SUBTYPE_PREDICATES.items()},
        "safe_beneficial": np.asarray([sample.split_metadata["safe_beneficial_evaluation_only"] for sample in samples], bool),
        "benefit_risk_tradeoff": np.asarray([sample.split_metadata["benefit_risk_tradeoff_evaluation_only"] for sample in samples], bool),
        "action": np.asarray([sample.split_metadata["candidate_action_id_audit"] for sample in samples], int),
        "motion": np.asarray([sample.split_metadata["motion_type_evaluation_only"] for sample in samples]),
        "context": np.asarray(["|".join(map(str, sample.split_metadata["contexts_evaluation_only"])) for sample in samples]),
        "profile_audit_only": np.asarray([sample.split_metadata["person_profile_id"] for sample in samples], int),
    }


def save_cache(path, representations, samples_by_split):
    payload = {}; dimensions = {}
    for split, stages in representations.items():
        for stage, value in stages.items():
            payload[f"{split}__{stage}"] = value.numpy(); dimensions[stage] = int(value.shape[1])
        for name, value in metadata(samples_by_split[split]).items(): payload[f"{split}__meta__{name}"] = value
    np.savez_compressed(path, **payload)
    return {"label": LABEL, "cache_file": str(path), "cache_sha256": e.file_sha(path), "splits": ["train", "validation"],
            "representation_dimensions": dimensions, "train_candidates": len(samples_by_split["train"]),
            "validation_candidates": len(samples_by_split["validation"]), "test_embeddings": 0, "test_reads": 0,
            "profile_id_in_probe_input": False, "GT_future_in_representation": False,
            "metadata_is_audit_only": sorted(metadata(samples_by_split["train"]))}


def train_linear(stage, label_name, train_x, validation_selection_x, validation_full_x,
                 train_target, validation_target, args, torch, device):
    class LinearProbe(torch.nn.Module):
        def __init__(self, width): super().__init__(); self.linear = torch.nn.Linear(width, 1)
        def forward(self, value): return self.linear(value).squeeze(-1)
    torch.manual_seed(args.seed)
    trained = e.train_head(LinearProbe(train_x.shape[1]).to(device), train_x, torch.tensor(train_target, dtype=torch.float32),
                           validation_selection_x, torch.tensor(validation_target, dtype=torch.float32), args, torch, device)
    probability = e.probabilities(trained["head"], validation_selection_x, args.batch_size, torch, device)
    full_probability = e.probabilities(trained["head"], validation_full_x, args.batch_size, torch, device)
    metrics = harm_metrics(probability, validation_target)
    row = {"synthetic_interaction": LABEL, "diagnostic_only": True, "representation_stage": stage,
           "target": label_name, "embedding_dimension": int(train_x.shape[1]), **metrics,
           "selected_epoch": int(trained["selected"]["epoch"]), "parameter_count": int(train_x.shape[1] + 1),
           "train_split_only_for_updates": True, "validation_updates": False, "formal_checkpoint_allowed": False}
    weight = trained["head"].linear.weight.detach().cpu().numpy()[0]
    return row, probability, full_probability, weight


def run_probes(representations, samples_by_split, args, torch, device):
    train_meta, validation_meta = metadata(samples_by_split["train"]), metadata(samples_by_split["validation"])
    global_rows, subtype_rows, probabilities, weights = [], [], {}, {}
    for stage in representations["train"]:
        train_x, validation_x = representations["train"][stage], representations["validation"][stage]
        row, probability, full_probability, weight = train_linear(stage, "HARM_V2_UNION", train_x, validation_x, validation_x,
            train_meta["harm_v2"], validation_meta["harm_v2"], args, torch, device)
        global_rows.append(row); probabilities[(stage, "HARM_V2_UNION")] = full_probability; weights[(stage, "HARM_V2_UNION")] = weight
        for subtype in SUBTYPE_ORDER:
            train_keep = train_meta[subtype] | ~train_meta["harm_v2"]
            validation_keep = validation_meta[subtype] | ~validation_meta["harm_v2"]
            row, _, full_probability, weight = train_linear(stage, subtype, train_x[train_keep], validation_x[validation_keep], validation_x,
                train_meta[subtype][train_keep], validation_meta[subtype][validation_keep], args, torch, device)
            row.update({"train_positive_count": int(train_meta[subtype].sum()), "train_negative_count": int((~train_meta["harm_v2"]).sum()),
                        "validation_positive_count": int(validation_meta[subtype].sum()), "validation_negative_count": int((~validation_meta["harm_v2"]).sum())})
            subtype_rows.append(row); weights[(stage, subtype)] = weight
            probabilities[(stage, subtype)] = full_probability
    return global_rows, subtype_rows, probabilities, weights


def subtype_combinations(samples_by_split):
    rows = []
    for split, samples in samples_by_split.items():
        values = metadata(samples); counts = Counter(tuple(bool(values[name][i]) for name in SUBTYPE_ORDER) for i in range(len(samples)))
        for bits in np.ndindex(*(2,) * 4):
            names = [name for name, active in zip(SUBTYPE_ORDER, bits) if active]
            rows.append({"synthetic_interaction": LABEL, "split": split, "combination": "+".join(names) if names else "NONE",
                         "positive_multiplicity": int(sum(bits)), "candidate_count": counts[tuple(map(bool, bits))]})
    return rows


def describe(values):
    values = np.asarray(values, np.float64)
    return {"count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "P10": float(np.percentile(values, 10)), "P90": float(np.percentile(values, 90)), "P95": float(np.percentile(values, 95)),
            "FP_at_0_5_diagnostic": float(np.mean(values >= .5))}


def false_negative_rows(samples, score, model="GlobalLinear_R0"):
    meta = metadata(samples); fn = meta["harm_v2"] & (score < .5); rows = []
    definitions = {**{f"subtype:{name}": meta[name] for name in SUBTYPE_ORDER},
                   "benefit_risk_tradeoff": meta["benefit_risk_tradeoff"], "motion:stop": meta["motion"] == "stop"}
    for value in np.unique(meta["action"]): definitions[f"action:{value}"] = meta["action"] == value
    for value in np.unique(meta["motion"]): definitions[f"motion:{value}"] = meta["motion"] == value
    for value in ("C7", "C8", "C9"): definitions[f"context:{value}"] = np.asarray([value in item for item in meta["context"]])
    for value in np.unique(meta["profile_audit_only"]): definitions[f"profile_audit_only:{value}"] = meta["profile_audit_only"] == value
    combo = np.asarray(["+".join(name for name in SUBTYPE_ORDER if meta[name][i]) for i in range(len(samples))])
    for value in np.unique(combo[meta["harm_v2"]]): definitions[f"subtype_combination:{value}"] = combo == value
    for group, mask in definitions.items():
        positive = meta["harm_v2"] & mask
        rows.append({"synthetic_interaction": LABEL, "model": model, "available": True, "group": group,
                     "harm_positive_count": int(positive.sum()), "false_negative_count_at_0_5_diagnostic": int((fn & mask).sum()),
                     "false_negative_rate_at_0_5_diagnostic": float((fn & mask).sum() / max(positive.sum(), 1))})
    rows.append({"synthetic_interaction": LABEL, "model": "MinimalMLP_1.7E-A", "available": False,
                 "group": "ALL", "reason": "per-candidate scores were not frozen; Phase1.7E-C forbids MLP retraining"})
    return rows


def make_figures(output, global_rows, subtype_rows, final_embedding, validation_meta):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    stages = [row["representation_stage"] for row in global_rows]; values = [row["AUROC"] for row in global_rows]
    plt.figure(figsize=(10, 5)); plt.bar(range(len(stages)), values); plt.xticks(range(len(stages)), stages, rotation=70, ha="right", fontsize=7); plt.ylim(.4, 1); plt.ylabel("Validation AUROC"); plt.title(f"{LABEL}\nStage-wise global harm-v2 linear probes")
    path = folder / "stagewise_global_auroc.png"; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path))
    x = final_embedding.numpy().astype(np.float64); centered = x - x.mean(0); _, _, vt = np.linalg.svd(centered, full_matrices=False); pca = centered @ vt[:2].T
    target = validation_meta["harm_v2"]; plt.figure(figsize=(7, 5)); plt.scatter(pca[~target, 0], pca[~target, 1], s=6, alpha=.3, label="negative"); plt.scatter(pca[target, 0], pca[target, 1], s=7, alpha=.45, label="harm-v2"); plt.legend(); plt.title(f"{LABEL}\nR0 PCA diagnostic only")
    path = folder / "r0_pca.png"; plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(); paths.append(str(path)); return paths


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite Phase5B-1.7E-C: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True); random.seed(args.seed); np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    checkpoint_sha = e.file_sha(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256: raise RuntimeError(f"source checkpoint SHA256 mismatch: {checkpoint_sha}")
    model, payload = e.load_frozen(args.checkpoint, torch, device); before = d.model_sha(model); state_before = model.state_dict()
    backbone_before = e.state_sha(state_before, exclude=("benefit.", "uncertainty.", "harm.")); benefit_before = e.state_sha(state_before, prefixes=("benefit.", "uncertainty.")); ranking_before = e.state_sha(state_before, exclude=("harm.",))
    samples = {"train": build_v2_temporal_samples(build_development_split("train", 240, GENERATOR_SEED, RISK_SEED)),
               "validation": build_v2_temporal_samples(build_development_split("validation", 240, GENERATOR_SEED + 1000, RISK_SEED + 1000))}
    contract = d.manifest_contract(args.manifest, samples["train"] + samples["validation"]); normalizers = payload["normalizer"]
    if normalizers["sha256"] != EXPECTED_NORMALIZER_SHA256: raise RuntimeError("normalizer checksum mismatch")
    representations = {split: extract_representations(model, values, normalizers, args.batch_size, torch, device) for split, values in samples.items()}
    cache_contract = save_cache(args.output_dir / "representation_embeddings.npz", representations, samples)
    registry = stage_registry(cache_contract["representation_dimensions"])

    # All learned diagnostics below are fixed linear probes trained on TRAIN only.
    global_rows, subtype_rows, probabilities, weights = run_probes(representations, samples, args, torch, device)
    validation_meta = metadata(samples["validation"])
    subtype_matrix = np.column_stack([probabilities[("R0_FINAL_FUSED", name)] for name in SUBTYPE_ORDER])
    noisy_or = fixed_noisy_or(subtype_matrix); noisy_metrics = harm_metrics(noisy_or, validation_meta["harm_v2"])
    global_r0 = probabilities[("R0_FINAL_FUSED", "HARM_V2_UNION")]
    safe = validation_meta["safe_beneficial"]
    noisy_record = {"label": LABEL, "diagnostic_only": True, "formula": "1-prod_k(1-p_k)", "learned_combination_weights": False,
                    "subtype_order": list(SUBTYPE_ORDER), **noisy_metrics, "safe_beneficial": describe(noisy_or[safe]),
                    "global_linear_R0_AUROC": harm_metrics(global_r0, validation_meta["harm_v2"])["AUROC"]}

    cooccurrence = subtype_combinations(samples)
    directions = [{"synthetic_interaction": LABEL, "representation_stage": "R0_FINAL_FUSED", **row} for row in cosine_similarity_rows({name: weights[("R0_FINAL_FUSED", name)] for name in SUBTYPE_ORDER})]
    geometry_masks = {"safe_beneficial": safe, "pure_negative": ~validation_meta["harm_v2"], **{name: validation_meta[name] for name in SUBTYPE_ORDER}}
    geometry = {"label": LABEL, "split": "validation", "representation_stage": "R0_FINAL_FUSED", **group_geometry(representations["validation"]["R0_FINAL_FUSED"], geometry_masks)}

    tradeoff = validation_meta["benefit_risk_tradeoff"]; x = representations["validation"]["R0_FINAL_FUSED"].numpy(); safe_centroid = x[safe].mean(0); harm_centroid = x[validation_meta["harm_v2"] & ~tradeoff].mean(0)
    benefit_risk_rows = [{"synthetic_interaction": LABEL, "candidate_id": samples["validation"][index].sample_id,
                          "distance_to_safe_beneficial_centroid": float(np.linalg.norm(x[index] - safe_centroid)),
                          "distance_to_other_harm_positive_centroid": float(np.linalg.norm(x[index] - harm_centroid)),
                          "closer_to_safe_beneficial": bool(np.linalg.norm(x[index] - safe_centroid) < np.linalg.norm(x[index] - harm_centroid))} for index in np.flatnonzero(tradeoff)]

    safe_rows = []
    for index in np.flatnonzero(safe):
        safe_rows.append({"synthetic_interaction": LABEL, "candidate_id": samples["validation"][index].sample_id,
                          "global_linear": float(global_r0[index]), **{f"p_{name}": float(probabilities[("R0_FINAL_FUSED", name)][index]) for name in SUBTYPE_ORDER},
                          "fixed_noisy_or": float(noisy_or[index])})
    stop = validation_meta["motion"] == "stop"; stop_rows = []
    for scoring, score in (("GlobalLinear_R0", global_r0), ("FixedNoisyOR", noisy_or)):
        row = {"synthetic_interaction": LABEL, "scoring": scoring, "true_harm_rate": float(validation_meta["harm_v2"][stop].mean()), **harm_metrics(score[stop], validation_meta["harm_v2"][stop])}
        row.update({f"{name}_positive_count": int((validation_meta[name] & stop).sum()) for name in SUBTYPE_ORDER})
        row.update({"safe_beneficial_count": int((safe & stop).sum()),
                    "mean_distance_to_safe_beneficial_centroid": float(np.mean(np.linalg.norm(x[stop] - safe_centroid, axis=1))),
                    "mean_distance_to_other_harm_positive_centroid": float(np.mean(np.linalg.norm(x[stop] - harm_centroid, axis=1)))})
        stop_rows.append(row)

    conditioning_rows = []
    for stage, embedding in representations["validation"].items():
        conditioning_rows.append({"synthetic_interaction": LABEL, "representation_stage": stage, **candidate_conditioning_distances(embedding, validation_meta["harm_v2"], validation_meta["episode_id"])})
    discrimination_rows = []
    for scoring, score, label in [("GlobalLinear_R0", global_r0, validation_meta["harm_v2"]), ("FixedNoisyOR", noisy_or, validation_meta["harm_v2"])] + [(f"Subtype:{name}", probabilities[("R0_FINAL_FUSED", name)], validation_meta[name]) for name in SUBTYPE_ORDER]:
        discrimination_rows.append({"synthetic_interaction": LABEL, "scoring": scoring, **pairwise_discrimination(score, label, validation_meta["episode_id"])})

    stage_global = {row["representation_stage"]: row for row in global_rows}; stage_subtype = {(row["representation_stage"], row["target"]): row for row in subtype_rows}
    human_candidate = [row for row in global_rows + subtype_rows if row["representation_stage"] in ("R1_HISTORY_CONTEXT_PREFUSION", "R2_CANDIDATE_PREFUSION", "R0_FINAL_FUSED", "R6_JOINT_TEMPORAL_POOL")]
    prefusion = [row for row in global_rows + subtype_rows if row["representation_stage"] in ("AUDIT_HUMAN_CANDIDATE_CONCAT", "R6_PREFINAL_FUSION_CONCAT", "R0_FINAL_FUSED")]
    gate_a_checks = {name: max(row["AUROC"] for row in subtype_rows if row["target"] == name) >= .80 for name in SUBTYPE_ORDER}
    gate_b_checks = {"Noisy_OR_AUROC_at_least_0_80": noisy_metrics["AUROC"] >= .80, "Noisy_OR_exceeds_global_linear_R0": noisy_metrics["AUROC"] > stage_global["R0_FINAL_FUSED"]["AUROC"]}
    global_safe = describe(global_r0[safe]); noisy_safe = describe(noisy_or[safe])
    gate_c_checks = {"safe_mean_not_higher_than_global_linear": noisy_safe["mean"] <= global_safe["mean"],
                     "safe_P90_not_higher_than_global_linear": noisy_safe["P90"] <= global_safe["P90"],
                     "safe_P95_not_higher_than_global_linear": noisy_safe["P95"] <= global_safe["P95"],
                     "safe_FP_at_0_5_not_higher_than_global_linear_diagnostic": noisy_safe["FP_at_0_5_diagnostic"] <= global_safe["FP_at_0_5_diagnostic"]}
    gates = {"Gate_A": {"name": "Representation Availability", "checks": gate_a_checks, "passed": all(gate_a_checks.values())},
             "Gate_B": {"name": "Factorization Evidence", "checks": gate_b_checks, "passed": all(gate_b_checks.values())},
             "Gate_C": {"name": "Safe-Beneficial Protection", "checks": gate_c_checks, "global_linear_safe": global_safe, "noisy_or_safe": noisy_safe, "passed": all(gate_c_checks.values())}}
    gates["all_passed"] = all(gate["passed"] for gate in gates.values())

    concat_gain = stage_global["AUDIT_HUMAN_CANDIDATE_CONCAT"]["AUROC"] - stage_global["R0_FINAL_FUSED"]["AUROC"]
    human_gap = stage_global["R0_FINAL_FUSED"]["AUROC"] - stage_global["R1_HISTORY_CONTEXT_PREFUSION"]["AUROC"]
    overlap_fraction = float(np.mean([row["closer_to_safe_beneficial"] for row in benefit_risk_rows]))
    causes = {"factorization": gates["Gate_B"]["passed"], "fusion_loss": concat_gain >= .02,
              "candidate_conditioning_weakness": human_gap <= .01, "benefit_risk_overlap": overlap_fraction >= .5,
              "subtype_information_weakness": not gates["Gate_A"]["passed"]}
    active = [name for name, value in causes.items() if value]
    if active == ["factorization"]: root_class, root_name, recommendation = "C", "SEMANTIC FACTORIZATION BOTTLENECK", "Factorized Harm-v2 Modeling"
    elif active == ["fusion_loss"]: root_class, root_name, recommendation = "A", "FINAL FUSION RISK INFORMATION LOSS", "Minimal Risk-Preserving Fusion Intervention"
    elif active == ["candidate_conditioning_weakness"]: root_class, root_name, recommendation = "B", "CANDIDATE-CONDITIONING WEAKNESS", "Candidate-Conditioned Risk Fusion Repair"
    elif active == ["benefit_risk_overlap"]: root_class, root_name, recommendation = "D", "BENEFIT-RISK REPRESENTATION OVERLAP", "Risk-specific shared representation intervention"
    elif active == ["subtype_information_weakness"]: root_class, root_name, recommendation = "E", "SUBTYPE-SPECIFIC INFORMATION WEAKNESS", "Subtype representation diagnosis"
    else: root_class, root_name, recommendation = "F", "MULTIPLE INTERACTING CAUSES", "Representation Risk Information Audit follow-up; no model intervention is authorized"
    root = {"label": LABEL, "selected_class": root_class, "selected_name": root_name, "evidence_flags": causes,
            "active_evidence": active, "recommended_single_variable_intervention": recommendation, "intervention_implemented": False}

    fn_rows = false_negative_rows(samples["validation"], global_r0)
    figures = make_figures(args.output_dir, global_rows, subtype_rows, representations["validation"]["R0_FINAL_FUSED"], validation_meta)
    after = d.model_sha(model); state_after = model.state_dict()
    frozen = {"label": LABEL, "source_checkpoint_sha256": checkpoint_sha, "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
              "manifest_sha256": e.file_sha(args.manifest), "expected_manifest_sha256": d.EXPECTED_MANIFEST_SHA,
              "normalizer_sha256": normalizers["sha256"], "expected_normalizer_sha256": EXPECTED_NORMALIZER_SHA256,
              "all_model_parameters_require_grad_false": not any(parameter.requires_grad for parameter in model.parameters()), "test_reads": 0,
              "full_model_checksum_before": before, "full_model_checksum_after": after, "full_model_unchanged": before == after,
              "temporal_backbone_checksum_unchanged": backbone_before == e.state_sha(state_after, exclude=("benefit.", "uncertainty.", "harm.")),
              "benefit_head_checksum_unchanged": benefit_before == e.state_sha(state_after, prefixes=("benefit.", "uncertainty.")),
              "ranking_behavior_checksum_unchanged": ranking_before == e.state_sha(state_after, exclude=("harm.",)),
              "optimizer_created_for_backbone": False, "formal_harm_checkpoint_written": False, "manifest_contract_passed": contract["passed"]}
    summary = {"label": LABEL, "stage": STAGE, "diagnostic_only": True, "test_reads": 0,
               "best_global_stage": max(global_rows, key=lambda row: row["AUROC"]),
               "best_subtype_stage": {name: max((row for row in subtype_rows if row["target"] == name), key=lambda row: row["AUROC"]) for name in SUBTYPE_ORDER},
               "factorized_noisy_or": noisy_record, "gates": gates, "root_cause": root,
               "benefit_risk_tradeoff_count": int(tradeoff.sum()), "benefit_risk_closer_to_safe_fraction": overlap_fraction,
               "MLP_false_negative_detail": "NOT AVAILABLE: not persisted by frozen 1.7E-A; retraining forbidden in 1.7E-C",
               "formal_checkpoint_written": False, "threshold_calibration_performed": False, "figures": figures}

    io.write_json(args.output_dir / "frozen_contract.json", frozen); io.write_json(args.output_dir / "representation_stage_registry.json", registry)
    io.write_json(args.output_dir / "embedding_cache_contract.json", cache_contract); io.write_csv(args.output_dir / "stagewise_global_probes.csv", global_rows)
    io.write_csv(args.output_dir / "stagewise_subtype_probes.csv", subtype_rows); io.write_csv(args.output_dir / "human_vs_candidate_probe.csv", human_candidate)
    io.write_csv(args.output_dir / "prefusion_vs_final.csv", prefusion); io.write_json(args.output_dir / "factorized_noisy_or.json", noisy_record)
    io.write_csv(args.output_dir / "subtype_cooccurrence.csv", cooccurrence); io.write_csv(args.output_dir / "subtype_direction_cosine.csv", directions)
    io.write_json(args.output_dir / "representation_geometry.json", geometry); io.write_csv(args.output_dir / "global_false_negative_audit.csv", fn_rows)
    io.write_csv(args.output_dir / "benefit_risk_geometry.csv", benefit_risk_rows); io.write_csv(args.output_dir / "safe_beneficial_scores.csv", safe_rows)
    io.write_csv(args.output_dir / "stop_risk_audit.csv", stop_rows); io.write_csv(args.output_dir / "candidate_conditioning_audit.csv", conditioning_rows)
    io.write_csv(args.output_dir / "within_episode_risk_discrimination.csv", discrimination_rows); io.write_json(args.output_dir / "root_cause_classification.json", root)
    io.write_json(args.output_dir / "gate_results.json", gates); io.write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()

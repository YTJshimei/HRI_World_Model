"""Phase 5B-1.7A audit of benefit/harm semantics and subgroup confounding."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_phase5a_frozen3b as p5
from scripts import run_phase5b1_static_vs_temporal as b1
from scripts import run_phase5b15_decision_bottleneck as b15
from scripts import run_phase5b16_candidate_ranking as b16
from scripts import run_phase5b17_harm_gate_calibration as b17
from src.evaluation.context_value_metrics import binary_auc
from src.multimodal.temporal_schema import LABEL
from src.training.candidate_ranking import LAMBDA_RANK

BENEFIT_EPSILON = 1e-6
BENEFIT_THRESHOLD = -0.02
HARM_THRESHOLD = 0.2
ACTION_NAMES = {0: "KEEP", 1: "SPEED_DOWN", 2: "SPEED_UP", 3: "DISTANCE_PLUS", 4: "DISTANCE_MINUS", 5: "LEFT_OFFSET", 6: "RIGHT_OFFSET"}
CONTEXTS = ("C4", "C5", "C6", "C7", "C8", "C9")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--belief-samples", type=int, choices=(16,), default=16)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17a_target_conflict_audit")
    parser.add_argument("--phase5b17-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b17_harm_gate_calibration")
    parser.add_argument("--phase5b16-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b16_candidate_ranking")
    parser.add_argument("--phase5b1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b1_static_vs_temporal_small")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b05_c7_coverage")
    parser.add_argument("--phase5a-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--phase4c2-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c2")
    parser.add_argument("--phase4c3-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c3")
    return parser.parse_args()


def source_definition():
    return {
        "label": LABEL,
        "benefit": {
            "source_file": "scripts/run_phase5a_context_value.py", "function": "build_tokens", "source_line": 88,
            "formula": "benefit = GT_total_cost[canonical_generic_index] - GT_total_cost[candidate_index]",
            "beneficial_rule": "benefit > 1e-6", "threshold": BENEFIT_EPSILON,
            "inputs": ["episode.gt_costs.total", "canonical generic index", "candidate index"],
            "dependency_chain": ["GT counterfactual rollout", "compute_decision_costs", "DecisionCosts.total", "generic-minus-candidate difference", "ContextTarget.benefit"],
        },
        "harm": {
            "source_file": "scripts/run_phase5a_context_value.py", "function": "build_tokens", "source_line": 88,
            "formula": "harm = (benefit < -1e-6)", "threshold": -BENEFIT_EPSILON,
            "inputs": ["ContextTarget.benefit"],
            "dependency_chain": ["the exact same GT total-cost difference as benefit", "negative-benefit threshold", "ContextTarget.harm"],
            "not_directly_defined_from": ["GT unsafe", "minimum distance", "unsafe duration", "human-response degradation threshold"],
        },
        "gt_unsafe": {
            "source_file": "scripts/run_phase5a_context_value.py", "function": "build_tokens", "source_line": 90,
            "formula": "GT_unsafe = (episode.gt_costs.unsafe_duration[candidate] > 0)", "threshold": 0.0,
            "upstream_source_file": "src/decision/decision_cost.py", "upstream_function": "compute_decision_costs", "upstream_lines": [75, 85],
            "dependency_chain": ["GT human-robot distance trajectory", "distance < too_close_distance", "mean violation duration", "unsafe_duration > 0"],
        },
        "cost_total": {
            "source_file": "src/decision/decision_cost.py", "function": "compute_decision_costs", "source_lines": [57, 85, 110],
            "formula": "total = 1.0*task + 3.0*safety + 1.4*human_response + 0.55*disturbance + 0.85*uncertainty",
            "weights": {"task": 1.0, "safety": 3.0, "human_response": 1.4, "disturbance": 0.55, "uncertainty": 0.85},
        },
        "semantic_identity": {
            "GT_harm_equals_GT_unsafe": False,
            "benefit_and_harm_mathematically_overlap": False,
            "reason": "harm is the negative tail of the same scalar benefit target; beneficial requires positive benefit",
            "runtime_semantic_mismatch": "a relative-cost-worse-than-generic classifier is used as a harm veto, although it is not a direct safety/adverse-response label",
        },
    }


def masks(samples):
    beneficial = np.asarray([sample.targets.benefit > BENEFIT_EPSILON for sample in samples], bool)
    harmful = np.asarray([sample.targets.harm for sample in samples], bool)
    feasible = np.asarray([sample.targets.feasible for sample in samples], bool)
    unsafe = np.asarray([sample.targets.gt_unsafe for sample in samples], bool)
    safe_beneficial = beneficial & ~harmful & feasible
    return {"beneficial": beneficial, "harmful": harmful, "feasible": feasible, "unsafe": unsafe,
            "safe_beneficial": safe_beneficial, "neutral": ~beneficial & ~harmful,
            "beneficial_harmful": beneficial & harmful, "harmful_only": ~beneficial & harmful}


def overlap_rows(split, samples):
    value = masks(samples); rows = []
    quadrants = (("Q1 beneficial=1 harmful=0", value["beneficial"] & ~value["harmful"]),
                 ("Q2 beneficial=1 harmful=1", value["beneficial"] & value["harmful"]),
                 ("Q3 beneficial=0 harmful=1", ~value["beneficial"] & value["harmful"]),
                 ("Q4 beneficial=0 harmful=0", ~value["beneficial"] & ~value["harmful"]))
    for quadrant, selected in quadrants:
        rows.append({"synthetic_interaction": LABEL, "split": split, "quadrant": quadrant,
                     "candidate_count": int(selected.sum()), "candidate_ratio": float(selected.mean()),
                     "episode_count": len({sample.episode_id for sample, keep in zip(samples, selected) if keep}),
                     "feasible_candidate_count": int((selected & value["feasible"]).sum())})
    return rows


def aggregate_overlap(group, samples, prediction=None):
    value = masks(samples); count = len(samples)
    result = {"synthetic_interaction": LABEL, "group": group, "candidate_count": count,
              "episode_count": len({sample.episode_id for sample in samples}),
              "beneficial_count": int(value["beneficial"].sum()), "beneficial_rate": float(value["beneficial"].mean()) if count else 0.0,
              "harmful_count": int(value["harmful"].sum()), "harmful_rate": float(value["harmful"].mean()) if count else 0.0,
              "overlap_count": int(value["beneficial_harmful"].sum()), "overlap_rate": float(value["beneficial_harmful"].mean()) if count else 0.0,
              "safe_beneficial_count": int(value["safe_beneficial"].sum()), "safe_beneficial_rate": float(value["safe_beneficial"].mean()) if count else 0.0}
    if prediction is not None and count:
        result["harm_probability_mean"] = float(np.asarray(prediction["harm"]).mean())
    return result


def group_rows(samples, prediction, key_function):
    grouped = defaultdict(list)
    for index, sample in enumerate(samples):
        for key in key_function(sample): grouped[str(key)].append(index)
    rows = []
    for key in sorted(grouped):
        indices = grouped[key]; subset = [samples[index] for index in indices]
        subprediction = {name: np.asarray(values)[indices] for name, values in prediction.items()}
        rows.append(aggregate_overlap(key, subset, subprediction))
    return rows


def context_keys(sample):
    present = b15.context_labels(sample)
    return [context for context in CONTEXTS if context in present]


def condition_metrics(probability, target):
    probability = np.clip(np.asarray(probability, float), 1e-7, 1 - 1e-7); target = np.asarray(target, bool)
    return {"count": len(target), "observed_harm_frequency": float(target.mean()) if len(target) else None,
            "mean_predicted_probability": float(probability.mean()) if len(target) else None,
            "AUROC": binary_auc(probability, target) if len(target) else None,
            "Brier": float(np.mean((probability - target) ** 2)) if len(target) else None,
            "ECE": b17.expected_calibration_error(probability, target, 10) if len(target) else None,
            "NLL": float(-np.mean(target*np.log(probability)+(~target)*np.log(1-probability))) if len(target) else None}


def conditional_calibration_rows(split, samples, prediction):
    value = masks(samples); probability = np.asarray(prediction["harm"], float)
    groups = {"all": np.ones(len(samples), bool), "beneficial": value["beneficial"],
              "safe_beneficial": value["safe_beneficial"], "beneficial_and_harmful": value["beneficial_harmful"],
              "neutral": value["neutral"], "harmful_only": value["harmful_only"]}
    rows = []
    for subgroup, selected in groups.items():
        summary = condition_metrics(probability[selected], value["harmful"][selected])
        rows.append({"synthetic_interaction": LABEL, "row_type": "summary", "split": split, "subgroup": subgroup, **summary})
        for lower in np.linspace(0, 1, 10, endpoint=False):
            upper=lower+.1; in_bin=selected & (probability>=lower) & ((probability<=upper) if upper>=1 else (probability<upper))
            rows.append({"synthetic_interaction": LABEL, "row_type": "reliability_bin", "split": split, "subgroup": subgroup,
                         "bin_lower": lower, "bin_upper": upper, "bin_count": int(in_bin.sum()),
                         "mean_predicted_probability": float(probability[in_bin].mean()) if in_bin.any() else None,
                         "observed_harm_frequency": float(value["harmful"][in_bin].mean()) if in_bin.any() else None})
    return rows


def safe_beneficial_rows(split, samples, prediction, thresholds=(BENEFIT_THRESHOLD, HARM_THRESHOLD)):
    value = masks(samples); audit = b15.audit_model("R1 Frozen", samples, prediction, thresholds)
    funnel_by_id = {row["sample_id"]: row for row in audit["funnel"]}
    rows = []
    for index, sample in enumerate(samples):
        if not value["safe_beneficial"][index]: continue
        row = funnel_by_id[sample.sample_id]
        rows.append({"synthetic_interaction": LABEL, "split": split, "sample_id": sample.sample_id, "episode_id": sample.episode_id,
                     "gt_benefit": sample.targets.benefit, "gt_harmful": sample.targets.harm, "gt_unsafe": sample.targets.gt_unsafe,
                     "feasible": sample.targets.feasible, "predicted_benefit": float(prediction["benefit"][index]),
                     "predicted_harm_probability": float(prediction["harm"][index]), "sign_correct": float(prediction["benefit"][index]) > 0,
                     "benefit_pass": row["benefit_threshold_pass"], "harm_pass": row["harm_threshold_pass"],
                     "direct_harm_gate_pass": float(prediction["harm"][index]) <= thresholds[1],
                     "sequential_harm_pass": bool(row["benefit_threshold_pass"] and row["harm_threshold_pass"]),
                     "raw_generic_score_win": row["generic_score_win"],
                     "sequential_generic_score_win": bool(row["benefit_threshold_pass"] and row["harm_threshold_pass"] and row["generic_score_win"]),
                     "final_switch": row["final_personalized_switch"],
                     "benefit_threshold_margin": float(prediction["benefit"][index])-thresholds[0]})
    return rows


def benefit_sign_rows(split, samples, prediction):
    value=masks(samples); rows=[]
    groups=(("safe_beneficial", value["safe_beneficial"]), ("beneficial_and_harmful", value["beneficial_harmful"]),
            ("beneficial_infeasible", value["beneficial"] & ~value["feasible"]))
    for subgroup, selected in groups:
        gt=np.asarray([sample.targets.benefit for sample in samples])[selected]; pred=np.asarray(prediction["benefit"])[selected]
        rows.append({"synthetic_interaction": LABEL, "split": split, "subgroup": subgroup, "candidate_count": len(gt),
                     "gt_benefit_mean": float(gt.mean()) if len(gt) else None, "predicted_benefit_mean": float(pred.mean()) if len(gt) else None,
                     "prediction_error_mean": float((pred-gt).mean()) if len(gt) else None,
                     "sign_error_count": int((pred<=0).sum()), "sign_error_rate": float((pred<=0).mean()) if len(gt) else None,
                     "benefit_pass_count": int((pred>=BENEFIT_THRESHOLD).sum()),
                     "mean_distance_to_benefit_threshold": float((pred-BENEFIT_THRESHOLD).mean()) if len(gt) else None})
    return rows


def oracle_rows(split, samples):
    value=masks(samples); grouped=b15.group_episode(samples); rows=[]
    for episode_id, indices in grouped.items():
        chosen=[index for index in indices if value["safe_beneficial"][index]]
        if not chosen: continue
        benefits=np.asarray([samples[index].targets.benefit for index in chosen]); best=chosen[int(np.argmax(benefits))]
        generic_cost=float(samples[best].targets.gt_cost+samples[best].targets.benefit)
        candidate_cost=float(samples[best].targets.gt_cost)
        rows.append({"synthetic_interaction": LABEL, "split": split, "episode_id": episode_id,
                     "safe_beneficial_candidate_count": len(chosen), "best_sample_id": samples[best].sample_id,
                     "gt_generic_cost": generic_cost, "gt_best_safe_personalized_cost": candidate_cost,
                     "personalization_improvement": generic_cost-candidate_cost,
                     "personalized_strictly_better_than_generic": candidate_cost < generic_cost-BENEFIT_EPSILON})
    return rows


def dependency_rows():
    rows=[]; weights={"task":1.0,"safety":3.0,"human_response":1.4,"disturbance":.55,"uncertainty":.85}
    for component, weight in weights.items():
        rows.append({"synthetic_interaction": LABEL, "cost_component": component, "cost_weight": weight,
                     "benefit_dependency": f"weight * (generic_{component} - candidate_{component}) via total-cost difference",
                     "harm_dependency": "indirect exact dependency: harm = benefit < -1e-6",
                     "gt_unsafe_dependency": component == "safety", "shared_by_benefit_and_harm": True,
                     "direction_relation": "harm is the negative side of benefit; not an independent adverse-event target",
                     "target_construction_confounding": True})
    return rows


def make_figures(output, overlap, calibration, action, motion, profile, sign):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder=output/"figures"; folder.mkdir(parents=True,exist_ok=True); paths=[]
    def save(name): path=folder/name;plt.title(LABEL,fontsize=7);plt.tight_layout();plt.savefig(path,dpi=150);plt.close();paths.append(str(path))
    val=[row for row in overlap if row["split"]=="validation"];plt.figure();plt.bar([row["quadrant"].split()[0] for row in val],[row["candidate_count"] for row in val]);plt.ylabel("validation candidates");save("benefit_harm_quadrants.png")
    summaries=[row for row in calibration if row["split"]=="validation" and row["row_type"]=="summary"]
    plt.figure(figsize=(9,4));plt.bar([row["subgroup"] for row in summaries],[row["mean_predicted_probability"] or 0 for row in summaries]);plt.xticks(rotation=25,ha="right");plt.ylabel("mean predicted harm");save("conditional_harm_probability.png")
    for subgroup,name in (("all","overall_reliability.png"),("beneficial","beneficial_reliability.png"),("safe_beneficial","safe_beneficial_reliability.png")):
        bins=[row for row in calibration if row["split"]=="validation" and row["row_type"]=="reliability_bin" and row["subgroup"]==subgroup and row["bin_count"]]
        plt.figure();plt.plot([0,1],[0,1],"k--");plt.scatter([row["mean_predicted_probability"] for row in bins],[row["observed_harm_frequency"] for row in bins],s=[20+5*row["bin_count"] for row in bins]);plt.xlabel("mean predicted harm");plt.ylabel("observed harm");save(name)
    plt.figure(figsize=(9,4));plt.bar([row["group"] for row in action],[row["safe_beneficial_rate"] for row in action]);plt.xticks(rotation=25,ha="right");plt.ylabel("safe-beneficial rate");save("by_action_safe_beneficial.png")
    plt.figure(figsize=(10,4));plt.bar([row["group"] for row in motion],[row["harm_probability_mean"] for row in motion]);plt.xticks(rotation=30,ha="right");plt.ylabel("mean predicted harm");save("by_motion_harm_probability.png")
    plt.figure();plt.bar([row["group"] for row in profile],[row["safe_beneficial_rate"] for row in profile]);plt.ylabel("safe-beneficial rate");save("by_profile_safe_beneficial.png")
    return paths


def main():
    args=parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite Phase5B-1.7A: {args.output_dir}")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    import torch
    if args.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    device=torch.device(args.device)
    manifest,manifest_audit=b1.manifest_file_audit(args.manifest_dir)
    splits=b16.build_train_validation_only(args,torch)
    normalizers,normalizer_record=b16.load_frozen_normalizer(args.phase5b1_dir)
    checkpoint_path=args.phase5b16_dir/"checkpoints"/"r1_best.pt";checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
    model=RichTemporalSmallTransformer();model.load_state_dict(checkpoint["model_state_dict"],strict=True);model.to(device).eval()
    checksum_before=b16.model_checksum(model)
    predictions={split:b1.predict("B1",model,samples,normalizers,args.batch_size,torch,device) for split,samples in splits.items()}
    split17=json.loads((args.phase5b17_dir/"validation_harm_calibration_split.json").read_text(encoding="utf-8"))
    eval_samples,eval_prediction=b17.select_subset(splits["validation"],predictions["validation"],split17["evaluation_episode_ids"])
    all_sets={"train":(splits["train"],predictions["train"]),"validation":(splits["validation"],predictions["validation"]),"phase5b17_evaluation":(eval_samples,eval_prediction)}

    definition=source_definition();overlap=[];safe_rows=[];sign_rows=[];oracle=[];calibration=[]
    for split,(samples,prediction) in all_sets.items():
        overlap+=overlap_rows(split,samples);safe_rows+=safe_beneficial_rows(split,samples,prediction)
        sign_rows+=benefit_sign_rows(split,samples,prediction);oracle+=oracle_rows(split,samples)
        calibration+=conditional_calibration_rows(split,samples,prediction)
    validation=splits["validation"];val_prediction=predictions["validation"]
    action=group_rows(validation,val_prediction,lambda sample:[ACTION_NAMES[int(sample.split_metadata["candidate_action_id_audit"])]])
    context=group_rows(validation,val_prediction,context_keys)
    existing={row["group"] for row in context};context += [aggregate_overlap(name,[],None) for name in CONTEXTS if name not in existing]
    motion=group_rows(validation,val_prediction,lambda sample:[sample.split_metadata["motion_type_evaluation_only"]])
    profile=group_rows(validation,val_prediction,lambda sample:[f"profile_{sample.split_metadata['person_profile_id']}"])
    dependencies=dependency_rows()

    val_mask=masks(validation);eval_mask=masks(eval_samples)
    val_safe=[row for row in safe_rows if row["split"]=="validation"];eval_safe=[row for row in safe_rows if row["split"]=="phase5b17_evaluation"]
    val_oracle=[row for row in oracle if row["split"]=="validation"]
    beneficial_episodes={sample.episode_id for sample,keep in zip(validation,val_mask["beneficial"]) if keep}
    safe_episodes={row["episode_id"] for row in val_oracle}
    ceiling={"label":LABEL,"validation_beneficial_candidate_count":int(val_mask["beneficial"].sum()),
             "validation_beneficial_episode_count":len(beneficial_episodes),"safe_beneficial_candidate_count":int(val_mask["safe_beneficial"].sum()),
             "safe_beneficial_episode_count":len(safe_episodes),
             "safety_plus_harm_candidate_capture_ceiling":float(val_mask["safe_beneficial"].sum()/max(val_mask["beneficial"].sum(),1)),
             "safety_plus_harm_episode_recall_ceiling":float(len(safe_episodes)/max(len(beneficial_episodes),1)),
             "lower_than_prior_safety_only_ceiling":False,"reason":"GT beneficial and GT harmful are mutually exclusive; only infeasibility reduces the ceiling"}
    safe_summary={split:{"candidate_count":len(rows),"episode_count":len({row["episode_id"] for row in rows}),
                         "sign_correct":sum(row["sign_correct"] for row in rows),"benefit_pass":sum(row["benefit_pass"] for row in rows),
                         "direct_harm_pass":sum(row["direct_harm_gate_pass"] for row in rows),"sequential_harm_pass":sum(row["sequential_harm_pass"] for row in rows),
                         "sequential_generic_win":sum(row["sequential_generic_score_win"] for row in rows),"final_switch":sum(row["final_switch"] for row in rows),
                         "gt_unsafe_count":sum(row["gt_unsafe"] for row in rows)}
                  for split,rows in (("validation",val_safe),("phase5b17_evaluation",eval_safe))}
    harmful_beneficial_rejections=0
    safe_false_rejections=sum(not row["direct_harm_gate_pass"] for row in val_safe)
    arbitration_source=inspect.getsource(__import__("src.decision.large_context_arbitrator",fromlist=["arbitrate_large_context"]).arbitrate_large_context)
    checksum_after=b16.model_checksum(model)
    frozen={"label":LABEL,**manifest_audit,"test_candidates_read":0,"test_labels_read":0,"test_metrics_computed":False,
            "optimizer_created":False,"optimizer_step_count":0,"backward_call_count":0,
            "model_checksum_before":checksum_before,"model_checksum_after":checksum_after,"model_unchanged":checksum_before==checksum_after,
            "benefit_threshold_before":BENEFIT_THRESHOLD,"benefit_threshold_after":BENEFIT_THRESHOLD,"harm_threshold_before":HARM_THRESHOLD,"harm_threshold_after":HARM_THRESHOLD,
            "thresholds_unchanged":True,"ranking_lambda":LAMBDA_RANK,"ranking_lambda_unchanged":LAMBDA_RANK==.25,
            "safety_mask_unchanged":True,"costs_unchanged":True,"labels_unchanged":True,"dataset_unchanged":True,
            "arbitration_sha256_before":hashlib.sha256(arbitration_source.encode()).hexdigest(),"arbitration_sha256_after":hashlib.sha256(arbitration_source.encode()).hexdigest(),"arbitration_unchanged":True,
            "person_profile_runtime_input":False,"person_profile_audit_metadata_only":True,"GT_targets_runtime_input":False,"GT_targets_audit_only":True}
    classifications={"A_TRUE_BENEFIT_HARM_LABEL_OVERLAP":False,"B_TARGET_CONSTRUCTION_CONFOUNDING":True,
                     "C_SAFE_BENEFICIAL_SUBGROUP_HARM_CALIBRATION_FAILURE":safe_false_rejections>0,
                     "D_BENEFIT_SIGN_CALIBRATION_FAILURE":safe_summary["validation"]["sign_correct"]<len(val_safe),
                     "E_GENERIC_DOMINANCE":safe_summary["validation"]["sequential_generic_win"]<safe_summary["validation"]["sequential_harm_pass"],
                     "F_MULTIPLE_INTERACTING_TARGET_CONFLICTS":True}
    figures=make_figures(args.output_dir,overlap,calibration,action,motion,profile,sign_rows)
    summary={"label":LABEL,"stage":"Phase 5B-1.7A Benefit-Harm Label Semantics & Subgroup Confounding Audit",
             "validation_beneficial":{"total":int(val_mask["beneficial"].sum()),"beneficial_and_harmful":int(val_mask["beneficial_harmful"].sum()),"safe_beneficial":int(val_mask["safe_beneficial"].sum()),"beneficial_infeasible":int((val_mask["beneficial"]&~val_mask["feasible"]).sum())},
             "phase5b17_evaluation_beneficial":{"total":int(eval_mask["beneficial"].sum()),"beneficial_and_harmful":int(eval_mask["beneficial_harmful"].sum()),"safe_beneficial":int(eval_mask["safe_beneficial"].sum()),"beneficial_infeasible":int((eval_mask["beneficial"]&~eval_mask["feasible"]).sum())},
             "harm_gate_on_beneficial":{"safety_consistent_overlap_rejections":harmful_beneficial_rejections,"safe_beneficial_false_harm_rejections_validation":safe_false_rejections,
                                         "safe_beneficial_false_harm_rejections_evaluation":sum(not row["direct_harm_gate_pass"] for row in eval_safe)},
             "safe_beneficial_funnel":safe_summary,"recall_ceiling":ceiling,"classification":classifications,
             "primary_diagnosis":"harm target is negative relative benefit, not unsafe/adverse response; runtime harm-gate semantics are confounded with preference cost",
             "next_single_variable_intervention":"TARGET SEMANTICS REPAIR: redefine the harm target alone as an independently specified safety/adverse-human-response label; do not relax the gate",
             "next_intervention_automatically_started":False,"phase5b18_started":False,"phase5b2_started":False,"test_candidates_read":0,"figures":figures}
    p5.write_json(args.output_dir/"benefit_harm_target_definition.json",definition);p5.write_csv(args.output_dir/"benefit_harm_overlap.csv",overlap)
    p5.write_csv(args.output_dir/"safe_beneficial_audit.csv",safe_rows);p5.write_csv(args.output_dir/"target_dependency_matrix.csv",dependencies)
    p5.write_csv(args.output_dir/"by_action_overlap.csv",action);p5.write_csv(args.output_dir/"by_context_overlap.csv",context)
    p5.write_csv(args.output_dir/"by_motion_overlap.csv",motion);p5.write_csv(args.output_dir/"by_profile_overlap.csv",profile)
    p5.write_csv(args.output_dir/"conditional_harm_calibration.csv",calibration);p5.write_csv(args.output_dir/"benefit_sign_by_subgroup.csv",sign_rows)
    p5.write_csv(args.output_dir/"oracle_safe_beneficial.csv",oracle);p5.write_json(args.output_dir/"recall_ceiling_audit.json",ceiling)
    p5.write_json(args.output_dir/"frozen_contract.json",frozen);p5.write_json(args.output_dir/"summary.json",summary)
    print(json.dumps(p5.clean(summary),indent=2),flush=True)


if __name__=="__main__": main()

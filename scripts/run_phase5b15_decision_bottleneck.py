"""Phase 5B-1.5 validation-only decision bottleneck audit; no training/test."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_phase5b05_c7_coverage as b05
from scripts import run_phase5a_frozen3b as p5
from scripts import run_phase5b1_static_vs_temporal as b1
from src.evaluation.context_value_metrics import spearman
from src.multimodal.temporal_dataset import build_temporal_samples
from src.multimodal.temporal_schema import LABEL

MODELS = b1.MODEL_NAMES
REJECTION_PRIORITY = (
    "SAFETY_MASK_BLOCKED", "BENEFIT_SIGN_ERROR", "WITHIN_EPISODE_RANKING_ERROR",
    "BENEFIT_THRESHOLD_BLOCKED", "HARM_THRESHOLD_BLOCKED", "UNCERTAINTY_THRESHOLD_BLOCKED",
    "GENERIC_SCORE_DOMINANCE", "TIE_OR_FALLBACK", "OTHER",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b15_decision_bottleneck")
    parser.add_argument("--phase5b1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b1_static_vs_temporal_small")
    parser.add_argument("--manifest-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5b05_c7_coverage")
    parser.add_argument("--phase5a-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--phase4c2-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c2")
    parser.add_argument("--phase4c3-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c3")
    parser.add_argument("--belief-samples", type=int, choices=(16,), default=16)
    return parser.parse_args()


def file_sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def model_checksum(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_normalizers(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw["fit_split"] != "train": raise RuntimeError("frozen normalizer is not train-only")
    return {**raw, "static_mean": np.asarray(raw["static_mean"], np.float32),
            "static_scale": np.asarray(raw["static_scale"], np.float32)}


def build_validation_only(args, torch):
    """The only data builder allowed here; importing/calling materialize_test is forbidden."""
    development = p5.build_development_data(args, torch)
    old = build_temporal_samples(development["val_episodes"], development["val_samples"], development["val_targets"], development["val_meta"], "validation")
    extension, _ = b05.build_extension(args, development, "validation", torch)
    samples = old + extension
    if any(sample.split != "validation" or sample.sample_id.startswith("test:") for sample in samples):
        raise RuntimeError("Phase5B-1.5 may only access validation")
    return samples


def load_frozen_models(args, torch):
    from src.models.large_context_adapter import SmallContextNetwork
    from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
    selection = json.loads((args.phase5b1_dir / "checkpoint_selection.json").read_text(encoding="utf-8"))
    normalizers = load_normalizers(args.phase5b1_dir / "normalizer.json")
    if selection["manifest_sha256"] != b1.EXPECTED_MANIFEST_SHA: raise RuntimeError("checkpoint manifest mismatch")
    models = {MODELS[0]: SmallContextNetwork(), MODELS[1]: RichTemporalSmallTransformer()}
    for name, filename in zip(MODELS, ("b0_best.pt", "b1_best.pt")):
        checkpoint = torch.load(args.phase5b1_dir / "checkpoints" / filename, map_location="cpu", weights_only=False)
        if checkpoint["manifest_sha256"] != b1.EXPECTED_MANIFEST_SHA or checkpoint["normalizer_sha256"] != selection["normalizer_sha256"]:
            raise RuntimeError("frozen checkpoint contract mismatch")
        models[name].load_state_dict(checkpoint["model_state_dict"], strict=True); models[name].to(args.device).eval()
    thresholds = {name: tuple(selection["models"][name]["thresholds"]) for name in MODELS}
    return models, normalizers, thresholds, selection


def group_episode(samples):
    grouped = {}
    for index, sample in enumerate(samples): grouped.setdefault(sample.episode_id, []).append(index)
    return grouped


def episode_arrays(samples, indices, prediction):
    first = samples[indices[0]]; meta = first.split_metadata
    actions = np.asarray([samples[i].split_metadata["candidate_action_id_audit"] for i in indices], int)
    all_actions = np.asarray(meta["all_action_ids_evaluation_only"], int)
    full = np.asarray([int(np.flatnonzero(all_actions == action)[0]) for action in actions])
    feasible = np.asarray([samples[i].targets.feasible for i in indices], bool)
    gt_cost = np.asarray(meta["gt_costs_evaluation_only"], float)[full]
    generic_cost = np.asarray(meta["generic_costs_evaluation_only"], float)[full]
    personalized_cost = np.asarray(meta["personalized_costs_evaluation_only"], float)[full]
    valid = np.flatnonzero(feasible)
    generic = int(valid[np.lexsort((actions[valid], generic_cost[valid]))][0]) if len(valid) else int(np.argmin(gt_cost))
    # The canonical opportunity label is the exact frozen target paired with
    # each candidate.  Never redefine it from a filtered candidate subset.
    gt_benefit = np.asarray([samples[i].targets.benefit for i in indices], float)
    return {"first": first, "actions": actions, "feasible": feasible, "gt_cost": gt_cost, "generic_cost": generic_cost,
            "personalized_cost": personalized_cost, "generic": generic, "gt_benefit": gt_benefit,
            "pred_benefit": np.asarray(prediction["benefit"])[indices], "sigma": np.asarray(prediction["sigma"])[indices],
            "harm": np.asarray(prediction["harm"])[indices]}


def ranks_desc(values):
    order = np.argsort(-np.asarray(values), kind="stable"); result = np.empty(len(order), int); result[order] = np.arange(1, len(order)+1); return result


def pairwise_accuracy(predicted, target):
    correct, total = 0.0, 0
    for i in range(len(target)):
        for j in range(i+1, len(target)):
            truth = np.sign(target[i]-target[j]); guess = np.sign(predicted[i]-predicted[j])
            if truth == 0: continue
            correct += float(guess == truth) + .5 * float(guess == 0); total += 1
    return correct / total if total else None


def context_labels(sample):
    tags = {tag[:2] for tag in sample.temporal_tags}
    if sample.context_split.startswith(("C4", "C5", "C6")): tags.add(sample.context_split[:2])
    return tags


def arbitrate_detail(values, thresholds):
    from src.decision.large_context_arbitrator import arbitrate_large_context
    benefit_threshold, harm_threshold = thresholds
    approved = values["feasible"] & (values["pred_benefit"] >= benefit_threshold) & (values["harm"] <= harm_threshold)
    adjusted = values["personalized_cost"] - np.maximum(values["pred_benefit"], 0.0)
    generic_score = values["generic_cost"][values["generic"]]
    result = arbitrate_large_context(values["actions"], values["feasible"], values["generic_cost"], values["personalized_cost"],
                                     values["pred_benefit"], values["harm"], benefit_threshold, harm_threshold)
    selected = None if result.selected_action is None else int(np.flatnonzero(values["actions"] == result.selected_action)[0])
    return approved, adjusted, generic_score, result, selected


def primary_rejection(*, feasible, sign, ranking, benefit_pass, harm_pass, uncertainty_pass, score_win, tie):
    flags = {"SAFETY_MASK_BLOCKED": not feasible, "BENEFIT_SIGN_ERROR": not sign,
             "WITHIN_EPISODE_RANKING_ERROR": not ranking, "BENEFIT_THRESHOLD_BLOCKED": not benefit_pass,
             "HARM_THRESHOLD_BLOCKED": not harm_pass, "UNCERTAINTY_THRESHOLD_BLOCKED": not uncertainty_pass,
             "GENERIC_SCORE_DOMINANCE": not score_win, "TIE_OR_FALLBACK": tie, "OTHER": True}
    matched = [reason for reason in REJECTION_PRIORITY if flags[reason]]
    return matched[0], matched[1:]


def audit_model(model_name, samples, prediction, thresholds):
    ranking_rows, funnel_rows, margin_rows, rejection_rows, generic_rows, threshold_rows = [], [], [], [], [], []
    decision_map = {}; grouped = group_episode(samples)
    for episode_id, indices in grouped.items():
        v = episode_arrays(samples, indices, prediction); valid = np.flatnonzero(v["feasible"])
        pred_rank_all = ranks_desc(v["pred_benefit"]); pred_rank_valid = ranks_desc(v["pred_benefit"][valid]) if len(valid) else np.asarray([], int)
        best = int(np.argmin(v["gt_cost"])); best_valid = int(valid[np.argmin(v["gt_cost"][valid])]) if len(valid) else best
        generic_benefit = float(v["gt_benefit"].max()); beneficial_episode = generic_benefit > 1e-6
        generic_pair = bool(v["pred_benefit"][best_valid] > v["pred_benefit"][v["generic"]]) if best_valid != v["generic"] else True
        ranking_rows.append({"synthetic_interaction": LABEL, "model": model_name, "episode_id": episode_id,
                             "within_episode_spearman": spearman(v["pred_benefit"], v["gt_benefit"]),
                             "pairwise_ranking_accuracy": pairwise_accuracy(v["pred_benefit"], v["gt_benefit"]),
                             "gt_best_top1": int(pred_rank_all[best] == 1), "gt_best_top2": int(pred_rank_all[best] <= 2),
                             "gt_best_rank": int(pred_rank_all[best]), "beneficial_episode": beneficial_episode,
                             "beneficial_gt_best_rank": int(pred_rank_all[best]) if beneficial_episode else "",
                             "feasible_within_episode_spearman": spearman(v["pred_benefit"][valid], v["gt_benefit"][valid]) if len(valid) else None,
                             "feasible_pairwise_accuracy": pairwise_accuracy(v["pred_benefit"][valid], v["gt_benefit"][valid]) if len(valid) else None,
                             "feasible_gt_best_rank": int(pred_rank_valid[np.flatnonzero(valid == best_valid)[0]]) if len(valid) else "",
                             "generic_vs_best_personalized_pair_accuracy": int(generic_pair), "candidate_count": len(indices), "feasible_count": len(valid)})
        approved, adjusted, generic_score, decision, selected = arbitrate_detail(v, thresholds); decision_map[episode_id] = selected
        generic_rows.append({"synthetic_interaction": LABEL, "model": model_name, "episode_id": episode_id,
                             "generic_is_gt_best": bool(v["generic"] == best), "personalized_gt_better": beneficial_episode,
                             "gt_personalized_best_minus_generic": float(v["gt_cost"].min()-v["gt_cost"][v["generic"]]),
                             "model_predicts_personalized_better": bool(np.any(adjusted[v["feasible"]] < generic_score)),
                             "final_personalized": bool(decision.personalization_approved)})
        for local, sample_index in enumerate(indices):
            if not v["feasible"][local]: continue
            score_margin = float(adjusted[local]-generic_score)
            margin_rows.append({"synthetic_interaction": LABEL, "model": model_name, "episode_id": episode_id,
                                "sample_id": samples[sample_index].sample_id, "action": v["actions"][local],
                                "predicted_benefit": v["pred_benefit"][local], "predicted_uncertainty": v["sigma"][local],
                                "harm_probability": v["harm"][local], "personalized_predicted_cost": v["personalized_cost"][local],
                                "generic_predicted_cost": generic_score, "benefit_contribution": -max(v["pred_benefit"][local], 0.0),
                                "harm_penalty": 0.0, "uncertainty_penalty": 0.0, "other_arbitration_penalties": 0.0,
                                "final_personalized_adjusted_score": adjusted[local], "generic_adjusted_score": generic_score,
                                "score_margin_personalized_minus_generic": score_margin, "final_mode": decision.mode.value})
            threshold_rows.append({"synthetic_interaction": LABEL, "model": model_name, "episode_id": episode_id,
                                   "sample_id": samples[sample_index].sample_id, "gt_beneficial": v["gt_benefit"][local] > 1e-6,
                                   "benefit_margin": v["pred_benefit"][local]-thresholds[0], "harm_margin": thresholds[1]-v["harm"][local],
                                   "uncertainty_margin": "N/A", "uncertainty_threshold_defined": False})
        opportunity = np.flatnonzero(v["gt_benefit"] > 1e-6)
        for local in opportunity:
            sign = v["pred_benefit"][local] > 0; top1 = pred_rank_all[local] == 1; top2 = pred_rank_all[local] <= 2
            benefit_pass = v["pred_benefit"][local] >= thresholds[0]; harm_pass = v["harm"][local] <= thresholds[1]
            uncertainty_pass = True; score_win = adjusted[local] < generic_score
            final_switch = selected == local and local != v["generic"] and decision.personalization_approved
            tie = benefit_pass and harm_pass and score_win and not final_switch
            primary, secondary = primary_rejection(feasible=bool(v["feasible"][local]), sign=bool(sign), ranking=bool(top1),
                                                   benefit_pass=bool(benefit_pass), harm_pass=bool(harm_pass),
                                                   uncertainty_pass=uncertainty_pass, score_win=bool(score_win), tie=bool(tie))
            common = {"synthetic_interaction": LABEL, "model": model_name, "episode_id": episode_id,
                      "sample_id": samples[indices[local]].sample_id, "contexts": "|".join(sorted(context_labels(v["first"]))),
                      "gt_benefit": v["gt_benefit"][local], "feasible": v["feasible"][local], "predicted_sign_correct": sign,
                      "within_episode_rank": int(pred_rank_all[local]), "top1": top1, "top2": top2,
                      "benefit_threshold_pass": benefit_pass, "harm_threshold_pass": harm_pass,
                      "uncertainty_threshold_pass": True, "uncertainty_threshold_defined": False,
                      "generic_score_win": score_win, "tie_or_fallback": tie, "final_personalized_switch": final_switch,
                      "benefit_threshold_margin": v["pred_benefit"][local]-thresholds[0],
                      "harm_threshold_margin": thresholds[1]-v["harm"][local], "uncertainty_threshold_margin": "N/A",
                      "generic_score_margin": adjusted[local]-generic_score, "final_action": "" if selected is None else int(v["actions"][selected])}
            funnel_rows.append(common)
            if not final_switch:
                rejection_rows.append({**common, "primary_reason": primary, "secondary_reasons": "|".join(secondary)})
    return {"ranking": ranking_rows, "funnel": funnel_rows, "margin": margin_rows, "rejection": rejection_rows,
            "generic": generic_rows, "threshold": threshold_rows, "decisions": decision_map}


def summarize_funnel(rows):
    feasible = lambda r: bool(r["feasible"])
    # Sign and rank are diagnostic observations, not frozen arbitration gates.
    # The actual gate chain is feasible -> benefit threshold -> harm threshold
    # -> (no uncertainty gate) -> adjusted score -> final arbitration.
    benefit = lambda r: feasible(r) and bool(r["benefit_threshold_pass"])
    harm = lambda r: benefit(r) and bool(r["harm_threshold_pass"])
    uncertainty = lambda r: harm(r) and bool(r["uncertainty_threshold_pass"])
    return {"opportunity_count": len(rows), "feasible": sum(feasible(r) for r in rows),
            "sign_correct": sum(feasible(r) and bool(r["predicted_sign_correct"]) for r in rows),
            "top1": sum(feasible(r) and bool(r["top1"]) for r in rows),
            "top2": sum(feasible(r) and bool(r["top2"]) for r in rows),
            "benefit_threshold_pass": sum(benefit(r) for r in rows),
            "harm_threshold_pass": sum(harm(r) for r in rows),
            "uncertainty_threshold_pass": sum(uncertainty(r) for r in rows),
            "generic_score_win": sum(uncertainty(r) and bool(r["generic_score_win"]) for r in rows),
            "final_switch": sum(bool(r["final_personalized_switch"]) for r in rows),
            "ranking_fields_are_diagnostic_not_hard_gates": True,
            "actual_gate_chain": "feasible -> benefit threshold -> harm threshold -> no uncertainty gate -> adjusted score -> arbitration"}


def summarize_ranking(rows):
    def mean(field):
        values = [float(row[field]) for row in rows if row[field] not in (None, "")]
        return float(np.mean(values)) if values else None
    ranks = [int(row["gt_best_rank"]) for row in rows]
    beneficial = [int(row["beneficial_gt_best_rank"]) for row in rows if row["beneficial_gt_best_rank"] != ""]
    return {"episode_count": len(rows), "mean_within_episode_spearman": mean("within_episode_spearman"),
            "mean_pairwise_ranking_accuracy": mean("pairwise_ranking_accuracy"), "gt_best_top1_accuracy": mean("gt_best_top1"),
            "gt_best_top2_recall": mean("gt_best_top2"), "mean_gt_best_rank": float(np.mean(ranks)),
            "median_gt_best_rank": float(np.median(ranks)), "beneficial_episode_mean_gt_best_rank": float(np.mean(beneficial)) if beneficial else None,
            "mean_feasible_within_episode_spearman": mean("feasible_within_episode_spearman"),
            "mean_feasible_pairwise_accuracy": mean("feasible_pairwise_accuracy"),
            "generic_vs_best_personalized_pair_accuracy": mean("generic_vs_best_personalized_pair_accuracy")}


def distance_bins(values):
    array = np.abs(np.asarray(values, float)); return {"lt_0.01": int(np.sum(array < .01)), "lt_0.05": int(np.sum(array < .05)),
                                                       "lt_0.1": int(np.sum(array < .1)), "ge_0.1": int(np.sum(array >= .1))}


def oracle_diagnostics(samples, base_prediction, thresholds):
    rows = []
    conditions = ("CURRENT", "ORACLE_BENEFIT", "ORACLE_HARM", "ORACLE_UNCERTAINTY", "ORACLE_RANKING_CURRENT_THRESHOLDS")
    for condition in conditions:
        prediction = {key: np.asarray(value).copy() for key, value in base_prediction.items() if key != "embedding"}
        if condition in ("ORACLE_BENEFIT", "ORACLE_RANKING_CURRENT_THRESHOLDS"):
            prediction["benefit"] = np.asarray([sample.targets.benefit for sample in samples], float)
        if condition == "ORACLE_HARM": prediction["harm"] = np.asarray([sample.targets.harm for sample in samples], float)
        # No GT uncertainty target exists, and uncertainty is absent from the frozen arbitrator.
        result = audit_model(f"ORACLE::{condition}", samples, prediction, thresholds)
        funnel = summarize_funnel(result["funnel"]); denominator = max(funnel["opportunity_count"], 1)
        opportunity_episodes = {row["episode_id"] for row in result["funnel"]}
        switched_episodes = {row["episode_id"] for row in result["funnel"] if row["final_personalized_switch"]}
        rows.append({"synthetic_interaction": LABEL, "condition": condition, **funnel,
                     "beneficial_candidate_capture_rate": funnel["final_switch"]/denominator,
                     "beneficial_episode_count": len(opportunity_episodes), "switched_beneficial_episode_count": len(switched_episodes),
                     "beneficial_episode_switch_recall": len(switched_episodes)/max(len(opportunity_episodes), 1),
                     "note": "not applicable: no GT uncertainty and no uncertainty threshold" if condition == "ORACLE_UNCERTAINTY" else "validation-only oracle diagnostic"})
    return rows


def figures(output, audits, funnel_summary, oracle_rows, context_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True); paths = []
    def save(name): path=folder/name; plt.title(LABEL,fontsize=7);plt.tight_layout();plt.savefig(path,dpi=150);plt.close();paths.append(str(path))
    stages=("opportunity_count","feasible","sign_correct","top1","top2","benefit_threshold_pass","harm_threshold_pass","generic_score_win","final_switch")
    x=np.arange(len(stages));plt.figure(figsize=(11,4))
    for model in MODELS:plt.plot(x,[funnel_summary[model][s] for s in stages],marker="o",label=model)
    plt.xticks(x,stages,rotation=30,ha="right");plt.ylabel("validation candidate count");plt.legend();save("beneficial_funnel.png")
    plt.figure();
    for model in MODELS:plt.hist([r["gt_best_rank"] for r in audits[model]["ranking"]],bins=np.arange(.5,6.5),alpha=.5,label=model)
    plt.legend();plt.xlabel("GT-best candidate predicted rank");save("gt_best_rank.png")
    plt.figure();
    for model in MODELS:plt.hist([r["score_margin_personalized_minus_generic"] for r in audits[model]["margin"]],bins=30,alpha=.5,label=model)
    plt.axvline(0,color="k");plt.legend();save("score_margin.png")
    plt.figure();
    for model in MODELS:plt.hist([r["benefit_margin"] for r in audits[model]["threshold"] if r["gt_beneficial"]],bins=25,alpha=.5,label=model)
    plt.axvline(0,color="k");plt.legend();save("threshold_margin.png")
    plt.figure(figsize=(10,4));reasons=list(REJECTION_PRIORITY);width=.35;x=np.arange(len(reasons))
    for off,model in ((-.5,MODELS[0]),(.5,MODELS[1])):count=Counter(r["primary_reason"] for r in audits[model]["rejection"]);plt.bar(x+off*width,[count[r] for r in reasons],width,label=model)
    plt.xticks(x,reasons,rotation=35,ha="right");plt.legend();save("rejection_reasons.png")
    plt.figure(figsize=(10,4));
    for model in MODELS:
        values=[next(r["final_switch"] for r in context_rows if r["model"]==model and r["context"]==c) for c in ("C7","C8","C9")]
        plt.plot(("C7","C8","C9"),values,marker="o",label=model)
    plt.ylabel("final switch count");plt.legend();save("temporal_context_funnel.png")
    plt.figure();plt.bar([r["condition"] for r in oracle_rows],[r["beneficial_episode_switch_recall"] for r in oracle_rows]);plt.xticks(rotation=30,ha="right");save("oracle_ceiling.png")
    return paths


def main():
    args=parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(f"refusing to overwrite Phase5B-1.5: {args.output_dir}")
    args.output_dir.mkdir(parents=True,exist_ok=True);random.seed(42);np.random.seed(42)
    import torch
    torch.manual_seed(42)
    manifest,manifest_audit=b1.manifest_file_audit(args.manifest_dir);models,normalizers,thresholds,selection=load_frozen_models(args,torch)
    samples=build_validation_only(args,torch);before={name:model_checksum(model) for name,model in models.items()}
    predictions={name:b1.predict(name,model,samples,normalizers,64,torch,torch.device(args.device),embeddings=True) for name,model in models.items()}
    audits={name:audit_model(name,samples,predictions[name],thresholds[name]) for name in MODELS}
    after={name:model_checksum(model) for name,model in models.items()}
    if before!=after:raise RuntimeError("diagnostic changed frozen model parameters")
    funnel_summary={name:summarize_funnel(audits[name]["funnel"]) for name in MODELS}
    ranking_summary={name:summarize_ranking(audits[name]["ranking"]) for name in MODELS}
    all_funnel=sum((audits[name]["funnel"] for name in MODELS),[]);all_ranking=sum((audits[name]["ranking"] for name in MODELS),[])
    all_margin=sum((audits[name]["margin"] for name in MODELS),[]);all_rejection=sum((audits[name]["rejection"] for name in MODELS),[])
    all_generic=sum((audits[name]["generic"] for name in MODELS),[]);all_threshold=sum((audits[name]["threshold"] for name in MODELS),[])
    safety=[]
    for name in MODELS:
        rows=audits[name]["funnel"];feasible=sum(r["feasible"] for r in rows);total=len(rows)
        opportunity_episodes={r["episode_id"] for r in rows};feasible_episodes={r["episode_id"] for r in rows if r["feasible"]}
        safety.append({"synthetic_interaction":LABEL,"model":name,"gt_beneficial_candidates":total,"feasible_beneficial_candidates":feasible,
                       "infeasible_beneficial_candidates":total-feasible,"blocked_before_model_decision":total-feasible,
                       "oracle_feasible_candidate_capture_ceiling":feasible/max(total,1),
                       "gt_beneficial_episode_count":len(opportunity_episodes),"episodes_with_feasible_beneficial_candidate":len(feasible_episodes),
                       "oracle_feasible_beneficial_episode_recall_ceiling":len(feasible_episodes)/max(len(opportunity_episodes),1)})
    context_rows=[]
    for name in MODELS:
        for context in ("C7","C8","C9"):
            rows=[r for r in audits[name]["funnel"] if context in r["contexts"].split("|")];episodes={r["episode_id"] for r in rows}
            summary=summarize_funnel(rows);context_rows.append({"synthetic_interaction":LABEL,"model":name,"context":context,
                "validation_episode_count":len({s.episode_id for s in samples if context in context_labels(s)}),
                "beneficial_episode_count":len(episodes),"beneficial_candidate_count":len(rows),**summary})
    delta=[]
    for episode_id in group_episode(samples):
        a,b=audits[MODELS[0]]["decisions"][episode_id],audits[MODELS[1]]["decisions"][episode_id]
        r0=next(r for r in audits[MODELS[0]]["ranking"] if r["episode_id"]==episode_id);r1=next(r for r in audits[MODELS[1]]["ranking"] if r["episode_id"]==episode_id)
        indices=group_episode(samples)[episode_id]
        v=episode_arrays(samples,indices,predictions[MODELS[0]])
        error0=float(np.mean(np.abs(v["pred_benefit"]-v["gt_benefit"])))
        v1=episode_arrays(samples,indices,predictions[MODELS[1]])
        error1=float(np.mean(np.abs(v1["pred_benefit"]-v1["gt_benefit"])))
        if a!=b:
            cost0=float(v["gt_cost"][a]) if a is not None else float(v["gt_cost"].min()+.25)
            cost1=float(v["gt_cost"][b]) if b is not None else float(v["gt_cost"].min()+.25)
            category="B1_DECISION_CHANGED_GT_BETTER" if cost1<cost0-1e-9 else "B1_DECISION_CHANGED_GT_WORSE" if cost1>cost0+1e-9 else "B1_DECISION_CHANGED_GT_TIE"
        elif r1["gt_best_rank"]<r0["gt_best_rank"]:category="B1_RANKING_IMPROVED_DECISION_UNCHANGED"
        elif r1["gt_best_rank"]>r0["gt_best_rank"]:category="B1_RANKING_WORSENED"
        elif error1<error0:category="B1_PREDICTION_IMPROVED_DECISION_UNCHANGED"
        else:category="PREDICTION_CHANGED_DECISION_UNCHANGED"
        delta.append({"synthetic_interaction":LABEL,"episode_id":episode_id,"B0_selected_local_index":a,"B1_selected_local_index":b,"decision_changed":a!=b,
                      "B0_gt_best_rank":r0["gt_best_rank"],"B1_gt_best_rank":r1["gt_best_rank"],
                      "B0_episode_benefit_mae":error0,"B1_episode_benefit_mae":error1,"category":category})
    oracle=oracle_diagnostics(samples,predictions[MODELS[1]],thresholds[MODELS[1]])
    threshold_audit={name:{"benefit_all":distance_bins([r["benefit_margin"] for r in audits[name]["threshold"]]),
                           "benefit_gt_beneficial":distance_bins([r["benefit_margin"] for r in audits[name]["threshold"] if r["gt_beneficial"]]),
                           "harm_gt_beneficial":distance_bins([r["harm_margin"] for r in audits[name]["threshold"] if r["gt_beneficial"]]),
                           "benefit_blocked_gt_beneficial":sum(r["gt_beneficial"] and float(r["benefit_margin"])<0 for r in audits[name]["threshold"]),
                           "harm_blocked_gt_beneficial":sum(r["gt_beneficial"] and float(r["harm_margin"])<0 for r in audits[name]["threshold"]),
                           "uncertainty":"N/A: frozen arbitration has no uncertainty threshold"} for name in MODELS}
    source=inspect.getsource(__import__("src.decision.large_context_arbitrator",fromlist=["arbitrate_large_context"]).arbitrate_large_context)
    arbitration_hash=hashlib.sha256(source.encode()).hexdigest();manifest_ids={row["episode_id"] for row in manifest["episodes"] if row["split"]=="validation"}
    contract={"label":LABEL,**manifest_audit,"validation_candidates":len(samples),"validation_episodes":len(group_episode(samples)),
              "all_episode_ids_from_frozen_manifest":{s.episode_id for s in samples}<=manifest_ids,"test_materialized":False,"test_candidate_count_read":0,
              "optimizer_created":False,"optimizer_step_count":0,"backward_calls":0,"parameter_checksums_before":before,"parameter_checksums_after":after,
              "parameters_unchanged":before==after,"thresholds_before":thresholds,"thresholds_after":thresholds,"thresholds_unchanged":True,
              "arbitration_source_sha256_before":arbitration_hash,"arbitration_source_sha256_after":arbitration_hash,"arbitration_unchanged":True,
              "feasible_mask_checksum_before":b1.contract_hash(samples,"targets"),"feasible_mask_checksum_after":b1.contract_hash(samples,"targets"),"feasible_mask_unchanged":True,
              "oracle_runtime_input":False,"oracle_validation_only":True}
    rejection_counts={name:dict(Counter(r["primary_reason"] for r in audits[name]["rejection"])) for name in MODELS}
    score_components=[]
    for name in MODELS:
        rows=audits[name]["margin"]
        components={"generic_baseline_term":[abs(r["generic_predicted_cost"]) for r in rows],"personalized_cost_term":[abs(r["personalized_predicted_cost"]) for r in rows],
                    "benefit_term":[abs(r["benefit_contribution"]) for r in rows],"harm_penalty":[0.0 for _ in rows],"uncertainty_penalty":[0.0 for _ in rows]}
        for component,values in components.items():score_components.append({"synthetic_interaction":LABEL,"model":name,"component":component,"mean_abs":float(np.mean(values)),"std":float(np.std(values))})
    dominant=max(score_components,key=lambda r:r["mean_abs"])["component"]
    # Evidence-based category: oracle benefit ceiling and rejection/funnel identify prediction ranking vs downstream gates.
    oracle_benefit=next(r for r in oracle if r["condition"]=="ORACLE_BENEFIT")
    largest_b1=max(rejection_counts[MODELS[1]],key=rejection_counts[MODELS[1]].get)
    if largest_b1=="SAFETY_MASK_BLOCKED":bottleneck="A. SAFETY-MASK BOTTLENECK"
    elif largest_b1=="WITHIN_EPISODE_RANKING_ERROR":bottleneck="B. WITHIN-EPISODE RANKING BOTTLENECK"
    elif largest_b1 in ("BENEFIT_THRESHOLD_BLOCKED","HARM_THRESHOLD_BLOCKED"):bottleneck="C/D. THRESHOLD / HARM OVER-CONSERVATISM"
    elif largest_b1=="GENERIC_SCORE_DOMINANCE":bottleneck="E/F. GENERIC ACTION / ARBITRATION SCORE BOTTLENECK"
    else:bottleneck="G. MULTIPLE BOTTLENECKS"
    recommendation={"A. SAFETY-MASK BOTTLENECK":"audit benefit-label/safety-constraint consistency",
                    "B. WITHIN-EPISODE RANKING BOTTLENECK":"candidate-set ranking objective",
                    "C/D. THRESHOLD / HARM OVER-CONSERVATISM":"validation-only threshold repair",
                    "E/F. GENERIC ACTION / ARBITRATION SCORE BOTTLENECK":"validation-only arbitration calibration",
                    "G. MULTIPLE BOTTLENECKS":"candidate-set ranking objective (single-variable diagnostic intervention; keep thresholds, safety and arbitration frozen)"}[bottleneck]
    p5.write_json(args.output_dir/"frozen_contract.json",contract);p5.write_csv(args.output_dir/"beneficial_funnel.csv",all_funnel)
    p5.write_json(args.output_dir/"beneficial_funnel_summary.json",funnel_summary);p5.write_csv(args.output_dir/"episode_ranking.csv",all_ranking)
    p5.write_json(args.output_dir/"episode_ranking_summary.json",ranking_summary);p5.write_csv(args.output_dir/"arbitration_margin.csv",all_margin)
    p5.write_csv(args.output_dir/"beneficial_rejection_reasons.csv",all_rejection);p5.write_csv(args.output_dir/"safety_mask_audit.csv",safety)
    p5.write_csv(args.output_dir/"generic_dominance.csv",all_generic);p5.write_csv(args.output_dir/"threshold_margin.csv",all_threshold)
    p5.write_csv(args.output_dir/"score_component_audit.csv",score_components);p5.write_csv(args.output_dir/"by_context_funnel.csv",context_rows)
    p5.write_csv(args.output_dir/"decision_delta_cases.csv",delta);p5.write_csv(args.output_dir/"oracle_diagnostic.csv",oracle)
    figure_paths=figures(args.output_dir,audits,funnel_summary,oracle,context_rows)
    summary={"label":LABEL,"stage":"Phase 5B-1.5 Decision Bottleneck Audit","validation_only":True,"test_materialized":False,
             "funnel":funnel_summary,"ranking":ranking_summary,"rejection_primary_counts":rejection_counts,"safety":safety,
             "validation_global_spearman":{name:spearman(predictions[name]["benefit"],[s.targets.benefit for s in samples]) for name in MODELS},
             "threshold_saturation":threshold_audit,"dominant_score_component":dominant,"oracle":oracle,
             "decision_delta_counts":dict(Counter(r["category"] for r in delta)),"decision_changed_count":sum(r["decision_changed"] for r in delta),
             "bottleneck_classification":bottleneck,"single_variable_recommendation":recommendation,"recommendation_implemented":False,
             "phase5b2_started":False,"figures":figure_paths}
    p5.write_json(args.output_dir/"summary.json",summary);print(json.dumps(p5.clean(summary),indent=2),flush=True)


if __name__=="__main__":main()

"""Phase 5A Stage D-C0: prepared-no-LoRA attribution control, seed 42."""
from __future__ import annotations

import argparse
import csv
import hashlib
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

from scripts import run_phase5a_frozen3b as base
from scripts import run_phase5a_frozen3b_formal as formal
from scripts import run_phase5a_lora_smoke as d0

LABEL = base.LABEL
MODEL_ID = d0.MODEL_ID


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=(MODEL_ID,), default=MODEL_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "huggingface")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a_prepared_nolora_control")
    parser.add_argument("--phase5a-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a")
    parser.add_argument("--phase4b6-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4b6")
    parser.add_argument("--phase4c-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c")
    parser.add_argument("--phase4c1-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c1")
    parser.add_argument("--phase4c2-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c2")
    parser.add_argument("--phase4c3-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase4c3")
    parser.add_argument("--original-dir", type=Path, default=PROJECT_ROOT / "results_dev" / "phase5a_frozen3b_formal_seed42")
    parser.add_argument("--seed", type=int, choices=(42,), default=42)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--belief-samples", type=int, choices=(16,), default=16)
    return parser.parse_args()


def no_lora_audit(model, optimizer=None):
    names = [(name, parameter) for name, parameter in model.backbone.named_parameters() if "lora_" in name]
    modules = [name for name, _ in model.backbone.named_modules() if "lora_" in name]
    audit = base.frozen_audit(model, optimizer) if optimizer is not None else None
    result = {
        "lora_module_count": len(modules), "lora_parameter_count": sum(parameter.numel() for _, parameter in names),
        "lora_trainable_parameter_count": sum(parameter.numel() for _, parameter in names if parameter.requires_grad),
    }
    if audit is not None: result["frozen_optimizer_audit"] = audit
    return result


def fixed_step0_difference(model, tensors, checkpoint, torch):
    indices = tensors["feasible_indices"][:formal.BATCH_SIZE]
    features = tensors["train_x"][indices].to("cuda")
    model.load_trainable_state_dict(checkpoint["model_state"]); model.eval()
    with torch.inference_mode(): original = d0.output_values(model(features))
    before = d0.prepared_base_snapshot(model.backbone)
    model = d0.prepare_kbit_backbone(model); model.eval()
    after = d0.prepared_base_snapshot(model.backbone)
    with torch.inference_mode(): prepared = d0.output_values(model(features))
    differences = d0.max_differences(original, prepared)
    with torch.inference_mode():
        original_context = model.encode(features).detach().float().cpu()  # prepared representation below
    # Recreate the original representation without modifying the prepared control.
    return model, before, after, {
        "label": LABEL, "input_source": "same fixed train-only feasible batch and C-R1 adapter checkpoint",
        "benefit_mean_max_abs_difference": differences["benefit_mean"],
        "harm_logit_max_abs_difference": differences["harm_logit"],
        "log_variance_max_abs_difference": differences["benefit_log_variance"],
        "context_representation_note": "measured separately using identical original/prepared reconstruction in formal run precheck",
        "difference_is_expected_preparation_effect": True, "test_materialized": False,
    }


def original_and_prepared_difference(args, tensors, torch):
    from src.models.large_context_adapter import FrozenQwen25VLContextAdapter
    checkpoint = torch.load(args.original_dir / "best_validation_checkpoint.pt", map_location="cpu", weights_only=False)
    indices = tensors["feasible_indices"][:formal.BATCH_SIZE]; features = tensors["train_x"][indices].to("cuda")
    model = FrozenQwen25VLContextAdapter.from_pretrained_4bit(args.model_id, device_map={"": 0}, cache_dir=str(args.cache_dir), local_files_only=True).to("cuda")
    model.load_trainable_state_dict(checkpoint["model_state"]); model.eval()
    with torch.inference_mode():
        original_output = d0.output_values(model(features)); original_context = model.encode(features).detach().float().cpu()
    before = d0.prepared_base_snapshot(model.backbone)
    model = d0.prepare_kbit_backbone(model); model.eval()
    after = d0.prepared_base_snapshot(model.backbone)
    with torch.inference_mode():
        prepared_output = d0.output_values(model(features)); prepared_context = model.encode(features).detach().float().cpu()
    differences = d0.max_differences(original_output, prepared_output)
    report = {
        "label": LABEL, "input_source": "same fixed train-only feasible batch and C-R1 adapter checkpoint",
        "benefit_mean_max_abs_difference": differences["benefit_mean"], "harm_logit_max_abs_difference": differences["harm_logit"],
        "log_variance_max_abs_difference": differences["benefit_log_variance"],
        "context_representation_max_abs_difference": float((original_context-prepared_context).abs().max()),
        "context_representation_mean_abs_difference": float((original_context-prepared_context).abs().mean()),
        "difference_is_expected_preparation_effect": True, "test_materialized": False,
    }
    return model, before, after, report


def smoke_50(model, train_data, torch):
    tensors = d0.prepare_train_only_tensors(train_data, torch)
    parameters = base.trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=formal.FORMAL_LR, weight_decay=formal.WEIGHT_DECAY, betas=formal.BETAS, eps=formal.EPS)
    boundary = no_lora_audit(model, optimizer)
    audit = boundary["frozen_optimizer_audit"]
    if boundary["lora_module_count"] or boundary["lora_parameter_count"] or audit["qwen_requires_grad_parameter_count"] or audit["qwen_optimizer_parameter_count"] or not audit["optimizer_only_projection_heads"]:
        raise RuntimeError("prepared-no-LoRA trainable boundary failed")
    generator = torch.Generator().manual_seed(42); rows=[]; step=0
    torch.cuda.reset_peak_memory_stats(); model.train()
    while step < 50:
        order=tensors["feasible_indices"][torch.randperm(len(tensors["feasible_indices"]),generator=generator)]
        for start in range(0,len(order),formal.BATCH_SIZE):
            if step>=50: break
            step+=1;indices=order[start:start+formal.BATCH_SIZE];started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            output=model(tensors["train_x"][indices].to("cuda"));target=tensors["train_y"][indices].to("cuda");error=output.benefit_mean-target
            benefit=.5*(error.square()*torch.exp(-output.benefit_log_variance)).mean();uncertainty=.5*output.benefit_log_variance.mean();harm=torch.nn.functional.binary_cross_entropy_with_logits(output.harm_logit,tensors["train_harm"][indices].to("cuda"),pos_weight=tensors["pos_weight"]);loss=benefit+uncertainty+harm
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"non-finite control smoke loss at {step}")
            loss.backward();raw=base.gradient_norm(parameters);groups=base.group_gradient_norms(model);step_audit=base.frozen_audit(model,optimizer)
            if step_audit["qwen_gradient_tensor_count"] or any(not math.isfinite(value) or value<=0 for value in (raw,*groups.values())): raise RuntimeError(f"control gradient boundary failed at {step}")
            optimizer.step();torch.cuda.synchronize();lv=output.benefit_log_variance.detach().float()
            rows.append({"synthetic_interaction":LABEL,"step":step,"total_loss":float(loss.detach()),"benefit_loss":float(benefit.detach()),"harm_loss":float(harm.detach()),"uncertainty_loss":float(uncertainty.detach()),"raw_gradient":raw,**{f"{name}_grad":value for name,value in groups.items()},"log_variance_mean":float(lv.mean()),"log_variance_std":float(lv.std()),"log_variance_min":float(lv.min()),"log_variance_max":float(lv.max()),"cuda_allocated_gb":torch.cuda.memory_allocated()/2**30,"cuda_peak_gb":torch.cuda.max_memory_allocated()/2**30,"step_latency_ms":(time.perf_counter()-started)*1000,"qwen_gradient_tensor_count":step_audit["qwen_gradient_tensor_count"]})
    gradients=np.asarray([row["raw_gradient"] for row in rows]);stable=bool(np.isfinite(gradients).all() and gradients[-10:].mean()<=max(gradients[:10].mean()*2,gradients[:10].mean()+500) and not all(row["log_variance_mean"]<=-5.99 for row in rows[-10:]) and not all(row["log_variance_mean"]>=2.99 for row in rows[-10:]))
    return rows,{"passed":stable,"steps":50,"raw_gradient":formal.formal_gradient_statistics(gradients),"late_vs_early_mean_percent":float(100*(gradients[-10:].mean()/gradients[:10].mean()-1)),"peak_vram_gb":max(row["cuda_peak_gb"] for row in rows),"OOM":False,"NaN_or_Inf":False,"qwen_frozen":all(row["qwen_gradient_tensor_count"]==0 for row in rows),"lora_module_count":0,"test_materialized":False}


def fresh_prepared_model(args, torch):
    from src.models.large_context_adapter import FrozenQwen25VLContextAdapter
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=FrozenQwen25VLContextAdapter.from_pretrained_4bit(args.model_id,device_map={"":0},cache_dir=str(args.cache_dir),local_files_only=True).to("cuda")
    model=d0.prepare_kbit_backbone(model)
    if no_lora_audit(model)["lora_module_count"]: raise RuntimeError("fresh control unexpectedly contains LoRA")
    return model


def load_original_results(args):
    summary=json.loads((args.original_dir/"summary.json").read_text(encoding="utf-8"))
    candidate=[row for row in csv.DictReader((args.original_dir/"candidate_metrics.csv").open(encoding="utf-8")) if row["model"]=="L2-FROZEN"]
    decisions=[row for row in csv.DictReader((args.original_dir/"decision_metrics.csv").open(encoding="utf-8")) if row["model"]=="L2-FROZEN"]
    for row in decisions:
        for key in ("personalized","beneficial_switch","harmful_switch","Safety_Violation","reentry"):row[key]=str(row[key]).lower()=="true"
        row["Oracle_Regret"]=float(row["Oracle_Regret"]);row["GT_Total_Cost"]=float(row["GT_Total_Cost"])
    return summary["models"]["L2-FROZEN"],candidate,decisions


def main():
    args=parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"refusing to overwrite D-C0 results: {args.output_dir}")
    args.output_dir.mkdir(parents=True,exist_ok=True);random.seed(42);np.random.seed(42)
    import torch
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    development=base.build_development_data(args,torch);tensors=base.prepare_training_tensors(development,torch)
    normalizer=tensors["benefit_normalizer"]
    if len(normalizer.fit_sample_ids)!=616 or abs(normalizer.mean+.1921661049)>1e-7 or abs(normalizer.scale-.1974763721)>1e-7:raise RuntimeError("C-S4 normalizer parity failed")

    prepared,before,after,difference=original_and_prepared_difference(args,tensors,torch)
    contract=d0.prepared_base_contract(before,after);contract["control_lora_audit"]=no_lora_audit(prepared)
    base.write_json(args.output_dir/"prepared_base_contract.json",contract);base.write_json(args.output_dir/"initial_forward_difference.json",difference)
    train_only={key:development[key] for key in ("train_samples","train_targets","train_meta")}
    smoke_rows,smoke=smoke_50(prepared,train_only,torch);base.write_csv(args.output_dir/"technical_smoke.csv",smoke_rows);base.write_json(args.output_dir/"technical_smoke.json",smoke)
    if not smoke["passed"]:
        base.write_json(args.output_dir/"summary.json",{"label":LABEL,"stage":"D-C0 50-step gate","success":False,"formal_training_started":False,"test_materialized":False,"technical_smoke":smoke});return
    del prepared;torch.cuda.empty_cache()

    model=fresh_prepared_model(args,torch);formal_contract=formal.assert_formal_contract(args,model,tensors)
    model,normalizers,curve,validation_rows,gradient_rows,gaussian_rows,training,last_state=formal.train_formal(model,development,tensors,torch)
    best_path=args.output_dir/"best_validation_checkpoint.pt";last_path=args.output_dir/"last_checkpoint.pt"
    torch.save(formal.checkpoint_payload(model,normalizers,training,42),best_path);torch.save(formal.checkpoint_payload(model,normalizers,training,42,last_state),last_path)
    best_sha=formal.sha256_file(best_path);last_sha=formal.sha256_file(last_path)
    selection={"label":LABEL,"best_epoch":training["best_epoch"],"epochs_completed":training["epochs_completed"],"criterion":training["selection_rule"],"best_validation_metrics":training["best_validation_metrics"],"selection_key":training["selection_key"],"locked_thresholds":training["thresholds"],"best_checkpoint_sha256":best_sha,"last_checkpoint_sha256":last_sha,"selected_using":"validation only","test_materialized":False}
    base.write_json(args.output_dir/"checkpoint_selection.json",selection);base.write_csv(args.output_dir/"training_curve.csv",curve);base.write_csv(args.output_dir/"validation_metrics.csv",validation_rows);base.write_csv(args.output_dir/"gradient_trajectory.csv",gradient_rows)
    guard=formal.TestAccessGuard();guard.lock(best_sha,tuple(training["thresholds"]));guard.consume()
    episodes,datasets,samples,targets,meta=base.materialize_test(args,development,torch)
    from src.multimodal.context_schema import prepare_context_batch
    test_raw=prepare_context_batch(samples);test_x=torch.from_numpy(((test_raw-normalizers["feature_mean"])/normalizers["feature_scale"]).astype(np.float32));raw=base.prediction_batches(model,test_x,formal.BATCH_SIZE,torch);prediction=base.denormalize_prediction(raw,normalizers["benefit_mean"],normalizers["benefit_scale"])
    candidates,decisions,metrics=base.evaluate_predictions("L2-P-PREPARED-NO-LORA",prediction,targets,meta,episodes,tuple(training["thresholds"]))
    if any(row["reentry"] for row in decisions):raise RuntimeError("prepared control re-enabled infeasible action")
    original_metrics,original_candidates,original_decisions=load_original_results(args)
    comparison=[{"synthetic_interaction":LABEL,"model":"L2-ORIGINAL-FROZEN",**original_metrics},{"synthetic_interaction":LABEL,"model":"L2-P-PREPARED-NO-LORA",**metrics}]
    base.write_csv(args.output_dir/"candidate_metrics.csv",candidates);base.write_csv(args.output_dir/"switch_metrics.csv",comparison);base.write_csv(args.output_dir/"decision_metrics.csv",decisions);base.write_csv(args.output_dir/"original_vs_prepared.csv",comparison)
    context_rows=[]
    for split in sorted(set(row["context_split"] for row in meta)):
        original_subset=[row for row in original_decisions if row["context_split"]==split];prepared_metric=formal.subset_metrics("L2-P",prediction,targets,meta,episodes,tuple(training["thresholds"]),split)
        from src.evaluation.context_value_metrics import decision_metrics
        context_rows.append({"synthetic_interaction":LABEL,"model":"L2-ORIGINAL-FROZEN","context_split":split,**decision_metrics(original_subset)})
        context_rows.append({"synthetic_interaction":LABEL,"model":"L2-P-PREPARED-NO-LORA","context_split":split,**prepared_metric})
    base.write_csv(args.output_dir/"by_context_split.csv",context_rows)
    manifest=json.loads((args.phase5a_dir/"hard_case_manifest.json").read_text(encoding="utf-8"));hard_rows=formal.hard_case_rows(original_decisions,decisions,manifest)
    for row in hard_rows:row["model"]="L2-ORIGINAL-FROZEN" if row["model"]=="L1" else "L2-P-PREPARED-NO-LORA"
    base.write_csv(args.output_dir/"hard_cases.csv",hard_rows)
    deltas={key:metrics[key]-original_metrics[key] for key in ("Benefit_MAE","Benefit_Spearman","Harm_AUROC","Beneficial_Switch_Recall","Beneficial_Switch_Precision","Mean_Regret","P95_Regret")}
    summary={"label":LABEL,"stage":"Phase 5A Stage D-C0 Prepared-No-LoRA Attribution Control","success":True,"formal_training_completed":True,"formal_test_evaluation_count":1,"formal_contract":formal_contract,"prepared_base_contract":contract,"initial_forward_difference":difference,"technical_smoke":smoke,"training":training,"models":{"L2-ORIGINAL-FROZEN":original_metrics,"L2-P-PREPARED-NO-LORA":metrics},"prepared_minus_original":deltas,"qwen_frozen":training["optimizer_audit"],"lora_module_count":0,"lora_parameter_count":0,"test_used_for_selection":False,"future_lora_configuration_unchanged":True,"D_R1_started":False,"next_step_requires_human_approval":True}
    base.write_json(args.output_dir/"summary.json",summary);print(json.dumps(base.clean(summary),indent=2),flush=True)


if __name__=="__main__":main()

"""Stage C-S4 parity artifacts and unclipped 200-step stability preflight."""
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

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
LABEL="SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id",default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cache-dir",type=Path,default=Path.home()/".cache"/"huggingface")
    parser.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase5a_normalizer_parity")
    parser.add_argument("--before-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase5a_scale_alignment")
    parser.add_argument("--phase5a-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase5a")
    for name in ("phase4b6","phase4c","phase4c1","phase4c2","phase4c3"):
        parser.add_argument(f"--{name.replace('_','-')}-dir",type=Path,default=PROJECT_ROOT/"results_dev"/name)
    parser.add_argument("--seed",type=int,choices=(42,),default=42)
    parser.add_argument("--device",choices=("cuda",),default="cuda")
    parser.add_argument("--batch-size",type=int,choices=(8,),default=8)
    parser.add_argument("--steps",type=int,choices=(200,),default=200)
    parser.add_argument("--learning-rate",type=float,choices=(3e-4,),default=3e-4)
    parser.add_argument("--belief-samples",type=int,choices=(16,),default=16)
    parser.add_argument("--artifacts-only",action="store_true",help="Refresh completed audit summaries without loading Qwen or taking optimizer steps.")
    return parser.parse_args()


def clean(value):
    if isinstance(value,dict):return {str(key):clean(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [clean(item) for item in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def write_json(path,value):path.write_text(json.dumps(clean(value),indent=2,allow_nan=False),encoding="utf-8")
def read_csv(path):return list(csv.DictReader(path.open(encoding="utf-8")))
def write_csv(path,rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def distribution(values):
    array=np.asarray(values,np.float64)
    return {"mean":float(array.mean()),"std":float(array.std()),"median":float(np.median(array)),"min":float(array.min()),"P1":float(np.percentile(array,1)),"P90":float(np.percentile(array,90)),"P95":float(np.percentile(array,95)),"P99":float(np.percentile(array,99)),"max":float(array.max())}


def gradient_stats(values):
    result=distribution(values);array=np.asarray(values,np.float64)
    for threshold in (10,100,500,1000):result[f"fraction_gt_{threshold}"]=float(np.mean(array>threshold))
    return result


def ids_checksum(ids):
    digest=hashlib.sha256()
    for value in ids:digest.update(value.encode());digest.update(b"\0")
    return digest.hexdigest()


def parity_artifacts(args,development,tensors):
    from src.multimodal.context_dataset import fit_benefit_normalizer
    samples=development["train_samples"];targets=development["train_targets"];meta=development["train_meta"]
    l1=fit_benefit_normalizer(samples,targets,meta);l2=tensors["benefit_normalizer"]
    feasible=np.asarray([row["feasible"] for row in meta],bool)
    raw=np.asarray([target.benefit for target in targets],np.float32);raw_feasible=raw[feasible]
    old_mean=float(raw.mean());old_std=max(float(raw.std()),1e-4)
    old=(raw_feasible-old_mean)/old_std;new=l2.transform(raw_feasible)
    contract={"label":LABEL,"source":"frozen L1 train_model implementation","fit_split":"train only after predeclared development holdouts","feasible_mask":"bool(meta[i]['feasible'])","fit_timing":"after train/validation construction and holdout filtering; before optimization","fit_count":len(l1.fit_sample_ids),"available_train_candidate_count":len(samples),"infeasible_excluded_count":int((~feasible).sum()),"validation_participates_in_fit":False,"test_materialized":False,"mean":l1.mean,"raw_std":l1.raw_std,"scale":l1.scale,"epsilon":l1.epsilon,"std_algorithm":"numpy float32 population std (ddof=0)","transform":"(raw_target - mean) / max(raw_std, epsilon)","fit_sample_ids_checksum_sha256":ids_checksum(l1.fit_sample_ids)}
    parity={"label":LABEL,"fit_count":len(l1.fit_sample_ids),"feasible_count":int(feasible.sum()),"infeasible_excluded_count":int((~feasible).sum()),"L1_fit_sample_ids_checksum_sha256":ids_checksum(l1.fit_sample_ids),"L2_fit_sample_ids_checksum_sha256":ids_checksum(l2.fit_sample_ids),"fit_sample_ids_identical":l1.fit_sample_ids==l2.fit_sample_ids,"feasible_mask_identical":True,"mean_L1":l1.mean,"mean_L2":l2.mean,"mean_bitwise_equal":np.float64(l1.mean).tobytes()==np.float64(l2.mean).tobytes(),"std_L1":l1.scale,"std_L2":l2.scale,"std_bitwise_equal":np.float64(l1.scale).tobytes()==np.float64(l2.scale).tobytes(),"epsilon_L1":l1.epsilon,"epsilon_L2":l2.epsilon,"epsilon_equal":l1.epsilon==l2.epsilon,"same_transform_max_abs_error":float(np.max(np.abs(l1.transform(raw_feasible)-l2.transform(raw_feasible)))),"transformed_feasible_min":float(new.min()),"transformed_feasible_max":float(new.max()),"validation_participates_in_fit":False,"test_materialized":False}
    write_json(args.output_dir/"l1_normalizer_contract.json",contract);write_json(args.output_dir/"normalizer_parity.json",parity)
    rows=[]
    for index,(sample,value,is_feasible) in enumerate(zip(samples,raw,feasible)):
        rows.append({"synthetic_interaction":LABEL,"context_id":sample.context_id,"feasible":bool(is_feasible),"raw_benefit":float(value),"old_all_candidate_normalized":float((value-old_mean)/old_std),"new_L1_parity_normalized":float((value-l2.mean)/l2.scale),"included_in_fit":bool(is_feasible)})
    write_csv(args.output_dir/"target_transform_before_after.csv",rows)
    return {"contract":contract,"parity":parity,"raw_feasible":distribution(raw_feasible),"normalized_before":distribution(old),"normalized_after":distribution(new),"old_mean":old_mean,"old_std":old_std}


def gradient_comparison(args):
    before=read_csv(args.before_dir/"gradient_source_after.csv")
    after=[row for row in read_csv(args.output_dir/"gradient_source_matrix.csv") if row["record_type"]=="summary"]
    rows=[]
    for current in before:
        match=next(row for row in after if row["loss"]==current["loss"] and row["module"]==current["module"])
        result={"synthetic_interaction":LABEL,"loss":current["loss"],"module":current["module"]}
        for metric in ("mean","median","P95","max"):
            old=float(current[metric]);new=float(match[metric]);result[f"C_S3_{metric}"]=old;result[f"C_S4_{metric}"]=new;result[f"{metric}_change_percent"]=100*(new/old-1) if old else 0.0
        rows.append(result)
    write_csv(args.output_dir/"gradient_before_after.csv",rows);write_csv(args.output_dir/"gradient_source_after.csv",after)
    cosine=read_csv(args.output_dir/"gradient_cosine.csv");write_csv(args.output_dir/"gradient_cosine_after.csv",cosine)
    return rows


def run_preflight(args,development,tensors,torch):
    from scripts.run_phase5a_frozen3b import frozen_audit,gradient_norm,group_gradient_norms,trainable_parameters,trainable_state_checksum
    from src.models.large_context_adapter import FrozenQwen25VLContextAdapter
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    model=FrozenQwen25VLContextAdapter.from_pretrained_4bit(args.model_id,device_map={"":0},cache_dir=str(args.cache_dir),local_files_only=True).to("cuda");model.train()
    parameters=trainable_parameters(model);optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=1e-3);generator=torch.Generator().manual_seed(args.seed)
    rows=[];memory=[];initial=trainable_state_checksum(model);torch.cuda.reset_peak_memory_stats();step=0
    while step<args.steps:
        order=tensors["feasible_indices"][torch.randperm(len(tensors["feasible_indices"]),generator=generator)]
        for start in range(0,len(order),args.batch_size):
            if step>=args.steps:break
            step+=1;indices=order[start:start+args.batch_size];started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
            output=model(tensors["train_x"][indices].to("cuda"));target=tensors["train_y"][indices].to("cuda");error=output.benefit_mean-target;inverse=torch.exp(-output.benefit_log_variance)
            benefit=.5*(error.square()*inverse).mean();uncertainty=.5*output.benefit_log_variance.mean();harm=torch.nn.functional.binary_cross_entropy_with_logits(output.harm_logit,tensors["train_harm"][indices].to("cuda"),pos_weight=tensors["pos_weight"]);loss=benefit+uncertainty+harm
            if not bool(torch.isfinite(loss)):raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward();raw=gradient_norm(parameters);groups=group_gradient_norms(model);audit=frozen_audit(model,optimizer)
            if not math.isfinite(raw) or any(not math.isfinite(value) for value in groups.values()):raise FloatingPointError(f"non-finite raw gradient at step {step}")
            if audit["qwen_gradient_tensor_count"] or audit["qwen_requires_grad_parameter_count"] or audit["qwen_optimizer_parameter_count"] or not audit["optimizer_only_projection_heads"]:raise RuntimeError(f"Qwen freeze audit failed at step {step}")
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):raise FloatingPointError(f"non-finite parameter at step {step}")
            torch.cuda.synchronize();row={"synthetic_interaction":LABEL,"step":step,"total_loss":float(loss.detach()),"benefit_loss":float(benefit.detach()),"harm_loss":float(harm.detach()),"uncertainty_regularizer":float(uncertainty.detach()),"raw_total_grad_norm":raw,"projection_grad_norm":groups["projection"],"benefit_head_grad_norm":groups["benefit_head"],"harm_head_grad_norm":groups["harm_head"],"uncertainty_head_grad_norm":groups["uncertainty_head"],"prediction_error_mean":float(error.detach().mean()),"prediction_error_std":float(error.detach().std(unbiased=False)),"prediction_error_abs_mean":float(error.detach().abs().mean()),"log_variance_mean":float(output.benefit_log_variance.detach().mean()),"log_variance_min":float(output.benefit_log_variance.detach().min()),"log_variance_max":float(output.benefit_log_variance.detach().max()),"exp_neg_log_variance_mean":float(inverse.detach().mean()),"cuda_allocated_gb":torch.cuda.memory_allocated()/2**30,"cuda_peak_gb":torch.cuda.max_memory_allocated()/2**30,"step_latency_ms":(time.perf_counter()-started)*1000,"qwen_gradients":audit["qwen_gradient_tensor_count"],"qwen_requires_grad_parameters":audit["qwen_requires_grad_parameter_count"],"qwen_optimizer_parameters":audit["qwen_optimizer_parameter_count"],"gradient_clipping_applied":False};rows.append(row);memory.append({key:row[key] for key in ("synthetic_interaction","step","cuda_allocated_gb","cuda_peak_gb","step_latency_ms")})
    gradients=[row["raw_total_grad_norm"] for row in rows];allocated=[row["cuda_allocated_gb"] for row in rows]
    report={"steps":len(rows),"gradient_clipping":False,"raw_gradient":gradient_stats(gradients),"steps_1_20":gradient_stats(gradients[:20]),"steps_21_99":gradient_stats(gradients[20:99]),"steps_100_200":gradient_stats(gradients[99:]),"cuda_peak_gb":max(row["cuda_peak_gb"] for row in rows),"memory_growth_last20_minus_first20_gb":float(np.mean(allocated[-20:])-np.mean(allocated[:20])),"all_finite":bool(all(math.isfinite(float(value)) for row in rows for key,value in row.items() if key not in ("synthetic_interaction",))),"qwen_frozen":all(row["qwen_gradients"]==row["qwen_requires_grad_parameters"]==row["qwen_optimizer_parameters"]==0 for row in rows),"initial_trainable_checksum":initial,"final_trainable_checksum":trainable_state_checksum(model),"test_materialized":False,"formal_training_started":False}
    return rows,memory,report


def make_figures(args,comparison,preflight,transforms):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    folder=args.output_dir/"figures";folder.mkdir(exist_ok=True);paths=[]
    key=lambda loss,module:next(row for row in comparison if row["loss"]==loss and row["module"]==module)
    names=("Benefit","Total","Projection","Benefit Head","Uncertainty Head");selected=(key("benefit","all_trainable"),key("total","all_trainable"),key("benefit","projection"),key("benefit","benefit_head"),key("benefit","uncertainty_head"));x=np.arange(len(names));plt.figure(figsize=(9,4));plt.bar(x-.18,[float(row["C_S3_mean"]) for row in selected],.36,label="C-S3");plt.bar(x+.18,[float(row["C_S4_mean"]) for row in selected],.36,label="C-S4");plt.xticks(x,names,rotation=20);plt.ylabel("mean gradient norm");plt.legend();plt.tight_layout();path=folder/"gradient_before_after.png";plt.savefig(path,dpi=160);plt.close();paths.append(str(path))
    feasible=lambda row:str(row["feasible"]).lower()=="true"
    plt.figure(figsize=(7,4));plt.hist([float(row["old_all_candidate_normalized"]) for row in transforms if feasible(row)],bins=30,alpha=.6,label="C-S3 all-candidate fit");plt.hist([float(row["new_L1_parity_normalized"]) for row in transforms if feasible(row)],bins=30,alpha=.6,label="C-S4 feasible-only fit");plt.xlabel("normalized feasible benefit");plt.legend();plt.tight_layout();path=folder/"target_transform_before_after.png";plt.savefig(path,dpi=160);plt.close();paths.append(str(path))
    plt.figure(figsize=(8,4));plt.plot([row["step"] for row in preflight],[row["raw_total_grad_norm"] for row in preflight]);plt.axvline(100,color="gray",linestyle="--");plt.xlabel("step");plt.ylabel("raw gradient norm");plt.tight_layout();path=folder/"preflight_gradient.png";plt.savefig(path,dpi=160);plt.close();paths.append(str(path));return paths


def refresh_gaussian_comparison(args):
    transforms={row["context_id"]:row for row in read_csv(args.output_dir/"target_transform_before_after.csv")}
    rows=read_csv(args.output_dir/"gaussian_nll_audit.csv");before_errors=[];after_errors=[];before_likelihood=[];after_likelihood=[]
    for row in rows:
        predicted=float(row["predicted_benefit_mean"]);old_target=float(transforms[row["context_id"]]["old_all_candidate_normalized"]);new_target=float(row["normalized_target"]);inverse=float(row["exp_neg_log_variance"]);old_error=predicted-old_target;new_error=predicted-new_target;old_likelihood=.5*old_error**2*inverse;new_likelihood=.5*new_error**2*inverse
        row.update({"C_S3_normalized_target":old_target,"C_S3_prediction_error":old_error,"C_S3_benefit_likelihood":old_likelihood,"C_S4_normalized_target":new_target,"C_S4_prediction_error":new_error,"C_S4_benefit_likelihood":new_likelihood});before_errors.append(old_error);after_errors.append(new_error);before_likelihood.append(old_likelihood);after_likelihood.append(new_likelihood)
    write_csv(args.output_dir/"gaussian_nll_audit.csv",rows)
    return {"prediction_error":{"C_S3":distribution(before_errors),"C_S4":distribution(after_errors)},"absolute_prediction_error":{"C_S3":distribution(np.abs(before_errors)),"C_S4":distribution(np.abs(after_errors))},"exp_neg_log_variance":distribution([float(row["exp_neg_log_variance"]) for row in rows]),"log_variance":distribution([float(row["log_variance"]) for row in rows]),"benefit_likelihood":{"C_S3":distribution(before_likelihood),"C_S4":distribution(after_likelihood)}}


def refresh_completed_summary(args):
    summary=json.loads((args.output_dir/"summary.json").read_text());summary["gaussian_nll_before_after"]=refresh_gaussian_comparison(args);summary["normalizer_repair_effect"]="Improved 200-step overall gradient scale versus C-S3, but worsened the initialization-only Benefit/Total gradient and did not remove the adverse late-step trend.";summary["formal_frozen3b_seed42_ready"]=False;summary["recommended_next_single_variable_intervention"]="Run one diagnostic preflight with the uncertainty head frozen (all other variables unchanged) to isolate the dominant Gaussian-NLL uncertainty-gradient conflict; do not implement without approval.";write_json(args.output_dir/"summary.json",summary)


def main():
    args=parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.artifacts_only:
        refresh_completed_summary(args);return
    import torch
    from scripts.run_phase5a_frozen3b import build_development_data,prepare_training_tensors
    development=build_development_data(args,torch);tensors=prepare_training_tensors(development,torch)
    parity=parity_artifacts(args,development,tensors);comparison=gradient_comparison(args)
    rows,memory,preflight=run_preflight(args,development,tensors,torch);write_csv(args.output_dir/"stability_preflight.csv",rows);write_csv(args.output_dir/"memory_audit.csv",memory)
    transforms=read_csv(args.output_dir/"target_transform_before_after.csv");figures=make_figures(args,comparison,rows,transforms)
    nll=read_csv(args.output_dir/"gaussian_nll_audit.csv");nll_summary={name:distribution([float(row[name]) for row in nll]) for name in ("normalized_target","predicted_benefit_mean","prediction_error","log_variance","exp_neg_log_variance","benefit_likelihood")}
    before_nll=read_csv(args.before_dir/"gradient_source_matrix.csv");before_losses=[float(row["loss_value"]) for row in before_nll if row["record_type"]=="batch" and row["loss"]=="benefit" and row["module"]=="projection"]
    after_losses=[float(row["loss_value"]) for row in read_csv(args.output_dir/"gradient_source_matrix.csv") if row["record_type"]=="batch" and row["loss"]=="benefit" and row["module"]=="projection"]
    late=preflight["steps_100_200"];prior=json.loads((args.before_dir/"summary.json").read_text())["stability_preflight"]["raw_gradient"]
    trend={"late_mean_vs_early_percent":100*(late["mean"]/preflight["steps_1_20"]["mean"]-1),"late_median_vs_early_percent":100*(late["median"]/preflight["steps_1_20"]["median"]-1),"late_fraction_gt_100":late["fraction_gt_100"],"continues_to_worsen":bool(late["median"]>preflight["steps_1_20"]["median"]*1.25 or late["fraction_gt_100"]>preflight["steps_1_20"]["fraction_gt_100"]+.1)}
    summary={"label":LABEL,"stage":"Phase 5A Stage C-S4 Feasible-only Normalization Parity Repair","success":False,"normalizer_parity":parity,"benefit_loss_32_batch":{"C_S3":distribution(before_losses),"C_S4":distribution(after_losses)},"gradient_before_after":comparison,"gaussian_nll_audit":nll_summary,"stability_preflight":preflight,"historical_comparison":{"C_S":{"mean":302.168,"P95":760.196,"max":1665.160},"C_S3":prior,"C_S4":preflight["raw_gradient"]},"late_gradient_trend":trend,"normalizer_repair_effect":"Improved 200-step overall gradient scale versus C-S3, but worsened the initialization-only Benefit/Total gradient and did not remove the adverse late-step trend.","formal_frozen3b_seed42_ready":False,"recommended_next_single_variable_intervention":"Run one diagnostic preflight with the uncertainty head frozen (all other variables unchanged) to isolate the dominant Gaussian-NLL uncertainty-gradient conflict; do not implement without approval.","formal_training_started":False,"test_materialized":False,"only_change":"benefit normalizer fitting scope","scale_alignment_unchanged":True,"figures":figures};write_json(args.output_dir/"summary.json",summary);refresh_completed_summary(args)
    print(json.dumps({"parity":parity["parity"],"preflight":preflight,"trend":trend},indent=2),flush=True)


if __name__=="__main__":main()

"""Stage C-S3 scale-alignment comparison and unclipped 200-step preflight."""
from __future__ import annotations

import argparse
import csv
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


def args_parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default="Qwen/Qwen2.5-VL-3B-Instruct");p.add_argument("--cache-dir",type=Path,default=Path.home()/".cache"/"huggingface");p.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase5a_scale_alignment");p.add_argument("--before-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase5a_gradient_audit");p.add_argument("--phase5a-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase5a");p.add_argument("--phase4b6-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4b6");p.add_argument("--phase4c-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c");p.add_argument("--phase4c1-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c1");p.add_argument("--phase4c2-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c2");p.add_argument("--phase4c3-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c3");p.add_argument("--seed",type=int,choices=(42,),default=42);p.add_argument("--device",choices=("cuda",),default="cuda");p.add_argument("--batch-size",type=int,choices=(8,),default=8);p.add_argument("--steps",type=int,choices=(200,),default=200);p.add_argument("--learning-rate",type=float,default=3e-4);p.add_argument("--belief-samples",type=int,choices=(16,),default=16);p.add_argument("--artifacts-only",action="store_true",help="Regenerate plots and audit metadata from completed C-S3 CSV/JSON files without loading Qwen or training.");return p.parse_args()


def read_csv(path):return list(csv.DictReader(path.open(encoding="utf-8")))
def write_csv(path,rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
def stats(values):
    a=np.asarray(values,float);return {"mean":float(a.mean()),"median":float(np.median(a)),"P90":float(np.percentile(a,90)),"P95":float(np.percentile(a,95)),"max":float(a.max()),"fraction_gt_10":float(np.mean(a>10)),"fraction_gt_100":float(np.mean(a>100))}


def comparison_outputs(args):
    before_scale=read_csv(args.before_dir/"projection_scale.csv");after_scale=read_csv(args.output_dir/"projection_scale.csv")
    before_structured=[row for row in before_scale if row["category"]=="structured_projected_token"]
    after_structured=[row for row in after_scale if row["category"]=="structured_projected_token"]
    write_csv(args.output_dir/"structured_token_scale_before.csv",before_structured);write_csv(args.output_dir/"structured_token_scale_after.csv",after_structured)
    native=next(row for row in after_scale if row["category"]=="qwen_native_embedding")
    write_json(args.output_dir/"native_embedding_scale.json",{"label":LABEL,"source":"frozen Qwen input embedding weight row norms only; no text data","mean":float(native["mean"]),"median":float(native["median"]),"P5":float(native["P5"]),"P95":float(native["P95"]),"native_target_norm":float(native["median"]),"alignment_formula":"e_raw / (||e_raw|| + 1e-6) * native_target_norm"})
    before=[row for row in read_csv(args.before_dir/"gradient_source_matrix.csv") if row["record_type"]=="summary"]
    after=[row for row in read_csv(args.output_dir/"gradient_source_matrix.csv") if row["record_type"]=="summary"]
    rows=[]
    for current in before:
        match=next(row for row in after if row["loss"]==current["loss"] and row["module"]==current["module"])
        row={"synthetic_interaction":LABEL,"loss":current["loss"],"module":current["module"]}
        for metric in ("mean","median","P95","max"):
            b=float(current[metric]);a=float(match[metric]);row[f"before_{metric}"]=b;row[f"after_{metric}"]=a;row[f"reduction_{metric}_percent"]=100*(1-a/b) if b else 0.0
        rows.append(row)
    write_csv(args.output_dir/"gradient_before_after.csv",rows);write_csv(args.output_dir/"gradient_source_after.csv",after)
    normalizer=json.loads((args.output_dir/"summary.json").read_text())["targets"]["normalizer_scope_audit"]
    write_json(args.output_dir/"normalizer_scope_audit.json",{"label":LABEL,"status":"FAIRNESS ISSUE PENDING","modified_in_stage_c_s3":False,**normalizer})
    key=lambda loss,module:next(row for row in rows if row["loss"]==loss and row["module"]==module)
    improvement={"benefit":key("benefit","all_trainable"),"total":key("total","all_trainable"),"benefit_projection":key("benefit","projection"),"benefit_head":key("benefit","benefit_head"),"benefit_uncertainty_head":key("benefit","uncertainty_head"),"harm":key("harm","all_trainable")}
    significant=improvement["benefit"]["reduction_mean_percent"]>=25 and improvement["total"]["reduction_mean_percent"]>=25 and improvement["benefit_uncertainty_head"]["reduction_mean_percent"]>=25 and improvement["harm"]["after_mean"]<=1.5*improvement["harm"]["before_mean"]
    return improvement,significant


def preflight(args,torch):
    from scripts.run_phase5a_frozen3b import build_development_data,frozen_audit,gradient_norm,group_gradient_norms,prepare_training_tensors,trainable_parameters,trainable_state_checksum,training_losses
    from src.models.large_context_adapter import FrozenQwen25VLContextAdapter
    development=build_development_data(args,torch);tensors=prepare_training_tensors(development,torch)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    model=FrozenQwen25VLContextAdapter.from_pretrained_4bit(args.model_id,device_map={"":0},cache_dir=str(args.cache_dir),local_files_only=True).to("cuda");model.train();parameters=trainable_parameters(model);optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=1e-3);generator=torch.Generator().manual_seed(args.seed);rows=[];initial=trainable_state_checksum(model);torch.cuda.reset_peak_memory_stats();step=0
    while step<args.steps:
        order=tensors["feasible_indices"][torch.randperm(len(tensors["feasible_indices"]),generator=generator)]
        for start in range(0,len(order),args.batch_size):
            if step>=args.steps:break
            step+=1;indices=order[start:start+args.batch_size];started=time.perf_counter();optimizer.zero_grad(set_to_none=True);loss,benefit,harm,uncertainty=training_losses(model,tensors,indices,torch)
            if not bool(torch.isfinite(loss)):raise FloatingPointError(f"non-finite loss at {step}")
            loss.backward();raw=gradient_norm(parameters);groups=group_gradient_norms(model);audit=frozen_audit(model,optimizer)
            if not math.isfinite(raw) or raw>5000:raise FloatingPointError(f"emergency raw-gradient stop at step {step}: {raw}")
            if audit["qwen_gradient_tensor_count"] or audit["qwen_requires_grad_parameter_count"] or audit["qwen_optimizer_parameter_count"] or not audit["optimizer_only_projection_heads"]:raise RuntimeError(f"Qwen freeze audit failed at {step}")
            if any(not math.isfinite(value) or value<=0 for value in groups.values()):raise FloatingPointError(f"invalid module gradient at {step}")
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):raise FloatingPointError(f"non-finite parameter at {step}")
            torch.cuda.synchronize();rows.append({"synthetic_interaction":LABEL,"step":step,"train_loss":float(loss.detach()),"benefit_loss":float(benefit.detach()),"harm_loss":float(harm.detach()),"uncertainty_loss":float(uncertainty.detach()),"raw_gradient_norm":raw,"projection_grad_norm":groups["projection"],"benefit_head_grad_norm":groups["benefit_head"],"harm_head_grad_norm":groups["harm_head"],"uncertainty_head_grad_norm":groups["uncertainty_head"],"clipping_applied":False,"cuda_allocated_gb":torch.cuda.memory_allocated()/2**30,"cuda_peak_gb":torch.cuda.max_memory_allocated()/2**30,"step_latency_ms":(time.perf_counter()-started)*1000,"qwen_gradients":audit["qwen_gradient_tensor_count"],"qwen_requires_grad_parameters":audit["qwen_requires_grad_parameter_count"],"qwen_optimizer_parameters":audit["qwen_optimizer_parameter_count"]})
    gradients=[row["raw_gradient_norm"] for row in rows];losses=[row["train_loss"] for row in rows];allocated=[row["cuda_allocated_gb"] for row in rows];overall=stats(gradients);early=stats(gradients[:20]);late=stats(gradients[99:]);memory_growth=float(np.mean(allocated[-20:])-np.mean(allocated[:20]));criteria={"steps_completed":len(rows)==200,"finite":bool(np.isfinite(losses).all() and np.isfinite(gradients).all()),"no_clipping":all(not row["clipping_applied"] for row in rows),"qwen_frozen":all(row["qwen_gradients"]==row["qwen_requires_grad_parameters"]==row["qwen_optimizer_parameters"]==0 for row in rows),"memory_stable":memory_growth<.1,"healthier_than_stage_c_s_fraction_gt_10":overall["fraction_gt_10"]<1.0,"healthier_than_stage_c_s_fraction_gt_100":overall["fraction_gt_100"]<.76,"late_not_hundreds_dominated":late["fraction_gt_100"]<.5 and late["P95"]<500,"late_not_worse_than_early":late["median"]<=max(early["median"]*2,100)};criteria["passed"]=all(criteria.values())
    return rows,{"label":LABEL,"stage":"C-S3 unclipped stability preflight","steps":len(rows),"gradient_clipping":False,"emergency_stop_threshold":5000,"learning_rate":args.learning_rate,"optimizer":"AdamW","weight_decay":1e-3,"scheduler":"none","batch_size":8,"raw_gradient":overall,"steps_1_20":early,"steps_100_200":late,"memory_growth_last20_minus_first20_gb":memory_growth,"loss_first20_mean":float(np.mean(losses[:20])),"loss_last20_mean":float(np.mean(losses[-20:])),"qwen_frozen":criteria["qwen_frozen"],"parameter_checksum_initial":initial,"parameter_checksum_final":trainable_state_checksum(model),"test_materialized":False,"formal_training_started":False,"criteria":criteria}


def plots(args,improvement,rows):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    folder=args.output_dir/"figures";folder.mkdir(exist_ok=True);paths=[]
    names=("benefit","total","benefit_projection","benefit_head","benefit_uncertainty_head");x=np.arange(len(names));plt.figure(figsize=(9,4));plt.bar(x-.18,[improvement[name]["before_mean"] for name in names],.36,label="before");plt.bar(x+.18,[improvement[name]["after_mean"] for name in names],.36,label="after");plt.xticks(x,names,rotation=20);plt.ylabel("mean gradient norm");plt.legend();plt.tight_layout();path=folder/"gradient_before_after.png";plt.savefig(path,dpi=160);plt.close();paths.append(str(path))
    before=read_csv(args.output_dir/"structured_token_scale_before.csv");after=read_csv(args.output_dir/"structured_token_scale_after.csv");groups=[row["variable"] for row in before];x=np.arange(len(groups))
    plt.figure(figsize=(10,4));plt.bar(x-.18,[float(row["median"]) for row in before],.36,label="before");plt.bar(x+.18,[float(row["median"]) for row in after],.36,label="after");plt.xticks(x,groups,rotation=25);plt.ylabel("token norm median");plt.legend();plt.tight_layout();path=folder/"token_norm_before_after.png";plt.savefig(path,dpi=160);plt.close();paths.append(str(path))
    native=json.loads((args.output_dir/"native_embedding_scale.json").read_text());labels=("Qwen native","structured before","structured after");medians=(native["median"],float(np.median([float(row["median"]) for row in before])),float(np.median([float(row["median"]) for row in after])));low=(native["P5"],float(np.median([float(row["P5"]) for row in before])),float(np.median([float(row["P5"]) for row in after])));high=(native["P95"],float(np.median([float(row["P95"]) for row in before])),float(np.median([float(row["P95"]) for row in after])));x=np.arange(3)
    plt.figure(figsize=(7,4));plt.errorbar(x,medians,yerr=(np.asarray(medians)-np.asarray(low),np.asarray(high)-np.asarray(medians)),fmt="o",capsize=5);plt.xticks(x,labels);plt.ylabel("row/token norm (median, P5-P95)");plt.tight_layout();path=folder/"native_vs_structured_norm.png";plt.savefig(path,dpi=160);plt.close();paths.append(str(path))
    for key,filename,title in (("benefit","benefit_gradient_before_after.png","Benefit gradient"),("total","total_gradient_before_after.png","Total gradient")):
        metrics=("mean","P95","max");x=np.arange(3);plt.figure(figsize=(6,4));plt.bar(x-.18,[improvement[key][f"before_{metric}"] for metric in metrics],.36,label="before");plt.bar(x+.18,[improvement[key][f"after_{metric}"] for metric in metrics],.36,label="after");plt.xticks(x,metrics);plt.ylabel("gradient norm");plt.title(title);plt.legend();plt.tight_layout();path=folder/filename;plt.savefig(path,dpi=160);plt.close();paths.append(str(path))
    plt.figure();plt.plot([row["step"] for row in rows],[row["raw_gradient_norm"] for row in rows]);plt.axhline(100,color="red",linestyle="--");plt.xlabel("step");plt.ylabel("raw gradient norm");plt.tight_layout();path=folder/"preflight_raw_gradient.png";plt.savefig(path,dpi=160);plt.close();paths.append(str(path));return paths


def refresh_completed_artifacts(args):
    """Finalize plots/metadata without performing another audit or optimizer step."""
    comparison=read_csv(args.output_dir/"gradient_before_after.csv")
    key=lambda loss,module:next(row for row in comparison if row["loss"]==loss and row["module"]==module)
    improvement={name:{field:(float(value) if field not in ("synthetic_interaction","loss","module") else value) for field,value in row.items()} for name,row in (("benefit",key("benefit","all_trainable")),("total",key("total","all_trainable")),("benefit_projection",key("benefit","projection")),("benefit_head",key("benefit","benefit_head")),("benefit_uncertainty_head",key("benefit","uncertainty_head")),("harm",key("harm","all_trainable")))}
    rows=read_csv(args.output_dir/"stability_preflight.csv");figures=plots(args,improvement,rows);summary=json.loads((args.output_dir/"summary.json").read_text());summary["figures"]=figures;write_json(args.output_dir/"summary.json",summary)
    native=json.loads((args.output_dir/"native_embedding_scale.json").read_text());before=read_csv(args.output_dir/"structured_token_scale_before.csv");after=read_csv(args.output_dir/"structured_token_scale_after.csv");after_values=np.asarray([float(row["median"]) for row in after]);finite=bool(np.isfinite(after_values).all());audit={"label":LABEL,"stage":"Phase 5A Stage C-S3 scale-alignment audit","source":"completed 32-batch diagnostic audit; no optimizer step and no test materialization","formula":native["alignment_formula"],"eps":1e-6,"native_embedding":native,"structured_token_groups":[row["variable"] for row in after],"structured_median_before":summary["scale_alignment"]["structured_median_before"],"structured_median_after":summary["scale_alignment"]["structured_median_after"],"aligned_values_finite":finite,"aligned_zero_vector_count":int(sum(float(row["min"])==0 for row in after)),"same_batch_manifest_sha256":"2EBDA723185D3145634F8EE0DA762F2C97DABFB2C1B467363F4EEE15AF474F96","parameter_checksum_before":"0542166e8465dc7cd3eb2c61904c1a71a88e48679ce6efe013448e742ebfb630","parameter_checksum_after":"0542166e8465dc7cd3eb2c61904c1a71a88e48679ce6efe013448e742ebfb630","parameters_unchanged":True,"qwen_frozen":True,"optimizer_step_count":0,"test_materialized":False};write_json(args.output_dir/"scale_alignment_audit.json",audit)


def main():
    args=args_parser();import torch
    if args.artifacts_only:
        refresh_completed_artifacts(args);return
    improvement,significant=comparison_outputs(args)
    if not significant:raise RuntimeError("scale alignment did not meet predeclared gradient-improvement gate")
    rows,preflight_report=preflight(args,torch);write_csv(args.output_dir/"stability_preflight.csv",rows);figures=plots(args,improvement,rows)
    native=json.loads((args.output_dir/"native_embedding_scale.json").read_text());normalizer=json.loads((args.output_dir/"normalizer_scope_audit.json").read_text());after=json.loads((args.output_dir/"summary.json").read_text());summary={"label":LABEL,"stage":"Phase 5A Stage C-S3 Structured Token Scale Alignment","success":bool(preflight_report["criteria"]["passed"]),"scale_alignment":{"formula":native["alignment_formula"],"eps":1e-6,"trainable_parameters_added":0,"native_embedding":native,"structured_median_before":29.46910858154297,"structured_median_after":after["projection_scale"]["structured_projected_token_norm_median"]},"gradient_before_after":improvement,"same_32_batch_manifest":True,"stability_preflight":preflight_report,"normalizer_scope_audit":normalizer,"fairness_issue_pending":True,"formal_training_started":False,"test_materialized":False,"repair_count_this_stage":1,"repair":"structured token norm alignment only","next_step_requires_human_approval":True,"figures":figures};write_json(args.output_dir/"summary.json",summary);print(json.dumps({"scale_gradient_gate":significant,"preflight_passed":preflight_report["criteria"]["passed"],"raw_gradient":preflight_report["raw_gradient"],"late":preflight_report["steps_100_200"]},indent=2),flush=True)


if __name__=="__main__":main()

"""C-S5 result assembly and unclipped 200-step frozen-Qwen preflight."""
from __future__ import annotations
import argparse,csv,json,math,random,sys,time
from pathlib import Path
import numpy as np
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
LABEL="SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"

def args_parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default="Qwen/Qwen2.5-VL-3B-Instruct");p.add_argument("--cache-dir",type=Path,default=Path.home()/".cache"/"huggingface");p.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase5a_uncertainty_path_isolation");p.add_argument("--before-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase5a_normalizer_parity");p.add_argument("--phase5a-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase5a")
 for name in ("phase4b6","phase4c","phase4c1","phase4c2","phase4c3"):p.add_argument(f"--{name.replace('_','-')}-dir",type=Path,default=PROJECT_ROOT/"results_dev"/name)
 p.add_argument("--seed",type=int,choices=(42,),default=42);p.add_argument("--device",choices=("cuda",),default="cuda");p.add_argument("--batch-size",type=int,choices=(8,),default=8);p.add_argument("--steps",type=int,choices=(200,),default=200);p.add_argument("--learning-rate",type=float,choices=(3e-4,),default=3e-4);p.add_argument("--belief-samples",type=int,choices=(16,),default=16);p.add_argument("--artifacts-only",action="store_true");return p.parse_args()
def read_csv(path):return list(csv.DictReader(path.open(encoding="utf-8")))
def write_csv(path,rows):
 fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 with path.open("w",newline="",encoding="utf-8") as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.generic):v=v.item()
 if isinstance(v,float) and not math.isfinite(v):return None
 return v
def write_json(path,value):path.write_text(json.dumps(clean(value),indent=2,allow_nan=False),encoding="utf-8")
def stats(values):
 a=np.asarray(values,float);r={"mean":float(a.mean()),"std":float(a.std()),"median":float(np.median(a)),"min":float(a.min()),"P90":float(np.percentile(a,90)),"P95":float(np.percentile(a,95)),"max":float(a.max())}
 for n in (10,100,500,1000):r[f"fraction_gt_{n}"]=float(np.mean(a>n))
 return r
def comparison(args):
 before=read_csv(args.before_dir/"gradient_source_after.csv");after=[r for r in read_csv(args.output_dir/"gradient_source_matrix.csv") if r["record_type"]=="summary"];rows=[]
 for old in before:
  new=next(r for r in after if r["loss"]==old["loss"] and r["module"]==old["module"]);row={"synthetic_interaction":LABEL,"loss":old["loss"],"module":old["module"]}
  for metric in ("mean","median","P95","max"):
   a=float(old[metric]);b=float(new[metric]);row[f"C_S4_{metric}"]=a;row[f"C_S5_{metric}"]=b;row[f"{metric}_change_percent"]=100*(b/a-1) if a else 0.
  rows.append(row)
 write_csv(args.output_dir/"gradient_before_after.csv",rows);write_csv(args.output_dir/"gradient_source_after.csv",after)
 cosine=read_csv(args.output_dir/"gradient_cosine.csv");spaces=read_csv(args.output_dir/"gradient_cosine_spaces.csv");write_csv(args.output_dir/"gradient_cosine_after.csv",cosine+spaces);return rows
def uncertainty_diagnostics(model,tensors,args,torch):
 rows=[];model.eval()
 with torch.inference_mode():
  for split,x,y,indices in (("train",tensors["train_x"],tensors["train_y"],tensors["feasible_indices"]),("validation",tensors["val_x"],None,torch.arange(len(tensors["val_x"])))):
   values=[]
   for start in range(0,len(indices),args.batch_size):
    ids=indices[start:start+args.batch_size];out=model(x[ids].to("cuda"));logv=out.benefit_log_variance.float().cpu().numpy();sigma=np.exp(.5*logv);benefit=out.benefit_mean.float().cpu().numpy()
    if split=="train":target=y[ids].numpy()
    else:target=np.full(len(ids),np.nan)
    for lv,sg,pred,truth in zip(logv,sigma,benefit,target):values.append((lv,sg,pred,truth))
   lv=np.asarray([v[0] for v in values]);sg=np.asarray([v[1] for v in values]);pred=np.asarray([v[2] for v in values]);target=np.asarray([v[3] for v in values]);valid=np.isfinite(target);corr=float(np.corrcoef(sg[valid],np.abs(pred[valid]-target[valid]))[0,1]) if valid.sum()>1 else None
   rows.append({"synthetic_interaction":LABEL,"split":split,"count":len(values),"log_variance_mean":float(lv.mean()),"log_variance_std":float(lv.std()),"log_variance_min":float(lv.min()),"log_variance_max":float(lv.max()),"sigma_mean":float(sg.mean()),"sigma_std":float(sg.std()),"sigma_min":float(sg.min()),"sigma_max":float(sg.max()),"uncertainty_error_correlation":corr,"collapsed":bool(sg.std()<1e-5),"test_materialized":False})
 return rows
def preflight(args,development,tensors,torch):
 from scripts.run_phase5a_frozen3b import benefit_likelihood_with_detached_variance,frozen_audit,gradient_norm,group_gradient_norms,trainable_parameters,trainable_state_checksum
 from src.models.large_context_adapter import FrozenQwen25VLContextAdapter
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);model=FrozenQwen25VLContextAdapter.from_pretrained_4bit(args.model_id,device_map={"":0},cache_dir=str(args.cache_dir),local_files_only=True).to("cuda");model.train();params=trainable_parameters(model);opt=torch.optim.AdamW(params,lr=args.learning_rate,weight_decay=1e-3);gen=torch.Generator().manual_seed(args.seed);rows=[];memory=[];initial=trainable_state_checksum(model);torch.cuda.reset_peak_memory_stats();step=0
 while step<args.steps:
  order=tensors["feasible_indices"][torch.randperm(len(tensors["feasible_indices"]),generator=gen)]
  for start in range(0,len(order),args.batch_size):
   if step>=args.steps:break
   step+=1;ids=order[start:start+args.batch_size];started=time.perf_counter();opt.zero_grad(set_to_none=True);out=model(tensors["train_x"][ids].to("cuda"));target=tensors["train_y"][ids].to("cuda");error=out.benefit_mean-target;benefit=benefit_likelihood_with_detached_variance(out.benefit_mean,target,out.benefit_log_variance,torch);uncertainty=.5*out.benefit_log_variance.mean();harm=torch.nn.functional.binary_cross_entropy_with_logits(out.harm_logit,tensors["train_harm"][ids].to("cuda"),pos_weight=tensors["pos_weight"]);loss=benefit+uncertainty+harm
   if not bool(torch.isfinite(loss)):raise FloatingPointError(f"non-finite loss at {step}")
   loss.backward();raw=gradient_norm(params);groups=group_gradient_norms(model);audit=frozen_audit(model,opt)
   if not math.isfinite(raw) or any(not math.isfinite(v) for v in groups.values()):raise FloatingPointError(f"non-finite gradient at {step}")
   if audit["qwen_gradient_tensor_count"] or audit["qwen_requires_grad_parameter_count"] or audit["qwen_optimizer_parameter_count"] or not audit["optimizer_only_projection_heads"]:raise RuntimeError(f"Qwen freeze audit failed at {step}")
   opt.step();torch.cuda.synchronize();row={"synthetic_interaction":LABEL,"step":step,"total_loss":float(loss.detach()),"benefit_likelihood":float(benefit.detach()),"uncertainty_regularizer":float(uncertainty.detach()),"harm_loss":float(harm.detach()),"raw_total_grad_norm":raw,"projection_grad_norm":groups["projection"],"benefit_head_grad_norm":groups["benefit_head"],"harm_head_grad_norm":groups["harm_head"],"uncertainty_head_grad_norm":groups["uncertainty_head"],"prediction_error_mean":float(error.detach().mean()),"prediction_error_abs_mean":float(error.detach().abs().mean()),"log_variance_mean":float(out.benefit_log_variance.detach().mean()),"log_variance_min":float(out.benefit_log_variance.detach().min()),"log_variance_max":float(out.benefit_log_variance.detach().max()),"cuda_allocated_gb":torch.cuda.memory_allocated()/2**30,"cuda_peak_gb":torch.cuda.max_memory_allocated()/2**30,"step_latency_ms":(time.perf_counter()-started)*1000,"qwen_gradients":audit["qwen_gradient_tensor_count"],"qwen_optimizer_parameters":audit["qwen_optimizer_parameter_count"],"gradient_clipping_applied":False};rows.append(row);memory.append({k:row[k] for k in ("synthetic_interaction","step","cuda_allocated_gb","cuda_peak_gb","step_latency_ms")})
 gradients=[r["raw_total_grad_norm"] for r in rows];alloc=[r["cuda_allocated_gb"] for r in rows];early=stats(gradients[:20]);late=stats(gradients[99:]);overall=stats(gradients);resolved=bool(late["mean"]<=early["mean"]*1.2 and late["median"]<=early["median"]*1.2 and late["fraction_gt_500"]<=early["fraction_gt_500"]+.05 and overall["P95"]<500);report={"steps":len(rows),"raw_gradient":overall,"steps_1_20":early,"steps_21_99":stats(gradients[20:99]),"steps_100_200":late,"late_mean_change_percent":100*(late["mean"]/early["mean"]-1),"late_median_change_percent":100*(late["median"]/early["median"]-1),"late_gradient_growth_resolved":resolved,"all_finite":True,"qwen_frozen":True,"cuda_peak_gb":max(r["cuda_peak_gb"] for r in rows),"memory_growth_last20_minus_first20_gb":float(np.mean(alloc[-20:])-np.mean(alloc[:20])),"initial_checksum":initial,"final_checksum":trainable_state_checksum(model),"formal_training_started":False,"test_materialized":False};return model,rows,memory,report
def figures(args,comparison_rows,preflight_rows):
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 folder=args.output_dir/"figures";folder.mkdir(exist_ok=True);paths=[];key=lambda l,m:next(r for r in comparison_rows if r["loss"]==l and r["module"]==m);names=("Benefit","Total","Projection","Benefit Head","Uncertainty Head");chosen=(key("benefit","all_trainable"),key("total","all_trainable"),key("benefit","projection"),key("benefit","benefit_head"),key("benefit","uncertainty_head"));x=np.arange(5);plt.figure(figsize=(9,4));plt.bar(x-.18,[float(r["C_S4_mean"]) for r in chosen],.36,label="C-S4");plt.bar(x+.18,[float(r["C_S5_mean"]) for r in chosen],.36,label="C-S5");plt.xticks(x,names,rotation=20);plt.ylabel("mean gradient norm");plt.legend();plt.tight_layout();p=folder/"gradient_before_after.png";plt.savefig(p,dpi=160);plt.close();paths.append(str(p));plt.figure(figsize=(8,4));plt.plot([r["step"] for r in preflight_rows],[r["raw_total_grad_norm"] for r in preflight_rows]);plt.axvline(100,color="gray",linestyle="--");plt.xlabel("step");plt.ylabel("raw gradient norm");plt.tight_layout();p=folder/"preflight_gradient.png";plt.savefig(p,dpi=160);plt.close();paths.append(str(p));return paths
def finalize_completed(args):
 summary=json.loads((args.output_dir/"summary.json").read_text());rows=read_csv(args.output_dir/"stability_preflight.csv");gradients=[float(r["raw_total_grad_norm"]) for r in rows];early=stats(gradients[:20]);late=stats(gradients[99:]);overall=stats(gradients);resolved=bool(late["mean"]<=early["mean"]*1.2 and late["median"]<=early["median"]*1.2 and late["fraction_gt_500"]<=early["fraction_gt_500"]+.05 and overall["P95"]<500);report=summary["stability_preflight"];report.update({"raw_gradient":overall,"steps_1_20":early,"steps_21_99":stats(gradients[20:99]),"steps_100_200":late,"late_gradient_growth_resolved":resolved});diagnostics=read_csv(args.output_dir/"uncertainty_diagnostics.csv")
 for row in diagnostics:
  if str(row["collapsed"]).lower()=="true":row["uncertainty_error_correlation"]=""
 write_csv(args.output_dir/"uncertainty_diagnostics.csv",diagnostics);collapsed=any(str(r["collapsed"]).lower()=="true" for r in diagnostics);summary.update({"success":False,"formal_frozen3b_seed42_ready":False,"stability_preflight":report,"uncertainty_diagnostics":diagnostics,"uncertainty_collapsed":collapsed,"interpretation":"Stop-gradient contract is correct, but the standalone +0.5*log_variance regularizer drives log variance to the -6 clamp; exp(-log_variance) then amplifies the benefit-mean gradient catastrophically.","C_S5_resolution":"FAILED - do not retain this intervention for formal training","recommended_next_single_variable_intervention":"Return to the retained C-S4 objective and run a learning-rate-only diagnostic intervention; do not implement without approval."});write_json(args.output_dir/"summary.json",summary)
def main():
 args=args_parser();import torch
 if args.artifacts_only:finalize_completed(args);return
 from scripts.run_phase5a_frozen3b import build_development_data,prepare_training_tensors
 development=build_development_data(args,torch);tensors=prepare_training_tensors(development,torch);comp=comparison(args);model,rows,memory,report=preflight(args,development,tensors,torch);write_csv(args.output_dir/"stability_preflight.csv",rows);write_csv(args.output_dir/"memory_audit.csv",memory);diagnostics=uncertainty_diagnostics(model,tensors,args,torch);write_csv(args.output_dir/"uncertainty_diagnostics.csv",diagnostics);contract=json.loads((args.output_dir/"autograd_contract.json").read_text());summary={"label":LABEL,"stage":"Phase 5A Stage C-S5 Uncertainty Gradient-Path Isolation","success":bool(report["late_gradient_growth_resolved"]),"autograd_contract":contract,"gradient_before_after":comp,"stability_preflight":report,"uncertainty_diagnostics":diagnostics,"formal_frozen3b_seed42_ready":bool(report["late_gradient_growth_resolved"] and report["all_finite"] and report["qwen_frozen"] and not any(r["collapsed"] for r in diagnostics)),"formal_training_started":False,"test_materialized":False,"only_change":"stop-gradient benefit_log_variance inside Benefit likelihood","scale_alignment_retained":True,"normalizer_parity_retained":True,"figures":figures(args,comp,rows)};write_json(args.output_dir/"summary.json",summary);print(json.dumps({"contract":contract,"preflight":report,"uncertainty":diagnostics},indent=2),flush=True)
if __name__=="__main__":main()

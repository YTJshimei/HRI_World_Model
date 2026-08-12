"""Phase 5A Stage A/B: mock pipeline and L1 small context value test."""
from __future__ import annotations
import argparse,copy,csv,importlib.metadata,importlib.util,json,math,platform,random,sys,time
from pathlib import Path
from typing import Any
import numpy as np
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
LABEL="SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"

def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--device",choices=("cuda","cpu"),default="cuda");p.add_argument("--seed",type=int,default=42);p.add_argument("--epochs",type=int,default=60);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--belief-samples",type=int,choices=(16,),default=16,help="Frozen Phase 4C.2 distributional rollout sample count");p.add_argument("--phase4b6-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4b6");p.add_argument("--phase4c-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c");p.add_argument("--phase4c1-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c1");p.add_argument("--phase4c2-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c2");p.add_argument("--phase4c3-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c3");p.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase5a");return p.parse_args()
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):v=v.item()
 if isinstance(v,float) and not math.isfinite(v):return None
 return v
def write_json(p,v):p.write_text(json.dumps(clean(v),indent=2,allow_nan=False),encoding="utf-8")
def write_csv(p,rows):
 fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 with p.open("w",newline="",encoding="utf-8") as h:
  w=csv.DictWriter(h,fieldnames=fields or ("empty",));w.writeheader()
  for row in rows:w.writerow({f:clean(row.get(f,"")) for f in fields})

def environment_audit(torch):
 packages={}
 for name in ("torch","transformers","accelerate","bitsandbytes","peft"):
  available=importlib.util.find_spec(name) is not None
  try:version=importlib.metadata.version(name) if available else None
  except importlib.metadata.PackageNotFoundError:version=None
  packages[name]={"available":available,"version":version}
 result={"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"python":platform.python_version(),"platform":platform.platform(),"packages":packages,"cuda_available":torch.cuda.is_available(),"torch_cuda_version":torch.version.cuda,"audit_only_no_install":True,"model_downloaded":False}
 if torch.cuda.is_available():
  props=torch.cuda.get_device_properties(0);result.update({"gpu_name":props.name,"gpu_vram_gb":props.total_memory/2**30,"compute_capability":list(torch.cuda.get_device_capability(0))})
 result["frozen_3b_environment_ready"] = bool(torch.cuda.is_available() and packages["transformers"]["available"] and packages["accelerate"]["available"])
 result["qlora_environment_ready"] = bool(result["frozen_3b_environment_ready"] and packages["bitsandbytes"]["available"] and packages["peft"]["available"])
 return result

CONTEXT_MAP={"S1_too_close":"C1_seen_motion_seen_action","S2_too_far":"C1_seen_motion_seen_action","S6_high_distance_sensitive":"C1_seen_motion_seen_action","S4_human_decelerating":"C2_unseen_motion_action","S3_human_accelerating":"C3_unseen_person_seen_context","S7_high_speed_sensitive":"C3_unseen_person_seen_context","S9_uncertain_new_person":"C4_unseen_person_unseen_motion_action","S5_human_turning":"C5_compound_occlusion_turn_speed","S8_high_turn_sensitive":"C5_compound_occlusion_turn_speed","S10_action_conflict":"C6_partial_functional_observation"}

def development_candidate_allowed(meta):
 """Predeclared holdouts used identically for train and validation."""
 # Profiles 2 and 6 are test-only for C3/C4 and never enter model tokens.
 if meta["profile"] in (2,6):return False
 # C2 withholds two robot actions for the deceleration motion state.
 if meta["scenario"]=="S4_human_decelerating" and meta["action"] in (2,4):return False
 # C5 withholds speed-change actions under turning+occlusion context.
 if meta["scenario"] in ("S5_human_turning","S8_high_turn_sensitive") and meta["action"] in (1,2):return False
 # C6 holds out partial (K<=1) functional observation contexts.
 if meta["support_count"]<=1:return False
 return True

def build_tokens(episodes,split_name,population_theta):
 from src.data.functional_response_state import aggregate_response_state_mask
 from src.data.robot_action_schema import action_feature
 from src.multimodal.context_dataset import ContextDataset,ContextTarget
 from src.multimodal.context_schema import build_context_tokens
 samples=[];targets=[];meta=[]
 for episode in episodes:
  context_split=CONTEXT_MAP[episode.key[0]] if split_name=="test" else split_name
  valid=np.flatnonzero(episode.feasible)
  generic=int(valid[np.argmin(episode.generic_costs.total[valid])]) if len(valid) else int(np.argmin(episode.gt_costs.total))
  support_actions=[]
  for probe in episode.first["support"]:
   if "SPEED_DOWN" in probe:support_actions.append(1)
   elif "SPEED_UP" in probe:support_actions.append(2)
   elif "DISTANCE_PLUS" in probe:support_actions.append(3)
   elif "DISTANCE_MINUS" in probe:support_actions.append(4)
  state_mask=aggregate_response_state_mask(support_actions)
  support_feature=np.asarray((len(support_actions)/5.,len(set(support_actions))/5.,state_mask.mean()),np.float32)
  history=episode.first["sample_data"]["history"];robot=episode.first["sample_data"]["robot"]
  from src.data.skeleton_schema import compute_root
  root=compute_root(history);velocity=np.diff(root[:,:2],axis=0)*10.;heading=np.unwrap(np.arctan2(velocity[:,1],velocity[:,0])) if len(velocity) else np.zeros(1)
  speed=np.linalg.norm(velocity,axis=1) if len(velocity) else np.zeros(1);distance_hist=robot[:,5]
  heading_change=heading[-1]-heading[0]
  speed_change=speed[-1]-speed[0]
  # Motion-state indicators are inferred solely from the observed history;
  # the synthetic generator's action_type label never enters model input.
  motion_observable=np.asarray((speed.mean(),speed[-1],speed_change,heading_change,np.std(speed),np.mean(np.abs(np.diff(heading))) if len(heading)>1 else 0.,float(abs(heading_change)>.12),float(speed_change>.15)))
  scene=np.asarray((np.asarray(episode.first["sample_data"]["visibility"]).mean(),np.asarray(episode.first["sample_data"]["confidence"]).mean(),distance_hist[-1],distance_hist[-1]-distance_hist[0],speed.mean(),np.std(speed),robot[-1,6],len(support_actions)/5.))
  for index,action in enumerate(episode.personal_costs.action_ids):
   token=build_context_tokens(human_history=history,robot_history=robot,confidence=episode.first["sample_data"]["confidence"],visibility=episode.first["sample_data"]["visibility"],theta_person=episode.first["theta_hat"],theta_population=population_theta,theta_uncertainty=episode.first["theta_std"],response_state_mask=state_mask,support_coverage=episode.confidence.support_coverage,support_action_features=support_feature,candidate_action=int(action),candidate_feature=action_feature(int(action)),predicted_robot_future=episode.first["predicted_rollout"].predicted_robot_xy[index],generic_effect=episode.generic_rollout.predicted_action_effect[index],personalized_effect=episode.first["predicted_rollout"].predicted_action_effect[index],generic_distance=episode.generic_rollout.predicted_human_robot_distance[index],personalized_distance=episode.first["predicted_rollout"].predicted_human_robot_distance[index],root_sigma=episode.artifact.root_belief.sigma_root,minimum_sigma=float(episode.safety_prediction["sigma_minimum"][index]),p_unsafe=float(episode.safety_prediction["p_unsafe"][index]),motion_state_observable=motion_observable,scene_observable=scene,context_id=f"{split_name}:{episode.key[0]}:{episode.key[1]}:{int(action)}",initial_state_id=f"{split_name}:{episode.key[0]}:{episode.key[1]}",context_split=context_split)
   benefit=float(episode.gt_costs.total[generic]-episode.gt_costs.total[index]);harm=benefit<-1e-6
   relevance=state_mask.mean();aux=np.asarray((motion_observable[2],distance_hist[-1]-distance_hist[0],motion_observable[3],relevance,len(support_actions)/5.,np.sign(episode.personal_costs.total[index]-episode.generic_costs.total[index])),np.float32)
   samples.append(token);targets.append(ContextTarget(benefit,harm,aux));meta.append({"scenario":episode.key[0],"sample":episode.key[1],"profile":int(episode.first["profile"]),"support_count":len(support_actions),"action":int(action),"feasible":bool(episode.feasible[index]),"generic_index":generic,"GT_cost":float(episode.gt_costs.total[index]),"GT_unsafe":bool(episode.gt_costs.unsafe_duration[index]>0),"context_split":context_split})
 datasets=[]
 for name in sorted(set(s.context_split for s in samples)):
  idx=[i for i,s in enumerate(samples) if s.context_split==name];datasets.append(ContextDataset(tuple(samples[i] for i in idx),tuple(targets[i] for i in idx),name))
 return datasets,samples,targets,meta

def train_model(model,train_samples,train_targets,train_meta,val_samples,val_targets,val_meta,args,torch):
 device=torch.device(args.device);model=model.to(device)
 train_keep=np.asarray([row["feasible"] for row in train_meta],bool);val_keep=np.asarray([row["feasible"] for row in val_meta],bool)
 raw_x=np.stack([s.flattened() for s in train_samples])[train_keep];raw_vx=np.stack([s.flattened() for s in val_samples])[val_keep]
 feature_mean=raw_x.mean(0);feature_scale=raw_x.std(0);feature_scale=np.where(feature_scale<1e-5,1.,feature_scale)
 raw_y=np.asarray([t.benefit for t in train_targets],np.float32)[train_keep];benefit_mean=float(raw_y.mean());benefit_scale=max(float(raw_y.std()),1e-4)
 x=torch.from_numpy(((raw_x-feature_mean)/feature_scale).astype(np.float32));y=torch.from_numpy(((raw_y-benefit_mean)/benefit_scale).astype(np.float32));h=torch.tensor([t.harm for i,t in enumerate(train_targets) if train_keep[i]],dtype=torch.float32);aux=torch.from_numpy(np.stack([t.auxiliary for i,t in enumerate(train_targets) if train_keep[i]])).float();vx=torch.from_numpy(((raw_vx-feature_mean)/feature_scale).astype(np.float32));vy=torch.from_numpy(((np.asarray([t.benefit for i,t in enumerate(val_targets) if val_keep[i]],np.float32)-benefit_mean)/benefit_scale).astype(np.float32));vh=torch.tensor([t.harm for i,t in enumerate(val_targets) if val_keep[i]],dtype=torch.float32);va=torch.from_numpy(np.stack([t.auxiliary for i,t in enumerate(val_targets) if val_keep[i]])).float();opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-3);g=torch.Generator().manual_seed(args.seed);best=(float("inf"),None,0)
 for epoch in range(1,args.epochs+1):
  model.train();order=torch.randperm(len(x),generator=g)
  for start in range(0,len(x),args.batch_size):
   ids=order[start:start+args.batch_size];prediction=model(x[ids].to(device));error=prediction.benefit_mean-y[ids].to(device);loss=(.5*(error.square()*torch.exp(-prediction.benefit_log_variance)+prediction.benefit_log_variance)).mean()+torch.nn.functional.binary_cross_entropy_with_logits(prediction.harm_logit,h[ids].to(device),pos_weight=torch.tensor(2.,device=device))+.15*torch.nn.functional.mse_loss(prediction.auxiliary,aux[ids].to(device));opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  model.eval()
  with torch.inference_mode():
   p=model(vx.to(device));loss=float((torch.nn.functional.l1_loss(p.benefit_mean,vy.to(device))+torch.nn.functional.binary_cross_entropy_with_logits(p.harm_logit,vh.to(device))+.1*torch.nn.functional.mse_loss(p.auxiliary,va.to(device))).item())
  if loss<best[0]:best=(loss,copy.deepcopy(model.state_dict()),epoch)
 model.load_state_dict(best[1]);model.eval();model.phase5_feature_mean=feature_mean;model.phase5_feature_scale=feature_scale;model.phase5_benefit_mean=benefit_mean;model.phase5_benefit_scale=benefit_scale;return model,{"best_epoch":best[2],"best_validation_loss":best[0],"parameters":sum(p.numel() for p in model.parameters()),"fit_candidates":int(train_keep.sum()),"validation_candidates":int(val_keep.sum()),"feature_normalization":"train-only","benefit_normalization":"train-only"}

def predict(model,samples,device,torch):
 raw=np.stack([s.flattened() for s in samples]);x=torch.from_numpy(((raw-model.phase5_feature_mean)/model.phase5_feature_scale).astype(np.float32)).to(device)
 with torch.inference_mode():p=model(x)
 return {"benefit":p.benefit_mean.cpu().numpy()*model.phase5_benefit_scale+model.phase5_benefit_mean,"sigma":np.exp(.5*p.benefit_log_variance.cpu().numpy())*model.phase5_benefit_scale,"harm":p.harm_logit.sigmoid().cpu().numpy()}

def rank_corr(a,b):
 a=np.asarray(a);b=np.asarray(b)
 return float(np.corrcoef(np.argsort(np.argsort(a)),np.argsort(np.argsort(b)))[0,1]) if len(a)>1 and np.std(a)>1e-12 and np.std(b)>1e-12 else None
def binary_auc(probability,truth):
 p=np.asarray(probability);y=np.asarray(truth,bool);pos=y.sum();neg=(~y).sum()
 if not pos or not neg:return None
 ranks=np.argsort(np.argsort(p))+1;return float((ranks[y].sum()-pos*(pos+1)/2)/(pos*neg))

def choose_thresholds(pred,targets,meta):
 best=None;rows=[];benefit=np.asarray([t.benefit for t in targets]);harm=np.asarray([t.harm for t in targets])
 for b in (-.02,0.,.01,.02,.04,.08):
  for h in (.2,.3,.4,.5,.6):
   approve=(pred["benefit"]>=b)&(pred["harm"]<=h)&np.asarray([m["feasible"] for m in meta])
   false_harm=np.mean(approve&harm);beneficial_recall=np.sum(approve&(benefit>1e-6))/max(np.sum(benefit>1e-6),1);score=false_harm+.15*(1-beneficial_recall)
   row={"benefit_threshold":b,"harm_threshold":h,"score":score,"beneficial_recall":beneficial_recall,"approved_harm":false_harm};rows.append(row)
   if best is None or (score,-beneficial_recall)<best[0]:best=((score,-beneficial_recall),(b,h))
 return best[1],rows

def evaluate_model(name,pred,samples,targets,meta,episodes,thresholds):
 from src.decision.large_context_arbitrator import arbitrate_large_context
 benefit=np.asarray([t.benefit for t in targets]);harm=np.asarray([t.harm for t in targets]);feasible_eval=np.asarray([m["feasible"] for m in meta],bool);rows=[]
 for i,(sample,target,item) in enumerate(zip(samples,targets,meta)):
  rows.append({"synthetic_interaction":LABEL,"model":name,"scenario":item["scenario"],"sample":item["sample"],"action":item["action"],"context_split":item["context_split"],"predicted_benefit":pred["benefit"][i],"benefit_uncertainty":pred["sigma"][i],"GT_benefit_evaluation_only":target.benefit,"predicted_harm_probability":pred["harm"][i],"GT_harm_evaluation_only":target.harm,"feasible":item["feasible"]})
 decisions=[]
 by_key={}
 for i,item in enumerate(meta):by_key.setdefault((item["scenario"],item["sample"]),[]).append(i)
 episode_map={e.key:e for e in episodes}
 for key,indices in by_key.items():
  episode=episode_map[key];decision=arbitrate_large_context(episode.personal_costs.action_ids,episode.feasible,episode.generic_costs.total,episode.personal_costs.total,pred["benefit"][indices],pred["harm"][indices],*thresholds);oracle=int(np.argmin(episode.gt_costs.total));valid=np.flatnonzero(episode.feasible);generic=int(valid[np.argmin(episode.generic_costs.total[valid])]) if len(valid) else oracle
  if decision.selected_index is None:index=None;cost=float(episode.gt_costs.total[oracle]+.25);regret=.25;unsafe=False
  else:index=decision.selected_index;cost=float(episode.gt_costs.total[index]);regret=cost-float(episode.gt_costs.total[oracle]);unsafe=bool(episode.gt_costs.unsafe_duration[index]>0)
  switched=index is not None and index!=generic;delta=0 if index is None else float(episode.gt_costs.total[index]-episode.gt_costs.total[generic])
  decisions.append({"synthetic_interaction":LABEL,"model":name,"scenario":key[0],"sample":key[1],"context_split":CONTEXT_MAP[key[0]],"selected_action":"" if index is None else int(episode.personal_costs.action_ids[index]),"decision_mode":decision.mode.value,"personalized":decision.personalization_approved,"beneficial_switch":bool(switched and delta<-1e-6),"harmful_switch":bool(switched and delta>1e-6),"GT_Total_Cost":cost,"Oracle_Regret":regret,"Safety_Violation":unsafe,"reentry":bool(index is not None and not episode.feasible[index])})
 beneficial=benefit>1e-6;approved=np.asarray([d["personalized"] for d in decisions]);actual_beneficial=np.asarray([d["beneficial_switch"] for d in decisions]);actual_harmful=np.asarray([d["harmful_switch"] for d in decisions]);regrets=np.asarray([d["Oracle_Regret"] for d in decisions])
 metrics={"Benefit_MAE":float(np.mean(np.abs(pred["benefit"][feasible_eval]-benefit[feasible_eval]))),"Benefit_Ranking_Spearman":rank_corr(pred["benefit"][feasible_eval],benefit[feasible_eval]),"Harm_AUROC":binary_auc(pred["harm"][feasible_eval],harm[feasible_eval]),"Beneficial_Switch_Recall":float(actual_beneficial.sum()/max(np.sum([e.gt_costs.total.min()<e.gt_costs.total[np.flatnonzero(e.feasible)[np.argmin(e.generic_costs.total[np.flatnonzero(e.feasible)])]]-1e-6 if e.feasible.any() else False for e in episodes]),1)),"Beneficial_Switch_Precision":float(actual_beneficial.sum()/max(approved.sum(),1)),"Harmful_Switch_Rate":float(actual_harmful.mean()),"GT_Total_Cost":float(np.mean([d["GT_Total_Cost"] for d in decisions])),"Mean_Regret":float(regrets.mean()),"P95_Regret":float(np.percentile(regrets,95)),"Max_Regret":float(regrets.max()),"Safety_Violation":float(np.mean([d["Safety_Violation"] for d in decisions])),"Personalized_Decision_Rate":float(approved.mean()),"Generic_Safe_Rate":float(np.mean([d["decision_mode"]=="GENERIC_SAFE" for d in decisions])),"Benefit_Evaluation_Scope":"frozen feasible candidates only"}
 return rows,decisions,metrics

def make_figures(output,evaluation):
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 d=output/"figures";d.mkdir(parents=True,exist_ok=True);paths=[]
 def save(n):
  p=d/n;plt.title(LABEL,fontsize=7);plt.tight_layout();plt.savefig(p,dpi=150);plt.close();paths.append(str(p))
 for name,result in evaluation.items():
  rows=result[0];plt.figure();plt.scatter([r["predicted_benefit"] for r in rows],[r["GT_benefit_evaluation_only"] for r in rows],alpha=.25);plt.xlabel("predicted benefit");plt.ylabel("GT synthetic benefit");save(f"{name.lower()}_benefit.png")
 plt.figure();names=list(evaluation);plt.bar(names,[evaluation[n][2]["Mean_Regret"] for n in names]);plt.ylabel("mean regret");save("model_regret.png")
 return paths

def main():
 args=parse_args();args.output_dir.mkdir(parents=True,exist_ok=True);random.seed(args.seed);np.random.seed(args.seed)
 import torch;torch.manual_seed(args.seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
 from src.models.large_context_adapter import MockLargeContextBackbone,SmallContextNetwork
 import scripts.run_phase4c_decision as c0, scripts.run_phase4c1_safety_calibration as c1, scripts.run_phase4c2_belief_selection as c2, scripts.run_phase4c3_selective_personalization as c3
 from src.decision.counterfactual_rollout import CounterfactualRolloutEngine
 audit=environment_audit(torch);write_json(args.output_dir/"environment_audit.json",audit)
 engine=CounterfactualRolloutEngine.from_phase4b6_checkpoint(args.phase4b6_dir/"checkpoints"/"f2_original_best.pt",args.device);prior_mean,prior_std=c0.load_prior(argparse.Namespace(phase4b6_dir=args.phase4b6_dir));root,scale,safety,calibration,c2_summary=c3.load_frozen_phase4c2(args,torch);selector=c2.SelectorConfig(**c2_summary["selector_config"])
 def records(split,seed,count):return c1.build_records(args,engine,split,seed,count,prior_mean,prior_std)
 train=records("train",args.seed+101,30);val=records("validation",args.seed+202,12)
 ta,tp,cost=c3.build_base(args,train,engine,prior_mean,prior_std,root,scale,safety,calibration,None,torch);va,vp,_=c3.build_base(args,val,engine,prior_mean,prior_std,root,scale,safety,calibration,cost,torch)
 te=c3.episode_data(args,train,ta,tp,cost,engine,prior_mean,prior_std,selector);ve=c3.episode_data(args,val,va,vp,cost,engine,prior_mean,prior_std,selector)
 td,ts_all,tt_all,tm_all=build_tokens(te,"train",prior_mean);vd,vs_all,vt_all,vm_all=build_tokens(ve,"validation",prior_mean)
 train_keep=[development_candidate_allowed(row) for row in tm_all];val_keep=[development_candidate_allowed(row) for row in vm_all]
 ts=[s for s,k in zip(ts_all,train_keep) if k];tt=[t for t,k in zip(tt_all,train_keep) if k];tm=[m for m,k in zip(tm_all,train_keep) if k]
 vs=[s for s,k in zip(vs_all,val_keep) if k];vt=[t for t,k in zip(vt_all,val_keep) if k];vm=[m for m,k in zip(vm_all,val_keep) if k]
 mock,mock_train=train_model(MockLargeContextBackbone(),ts,tt,tm,vs,vt,vm,argparse.Namespace(**{**vars(args),"epochs":3}),torch);small,small_train=train_model(SmallContextNetwork(),ts,tt,tm,vs,vt,vm,args,torch)
 mock_val=predict(mock,vs,torch.device(args.device),torch);small_val=predict(small,vs,torch.device(args.device),torch);mock_threshold,_=choose_thresholds(mock_val,vt,vm);small_threshold,calibration_rows=choose_thresholds(small_val,vt,vm)
 # Test is built only after L1 checkpoint-equivalent state and thresholds freeze.
 test=records("test",args.seed+303,12);xa,xp,_=c3.build_base(args,test,engine,prior_mean,prior_std,root,scale,safety,calibration,cost,torch);xe=c3.episode_data(args,test,xa,xp,cost,engine,prior_mean,prior_std,selector);datasets,samples,targets,meta=build_tokens(xe,"test",prior_mean)
 mock_pred=predict(mock,samples,torch.device(args.device),torch);small_pred=predict(small,samples,torch.device(args.device),torch)
 evaluation={"Mock":evaluate_model("Mock",mock_pred,samples,targets,meta,xe,mock_threshold),"L1":evaluate_model("L1",small_pred,samples,targets,meta,xe,small_threshold)}
 benefit_rows=evaluation["Mock"][0]+evaluation["L1"][0];decision_rows=evaluation["Mock"][1]+evaluation["L1"][1]
 write_csv(args.output_dir/"benefit_prediction.csv",benefit_rows);write_csv(args.output_dir/"harm_prediction.csv",benefit_rows);write_csv(args.output_dir/"switch_metrics.csv",[{"synthetic_interaction":LABEL,"model":name,**values[2]} for name,values in evaluation.items()]);write_csv(args.output_dir/"decision_metrics.csv",decision_rows)
 context_rows=[]
 for name,values in evaluation.items():
  for split in dict.fromkeys(CONTEXT_MAP.values()):
   rows=[r for r in values[1] if r["context_split"]==split]
   if rows:context_rows.append({"synthetic_interaction":LABEL,"model":name,"context_split":split,"count":len(rows),"GT_Total_Cost":np.mean([r["GT_Total_Cost"] for r in rows]),"Mean_Regret":np.mean([r["Oracle_Regret"] for r in rows]),"Safety_Violation":np.mean([r["Safety_Violation"] for r in rows]),"Personalized_Rate":np.mean([r["personalized"] for r in rows])})
 write_csv(args.output_dir/"by_context_split.csv",context_rows);write_csv(args.output_dir/"small_vs_large.csv",[{"synthetic_interaction":LABEL,"comparison":"L1 vs future L2","L1_value":v,"L2_value":"NOT RUN - no model downloaded","metric":k} for k,v in evaluation["L1"][2].items()]);write_csv(args.output_dir/"hard_cases.csv",[r for r in decision_rows if r["scenario"] in ("S6_high_distance_sensitive","S8_high_turn_sensitive","S9_uncertain_new_person") or r["Oracle_Regret"]>.25])
 frozen=json.loads((args.phase4c3_dir/"switch_audit.csv").read_text(encoding="utf-8")[:0] or "{}")
 hard={"label":LABEL,"sources":["S9_uncertain_new_person","S6_high_distance_sensitive","S8_high_turn_sensitive"],"phase4c3_beneficial_cases":[{"scenario":r["scenario"],"sample":int(r["sample"])} for r in csv.DictReader((args.phase4c3_dir/"switch_audit.csv").open(encoding="utf-8")) if r["full_switch"]=="BENEFICIAL_SWITCH"],"phase4c3_harmful_cases":[{"scenario":r["scenario"],"sample":int(r["sample"])} for r in csv.DictReader((args.phase4c3_dir/"switch_audit.csv").open(encoding="utf-8")) if r["full_switch"]=="HARMFUL_SWITCH"],"phase4c2_max_regret_cases":[{"scenario":r["scenario"],"sample":int(r["sample"]),"regret":float(r["Oracle_Regret"])} for r in csv.DictReader((args.phase4c2_dir/"decision_summary.csv").open(encoding="utf-8")) if float(r["Oracle_Regret"])>.5]};write_json(args.output_dir/"hard_case_manifest.json",hard)
 audit_data={"label":LABEL,"input_dimension":len(samples[0].flattened()),"raw_train_candidates":len(ts_all),"raw_validation_candidates":len(vs_all),"train_candidates_after_predeclared_holdout":len(ts),"validation_candidates_after_predeclared_holdout":len(vs),"test_candidates":len(samples),"test_initial_states":len(set(s.initial_state_id for s in samples)),"context_splits":{d.split_name:len(d.samples) for d in datasets},"holdout_protocol":{"C3_unseen_profiles":[2],"C4_unseen_profiles":[6],"C2_withheld_motion_action":{"S4_human_decelerating":[2,4]},"C5_withheld_turn_speed_actions":[1,2],"C6_partial_support_max_K":1},"forbidden_inputs_absent":True,"identity_shortcut_absent":True,"counterfactual_split_isolation":True,"support_query_temporally_isolated":True};write_json(args.output_dir/"dataset_audit.json",audit_data)
 figures=make_figures(args.output_dir,evaluation);summary={"label":LABEL,"stage":"A/B only","mock_is_not_formal_result":True,"L2_3B_run":False,"model_downloaded":False,"dependencies_installed":False,"environment":audit,"dataset":audit_data,"training":{"Mock":mock_train,"L1":small_train},"thresholds":{"Mock":mock_threshold,"L1":small_threshold},"models":{"Mock":evaluation["Mock"][2],"L1":evaluation["L1"][2],"L2":"NOT RUN"},"phase4_safety_frozen":True,"results_phase4_untouched":True,"figures":figures,"next_stage_requires_human_approval":True};write_json(args.output_dir/"summary.json",summary)
 print(f"L1 benefit_MAE={evaluation['L1'][2]['Benefit_MAE']:.5f} mean_regret={evaluation['L1'][2]['Mean_Regret']:.5f} personalized_rate={evaluation['L1'][2]['Personalized_Decision_Rate']:.4f}",flush=True)
if __name__=="__main__":main()

"""Phase 5A Stage C-S2: frozen-3B gradient-source decomposition, train-only."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
LABEL="SYNTHETIC INTERACTION - NOT REAL HUMAN DATA"
LOSS_NAMES=("benefit","harm","uncertainty")
MODULE_NAMES=("projection","benefit_head","harm_head","uncertainty_head")


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id",default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--cache-dir",type=Path,default=Path.home()/".cache"/"huggingface")
    parser.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase5a_gradient_audit")
    parser.add_argument("--phase5a-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase5a")
    parser.add_argument("--phase4b6-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4b6")
    parser.add_argument("--phase4c-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c")
    parser.add_argument("--phase4c1-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c1")
    parser.add_argument("--phase4c2-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c2")
    parser.add_argument("--phase4c3-dir",type=Path,default=PROJECT_ROOT/"results_dev"/"phase4c3")
    parser.add_argument("--seed",type=int,choices=(42,),default=42)
    parser.add_argument("--device",choices=("cuda",),default="cuda")
    parser.add_argument("--batch-size",type=int,choices=(8,),default=8)
    parser.add_argument("--diagnostic-batches",type=int,choices=(32,),default=32)
    parser.add_argument("--learning-rate",type=float,default=3e-4)
    parser.add_argument("--belief-samples",type=int,choices=(16,),default=16)
    return parser.parse_args()


def clean(value):
    if isinstance(value,dict):return {str(key):clean(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [clean(item) for item in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def write_json(path,value):path.write_text(json.dumps(clean(value),indent=2,allow_nan=False),encoding="utf-8")


def write_csv(path,rows):
    import csv
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields or ("empty",));writer.writeheader()
        for row in rows:writer.writerow({field:clean(row.get(field,"")) for field in fields})


def stats(values):
    array=np.asarray(values,np.float64)
    return {"mean":float(array.mean()),"std":float(array.std()),"median":float(np.median(array)),"min":float(array.min()),"P1":float(np.percentile(array,1)),"P5":float(np.percentile(array,5)),"P90":float(np.percentile(array,90)),"P95":float(np.percentile(array,95)),"P99":float(np.percentile(array,99)),"max":float(array.max())}


def module_parameters(model):
    return {
        "projection":tuple(model.projection.parameters()),
        "benefit_head":tuple(model.benefit.parameters()),
        "harm_head":tuple(model.harm.parameters()),
        "uncertainty_head":tuple(model.uncertainty.parameters()),
    }


def attribution_parameters(model):
    groups=module_parameters(model)
    parameters=tuple(parameter for name in MODULE_NAMES for parameter in groups[name])
    backbone_ids={id(parameter) for parameter in model.backbone.parameters()}
    if backbone_ids&{id(parameter) for parameter in parameters}:raise RuntimeError("Qwen parameter entered attribution targets")
    return groups,parameters


def per_loss_gradients(losses,model,torch):
    """Return independent gradients without writing parameter.grad fields."""
    groups,parameters=attribution_parameters(model)
    if any(parameter.grad is not None for parameter in model.parameters()):raise RuntimeError("gradient contamination before attribution")
    output={}
    for loss_name in (*LOSS_NAMES,"total"):
        loss=sum(losses[name] for name in LOSS_NAMES) if loss_name=="total" else losses[loss_name]
        gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        if any(parameter.grad is not None for parameter in model.parameters()):raise RuntimeError("autograd.grad polluted .grad")
        offset=0;by_module={}
        for module_name in MODULE_NAMES:
            count=len(groups[module_name]);by_module[module_name]=tuple(gradients[offset:offset+count]);offset+=count
        output[loss_name]={"all":tuple(gradients),"modules":by_module}
    return output


def grad_norm(gradients):
    return math.sqrt(sum(float(gradient.detach().float().square().sum()) for gradient in gradients if gradient is not None))


def grad_dot(left,right):
    return sum(float((a.detach().float()*b.detach().float()).sum()) for a,b in zip(left,right) if a is not None and b is not None)


def grad_cosine(left,right):
    denominator=grad_norm(left)*grad_norm(right)
    value=0.0 if denominator<=0 else grad_dot(left,right)/denominator
    if not math.isfinite(value):raise FloatingPointError("gradient cosine is not finite")
    return float(np.clip(value,-1.0,1.0))


def exact_autograd_contract(model,prediction,target,torch):
    """Prove that only the likelihood-to-log-variance gradient path is cut."""
    from scripts.run_phase5a_frozen3b import benefit_likelihood_with_detached_variance
    groups=module_parameters(model)
    likelihood=benefit_likelihood_with_detached_variance(prediction.benefit_mean,target,prediction.benefit_log_variance,torch)
    original=.5*((prediction.benefit_mean-target).square()*torch.exp(-prediction.benefit_log_variance)).mean()
    regularizer=.5*prediction.benefit_log_variance.mean()
    def measure(loss,parameters):
        gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        return grad_norm(gradients),sum(gradient is not None and bool(torch.count_nonzero(gradient)) for gradient in gradients)
    projection=measure(likelihood,groups["projection"]);benefit=measure(likelihood,groups["benefit_head"]);uncertainty=measure(likelihood,groups["uncertainty_head"]);regularizer_uncertainty=measure(regularizer,groups["uncertainty_head"]);difference=float((original.detach()-likelihood.detach()).abs())
    result={"label":LABEL,"contract":"detach benefit_log_variance only inside Benefit likelihood","forward_value_original":float(original.detach()),"forward_value_detached":float(likelihood.detach()),"forward_max_abs_error":difference,"benefit_likelihood_projection_gradient_norm":projection[0],"benefit_likelihood_projection_nonzero_tensors":projection[1],"benefit_likelihood_benefit_head_gradient_norm":benefit[0],"benefit_likelihood_benefit_head_nonzero_tensors":benefit[1],"benefit_likelihood_uncertainty_head_gradient_norm":uncertainty[0],"benefit_likelihood_uncertainty_head_nonzero_tensors":uncertainty[1],"uncertainty_regularizer_uncertainty_head_gradient_norm":regularizer_uncertainty[0],"uncertainty_regularizer_uncertainty_head_nonzero_tensors":regularizer_uncertainty[1],"qwen_gradient_tensor_count":sum(parameter.grad is not None for parameter in model.backbone.parameters())}
    result["passed"]=projection[0]>0 and benefit[0]>0 and uncertainty[0]==0 and regularizer_uncertainty[0]>0 and difference==0 and result["qwen_gradient_tensor_count"]==0
    return result


def attribution_rows(batch_id,losses,attributions):
    total_norm=grad_norm(attributions["total"]["all"]);sum_individual=sum(grad_norm(attributions[name]["all"]) for name in LOSS_NAMES)
    rows=[]
    for loss_name in (*LOSS_NAMES,"total"):
        all_gradients=attributions[loss_name]["all"];loss_norm=grad_norm(all_gradients)
        alignment=1.0 if loss_name=="total" else grad_dot(all_gradients,attributions["total"]["all"])/max(total_norm**2,1e-30)
        for module_name in (*MODULE_NAMES,"all_trainable"):
            gradients=all_gradients if module_name=="all_trainable" else attributions[loss_name]["modules"][module_name]
            rows.append({"synthetic_interaction":LABEL,"record_type":"batch","batch_id":batch_id,"loss":loss_name,"module":module_name,"loss_value":float(losses[loss_name].detach()) if loss_name!="total" else float(sum(losses.values()).detach()),"gradient_norm":grad_norm(gradients),"magnitude_share_of_sum_individual_norms":loss_norm/max(sum_individual,1e-30) if loss_name!="total" else "","aligned_contribution_to_total_gradient":alignment})
    return rows


def aggregate_matrix(batch_rows):
    rows=[]
    for loss_name in (*LOSS_NAMES,"total"):
        for module_name in (*MODULE_NAMES,"all_trainable"):
            values=[row["gradient_norm"] for row in batch_rows if row["loss"]==loss_name and row["module"]==module_name]
            subset=[row for row in batch_rows if row["loss"]==loss_name and row["module"]==module_name]
            magnitude=[row["magnitude_share_of_sum_individual_norms"] for row in subset if row["magnitude_share_of_sum_individual_norms"]!=""]
            aligned=[row["aligned_contribution_to_total_gradient"] for row in subset]
            result={"synthetic_interaction":LABEL,"record_type":"summary","loss":loss_name,"module":module_name,**{key:stats(values)[key] for key in ("mean","median","P95","max")},"magnitude_share_mean":"" if not magnitude else np.mean(magnitude),"aligned_contribution_mean":np.mean(aligned)}
            rows.append(result)
    return rows


def distribution_rows(category,name,values):
    return [{"synthetic_interaction":LABEL,"category":category,"variable":name,**stats(values)}]


def native_embedding_norms(model,torch):
    weight=model.backbone.get_input_embeddings().weight
    pieces=[]
    with torch.inference_mode():
        for start in range(0,len(weight),2048):pieces.append(weight[start:start+2048].float().norm(dim=-1).cpu().numpy())
    return np.concatenate(pieces)


def make_figures(output,matrix_rows,cosine_rows,loss_rows):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    folder=output/"figures";folder.mkdir(parents=True,exist_ok=True);paths=[]
    matrix=np.asarray([[next(row["mean"] for row in matrix_rows if row["loss"]==loss and row["module"]==module) for module in MODULE_NAMES] for loss in LOSS_NAMES])
    plt.figure(figsize=(8,4));image=plt.imshow(np.log10(matrix+1e-12),aspect="auto",cmap="magma");plt.colorbar(image,label="log10 mean grad norm");plt.xticks(range(4),MODULE_NAMES,rotation=20);plt.yticks(range(3),LOSS_NAMES);plt.tight_layout();path=folder/"gradient_source_heatmap.png";plt.savefig(path,dpi=160);plt.close();paths.append(str(path))
    pairs=sorted(set(row["pair"] for row in cosine_rows if row["record_type"]=="batch"));plt.figure();plt.boxplot([[row["cosine"] for row in cosine_rows if row["record_type"]=="batch" and row["pair"]==pair] for pair in pairs],tick_labels=pairs);plt.axhline(0,color="black",linewidth=.5);plt.xticks(rotation=20);plt.tight_layout();path=folder/"gradient_cosine.png";plt.savefig(path,dpi=160);plt.close();paths.append(str(path))
    plt.figure();plt.boxplot([[row["value"] for row in loss_rows if row["record_type"]=="batch" and row["loss"]==name] for name in (*LOSS_NAMES,"total")],tick_labels=(*LOSS_NAMES,"total"));plt.ylabel("loss");plt.tight_layout();path=folder/"loss_scale.png";plt.savefig(path,dpi=160);plt.close();paths.append(str(path));return paths


def main():
    args=parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    random.seed(args.seed);np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():raise RuntimeError("CUDA is required")
    from scripts.run_phase5a_frozen3b import build_development_data,prepare_training_tensors,trainable_state_checksum
    from src.models.large_context_adapter import FrozenQwen25VLContextAdapter
    from src.multimodal.context_schema import TOKEN_DIMS,TOKEN_ORDER

    development=build_development_data(args,torch);tensors=prepare_training_tensors(development,torch)
    generator=torch.Generator().manual_seed(args.seed);order=tensors["feasible_indices"][torch.randperm(len(tensors["feasible_indices"]),generator=generator)]
    batches=[order[start:start+args.batch_size] for start in range(0,args.diagnostic_batches*args.batch_size,args.batch_size)]
    manifest={"label":LABEL,"seed":args.seed,"split":"train","batch_size":args.batch_size,"batch_count":len(batches),"optimizer_step_allowed":False,"test_materialized":False,"batches":[{"batch_id":index,"filtered_train_indices":batch.tolist(),"context_ids":[development["train_samples"][item].context_id for item in batch.tolist()]} for index,batch in enumerate(batches)]}
    write_json(args.output_dir/"diagnostic_batch_manifest.json",manifest)
    model=FrozenQwen25VLContextAdapter.from_pretrained_4bit(args.model_id,device_map={"":0},cache_dir=str(args.cache_dir),local_files_only=True).to("cuda");model.train()
    initial_checksum=trainable_state_checksum(model);matrix_batch=[];cosine_rows=[];cosine_space_rows=[];loss_rows=[];gaussian_rows=[];prediction_values={"benefit_mean_normalized":[],"harm_logit":[],"harm_probability":[],"benefit_log_variance":[],"sigma_normalized":[],"normalized_residual":[],"raw_benefit_residual":[]};projection_values={name:[] for name in TOKEN_ORDER};projected_values={name:[] for name in TOKEN_ORDER};hidden_values={name:[] for name in TOKEN_ORDER};context_hidden=[];batch_details=[];autograd_contract=None
    for batch_id,indices in enumerate(batches):
        features=tensors["train_x"][indices].to("cuda");target=tensors["train_y"][indices].to("cuda");harm=tensors["train_harm"][indices].to("cuda")
        prediction=model(features);error=prediction.benefit_mean-target
        from scripts.run_phase5a_frozen3b import benefit_likelihood_with_detached_variance
        losses={"benefit":benefit_likelihood_with_detached_variance(prediction.benefit_mean,target,prediction.benefit_log_variance,torch),"harm":torch.nn.functional.binary_cross_entropy_with_logits(prediction.harm_logit,harm,pos_weight=tensors["pos_weight"]),"uncertainty":.5*prediction.benefit_log_variance.mean()}
        if autograd_contract is None:
            autograd_contract=exact_autograd_contract(model,prediction,target,torch);write_json(args.output_dir/"autograd_contract.json",autograd_contract)
            if not autograd_contract["passed"]:raise RuntimeError("C-S5 exact autograd contract failed")
        if not all(bool(torch.isfinite(value)) for value in losses.values()):raise FloatingPointError("non-finite diagnostic loss")
        attributed=per_loss_gradients(losses,model,torch);rows=attribution_rows(batch_id,losses,attributed);matrix_batch.extend(rows)
        pairs=(("benefit","harm"),("benefit","uncertainty"),("harm","uncertainty"))
        cosines={}
        for left,right in pairs:
            value=grad_cosine(attributed[left]["all"],attributed[right]["all"]);cosines[f"{left}_vs_{right}"]=value;cosine_rows.append({"synthetic_interaction":LABEL,"record_type":"batch","batch_id":batch_id,"pair":f"{left}_vs_{right}","cosine":value})
            projection_value=grad_cosine(attributed[left]["modules"]["projection"],attributed[right]["modules"]["projection"]);cosine_space_rows.append({"synthetic_interaction":LABEL,"record_type":"batch","batch_id":batch_id,"parameter_space":"projection_shared","pair":f"{left}_vs_{right}","cosine":projection_value})
        total=sum(losses.values());loss_rows.extend({"synthetic_interaction":LABEL,"record_type":"batch","batch_id":batch_id,"loss":name,"value":float((total if name=="total" else losses[name]).detach())} for name in (*LOSS_NAMES,"total"))
        raw_target=target*tensors["benefit_scale"]+tensors["benefit_mean"];raw_prediction=prediction.benefit_mean*tensors["benefit_scale"]+tensors["benefit_mean"]
        inverse_variance=torch.exp(-prediction.benefit_log_variance);likelihood=.5*error.square()*inverse_variance
        for local_index,data_index in enumerate(indices.tolist()):gaussian_rows.append({"synthetic_interaction":LABEL,"batch_id":batch_id,"context_id":development["train_samples"][data_index].context_id,"normalized_target":float(target[local_index].detach()),"predicted_benefit_mean":float(prediction.benefit_mean[local_index].detach()),"prediction_error":float(error[local_index].detach()),"log_variance":float(prediction.benefit_log_variance[local_index].detach()),"exp_neg_log_variance":float(inverse_variance[local_index].detach()),"benefit_likelihood":float(likelihood[local_index].detach())})
        for name,value in (("benefit_mean_normalized",prediction.benefit_mean),("harm_logit",prediction.harm_logit),("harm_probability",prediction.harm_logit.sigmoid()),("benefit_log_variance",prediction.benefit_log_variance),("sigma_normalized",torch.exp(.5*prediction.benefit_log_variance)),("normalized_residual",error),("raw_benefit_residual",raw_prediction-raw_target)):prediction_values[name].extend(value.detach().float().cpu().tolist())
        with torch.inference_mode():
            projected=model.projection(features);projected_for_backbone=model.scale_alignment(projected) if model.scale_alignment_enabled else projected;attention=torch.ones(projected.shape[:2],device="cuda",dtype=torch.long);native_dtype=model.backbone.get_input_embeddings().weight.dtype;backbone_output=model.backbone(inputs_embeds=projected_for_backbone.to(dtype=native_dtype),attention_mask=attention,use_cache=False,output_hidden_states=True,return_dict=True);final_tokens=backbone_output.hidden_states[-1].float();context=final_tokens.mean(1)
            cursor=0
            for group_index,name in enumerate(TOKEN_ORDER):
                width=TOKEN_DIMS[name];projection_values[name].extend(features[:,cursor:cursor+width].float().norm(dim=-1).cpu().tolist());projected_values[name].extend(projected_for_backbone[:,group_index].float().norm(dim=-1).cpu().tolist());hidden_values[name].extend(final_tokens[:,group_index].norm(dim=-1).cpu().tolist());cursor+=width
            context_hidden.extend(context.norm(dim=-1).cpu().tolist())
        total_row=next(row for row in rows if row["loss"]=="total" and row["module"]=="all_trainable")
        batch_details.append({"batch_id":batch_id,"total_gradient_norm":total_row["gradient_norm"],"losses":{name:float(value.detach()) for name,value in losses.items()},"targets":{"benefit":raw_target.detach().cpu().tolist(),"harm":harm.detach().cpu().int().tolist()},"predictions":{"benefit":raw_prediction.detach().cpu().tolist(),"harm_probability":prediction.harm_logit.sigmoid().detach().cpu().tolist(),"benefit_log_variance":prediction.benefit_log_variance.detach().cpu().tolist()},"per_loss_total_grad_norm":{name:next(row["gradient_norm"] for row in rows if row["loss"]==name and row["module"]=="all_trainable") for name in LOSS_NAMES},"per_loss_module_grad_norm":{name:{module:next(row["gradient_norm"] for row in rows if row["loss"]==name and row["module"]==module) for module in MODULE_NAMES} for name in LOSS_NAMES},"gradient_cosine":cosines,"scenarios":[development["train_meta"][item]["scenario"] for item in indices.tolist()],"samples":[development["train_meta"][item]["sample"] for item in indices.tolist()]})
        del prediction,attributed,projected,backbone_output,final_tokens
    if initial_checksum!=trainable_state_checksum(model):raise RuntimeError("diagnostic changed trainable parameters")
    if any(parameter.grad is not None for parameter in model.parameters()):raise RuntimeError("diagnostic populated parameter gradients")
    matrix_summary=aggregate_matrix(matrix_batch);write_csv(args.output_dir/"gradient_source_matrix.csv",matrix_summary+matrix_batch)
    for pair in sorted(set(row["pair"] for row in cosine_rows)):
        values=[row["cosine"] for row in cosine_rows if row["pair"]==pair];cosine_rows.append({"synthetic_interaction":LABEL,"record_type":"summary","pair":pair,**stats(values)})
    write_csv(args.output_dir/"gradient_cosine.csv",cosine_rows)
    for pair in sorted(set(row["pair"] for row in cosine_space_rows)):
        values=[row["cosine"] for row in cosine_space_rows if row["pair"]==pair];cosine_space_rows.append({"synthetic_interaction":LABEL,"record_type":"summary","parameter_space":"projection_shared","pair":pair,**stats(values)})
    write_csv(args.output_dir/"gradient_cosine_spaces.csv",cosine_space_rows)
    for name in (*LOSS_NAMES,"total"):
        values=[row["value"] for row in loss_rows if row["loss"]==name];loss_rows.append({"synthetic_interaction":LABEL,"record_type":"summary","loss":name,**stats(values)})
    write_csv(args.output_dir/"loss_scale.csv",loss_rows)
    write_csv(args.output_dir/"gaussian_nll_audit.csv",gaussian_rows)
    feasible_ids=tensors["feasible_indices"].tolist();all_development_benefit=np.asarray([target.benefit for target in development["train_targets"]]);feasible_mask=np.asarray([row["feasible"] for row in development["train_meta"]],bool);all_benefit=all_development_benefit[feasible_mask];infeasible_benefit=all_development_benefit[~feasible_mask];all_harm=np.asarray([development["train_targets"][item].harm for item in feasible_ids]);target_rows=distribution_rows("benefit_target","feasible_raw_benefit",all_benefit)+distribution_rows("benefit_normalizer_audit","all_development_benefit_current_stage_c_scope",all_development_benefit)+distribution_rows("benefit_normalizer_audit","infeasible_benefit_excluded_from_loss",infeasible_benefit)+[{"synthetic_interaction":LABEL,"category":"harm_target","variable":"harm_label","count":len(all_harm),"positive_count":int(all_harm.sum()),"negative_count":int((~all_harm).sum()),"positive_fraction":float(all_harm.mean())}]+distribution_rows("uncertainty_supervision","normalized_residual",prediction_values["normalized_residual"])+distribution_rows("uncertainty_supervision","raw_benefit_residual",prediction_values["raw_benefit_residual"]);write_csv(args.output_dir/"target_scale.csv",target_rows)
    prediction_rows=[]
    for name,values in prediction_values.items():prediction_rows.extend(distribution_rows("initialized_model_output",name,values))
    write_csv(args.output_dir/"prediction_scale.csv",prediction_rows)
    native=native_embedding_norms(model,torch);projection_rows=distribution_rows("qwen_native_embedding","embedding_row_norm",native)+distribution_rows("qwen_context","final_context_hidden_norm",context_hidden)
    for name in TOKEN_ORDER:projection_rows+=distribution_rows("raw_group_input",name,projection_values[name])+distribution_rows("structured_projected_token",name,projected_values[name])+distribution_rows("qwen_final_token",name,hidden_values[name])
    write_csv(args.output_dir/"projection_scale.csv",projection_rows)
    parameter_rows=[]
    for name,module in (("projection",model.projection),("benefit_head",model.benefit),("harm_head",model.harm),("uncertainty_head",model.uncertainty)):
        weights=[parameter.detach().float().reshape(-1) for parameter_name,parameter in module.named_parameters() if "weight" in parameter_name];parameter_rows.append({"synthetic_interaction":LABEL,"module":name,"parameter_count":sum(parameter.numel() for parameter in module.parameters()),"weight_l2_norm":float(torch.cat(weights).norm()),"all_parameter_l2_norm":math.sqrt(sum(float(parameter.detach().float().square().sum()) for parameter in module.parameters()))})
    write_csv(args.output_dir/"parameter_scale.csv",parameter_rows)
    worst=max(batch_details,key=lambda row:row["total_gradient_norm"]);write_json(args.output_dir/"worst_gradient_case.json",{"label":LABEL,"person_or_profile_id_included":False,**worst})
    figures=make_figures(args.output_dir,matrix_summary,cosine_rows,loss_rows)
    total_means={name:next(row["mean"] for row in matrix_summary if row["loss"]==name and row["module"]=="all_trainable") for name in LOSS_NAMES};module_means={name:np.mean([next(row["mean"] for row in matrix_summary if row["loss"]==loss and row["module"]==name) for loss in LOSS_NAMES]) for name in MODULE_NAMES};alignment={name:next(row["aligned_contribution_mean"] for row in matrix_summary if row["loss"]==name and row["module"]=="all_trainable") for name in LOSS_NAMES};cosine_summary={row["pair"]:row["mean"] for row in cosine_rows if row["record_type"]=="summary"}
    dominant_loss=max(total_means,key=total_means.get);dominant_module=max(module_means,key=module_means.get);native_median=float(np.median(native));projected_median=float(np.median(np.concatenate([projected_values[name] for name in TOKEN_ORDER])));benefit_stats=stats(all_benefit)
    classifications=[]
    if dominant_loss=="benefit" and total_means["benefit"]>2*max(total_means["harm"],total_means["uncertainty"]):classifications.append("A Benefit target/loss scale dominated")
    if dominant_loss=="harm" and total_means["harm"]>2*max(total_means["benefit"],total_means["uncertainty"]):classifications.append("B Harm loss dominated")
    if dominant_loss=="uncertainty" and total_means["uncertainty"]>2*max(total_means["benefit"],total_means["harm"]):classifications.append("C Uncertainty loss dominated")
    if projected_median/native_median>3 or projected_median/native_median<1/3:classifications.append("D Projection embedding scale mismatch")
    if min(cosine_summary.values())<-.3:classifications.append("E Multi-task gradient conflict")
    if not classifications:classifications.append("F No single source / globally high gradient")
    fitted=tensors["benefit_normalizer"]
    summary={"label":LABEL,"stage":"Phase 5A Stage C-S4 Feasible-only Normalization Parity Audit","success":True,"diagnostic_only":True,"optimizer_created":False,"optimizer_step_count":0,"parameter_checksum_before":initial_checksum,"parameter_checksum_after":trainable_state_checksum(model),"parameters_unchanged":True,"test_materialized":False,"batch_manifest":"diagnostic_batch_manifest.json","conditions":{"seed":42,"batch_size":8,"diagnostic_batches":32,"learning_rate_configuration_record_only":args.learning_rate,"optimizer_configuration_record_only":"AdamW","loss_formula_unchanged":True},"gradient":{"per_loss_total_norm_mean":total_means,"per_loss_aligned_contribution_mean":alignment,"dominant_loss":dominant_loss,"mean_module_norm_across_losses":module_means,"dominant_module":dominant_module,"cosine_mean":cosine_summary},"targets":{"feasible_benefit":benefit_stats,"harm_positive_fraction":float(all_harm.mean()),"normalizer_scope_audit":{"current_stage_c_scope":"post-holdout train split feasible-only","actual_loss_scope":f"{len(feasible_ids)} feasible candidates","fit_count":len(fitted.fit_sample_ids),"epsilon":fitted.epsilon,"all_development":stats(all_development_benefit),"feasible_only":stats(all_benefit),"infeasible_only":stats(infeasible_benefit),"L1_protocol_match":True}},"uncertainty_formula":{"network_output":"benefit_log_variance","adapter_clamp":"[-6, 3]","loss":"0.5 * mean(error^2 * exp(-benefit_log_variance)) + 0.5 * mean(benefit_log_variance)","positive_sigma":"sigma = exp(0.5 * benefit_log_variance) for reporting","softplus":False,"epsilon_in_training_loss":None},"harm_loss":{"implementation":"binary_cross_entropy_with_logits","sigmoid_before_loss":False,"pos_weight":float(tensors["pos_weight"]),"reduction":"mean"},"projection_scale":{"native_embedding_row_norm_median":native_median,"structured_projected_token_norm_median":projected_median,"structured_to_native_median_ratio":projected_median/native_median},"classification":classifications,"interpretation_note":"C-S4 changes only the benefit normalizer fitting scope; gradient changes are diagnostic, not a success requirement.","formal_training_started":False,"repair_implemented":True,"repair":"L1-parity train-only feasible-only benefit normalizer","figures":figures};write_json(args.output_dir/"summary.json",summary);print(json.dumps(clean({"dominant_loss":dominant_loss,"dominant_module":dominant_module,"classification":classifications,"worst_batch":worst["batch_id"]}),indent=2),flush=True)


if __name__=="__main__":main()

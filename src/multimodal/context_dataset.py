"""Leakage-safe Phase 5 context dataset and named complex-context splits."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.multimodal.context_schema import StructuredContextTokens,prepare_context_batch,validate_branch_split_isolation

CONTEXT_SPLITS=("C1_seen_motion_seen_action","C2_unseen_motion_action","C3_unseen_person_seen_context","C4_unseen_person_unseen_motion_action","C5_compound_occlusion_turn_speed","C6_partial_functional_observation")

@dataclass(frozen=True)
class ContextTarget:
    benefit:float;harm:bool;auxiliary:np.ndarray

@dataclass(frozen=True)
class ContextDataset:
    samples:tuple[StructuredContextTokens,...];targets:tuple[ContextTarget,...];split_name:str
    def __post_init__(self)->None:
        if len(self.samples)!=len(self.targets):raise ValueError("sample/target length mismatch")
        if any(sample.context_split!=self.split_name for sample in self.samples):raise ValueError("dataset contains wrong split")
        validate_branch_split_isolation(list(self.samples))
    def features(self)->np.ndarray:return prepare_context_batch(self.samples)
    def benefit(self)->np.ndarray:return np.asarray([target.benefit for target in self.targets],np.float32)
    def harm(self)->np.ndarray:return np.asarray([target.harm for target in self.targets],np.float32)
    def auxiliary(self)->np.ndarray:return np.stack([target.auxiliary for target in self.targets]).astype(np.float32)


@dataclass(frozen=True)
class BenefitNormalizer:
    """Frozen L1 benefit-target contract: train-only, feasible-only, population std."""

    mean:float
    scale:float
    raw_std:float
    epsilon:float
    fit_sample_ids:tuple[str,...]

    def transform(self,values:np.ndarray)->np.ndarray:
        return (np.asarray(values,np.float32)-self.mean)/self.scale

    def inverse_transform(self,values:np.ndarray)->np.ndarray:
        return np.asarray(values,np.float32)*self.scale+self.mean


def fit_benefit_normalizer(
    samples:list[StructuredContextTokens]|tuple[StructuredContextTokens,...],
    targets:list[ContextTarget]|tuple[ContextTarget,...],
    meta:list[dict]|tuple[dict,...],
    *,epsilon:float=1e-4,
)->BenefitNormalizer:
    """Fit exactly as frozen L1: after holdouts, on feasible train candidates only."""
    if not (len(samples)==len(targets)==len(meta)):
        raise ValueError("normalizer inputs must have equal lengths")
    if not samples:
        raise ValueError("normalizer fit set is empty")
    if any(sample.context_id.split(":",1)[0]!="train" for sample in samples):
        raise ValueError("benefit normalizer may only be fit on the train split")
    feasible=np.asarray([bool(row["feasible"]) for row in meta],bool)
    if not feasible.any():
        raise ValueError("benefit normalizer requires feasible train candidates")
    values=np.asarray([target.benefit for target in targets],np.float32)[feasible]
    raw_std=float(values.std())
    return BenefitNormalizer(
        mean=float(values.mean()),scale=max(raw_std,float(epsilon)),raw_std=raw_std,
        epsilon=float(epsilon),fit_sample_ids=tuple(sample.context_id for sample,keep in zip(samples,feasible) if keep),
    )

def validate_global_split_isolation(datasets:list[ContextDataset])->None:
    validate_branch_split_isolation([sample for dataset in datasets for sample in dataset.samples])

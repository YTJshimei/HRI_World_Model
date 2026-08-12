"""Leakage-safe Phase 5 context dataset and named complex-context splits."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.multimodal.context_schema import StructuredContextTokens,validate_branch_split_isolation

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
    def features(self)->np.ndarray:return np.stack([sample.flattened() for sample in self.samples]).astype(np.float32)
    def benefit(self)->np.ndarray:return np.asarray([target.benefit for target in self.targets],np.float32)
    def harm(self)->np.ndarray:return np.asarray([target.harm for target in self.targets],np.float32)
    def auxiliary(self)->np.ndarray:return np.stack([target.auxiliary for target in self.targets]).astype(np.float32)

def validate_global_split_isolation(datasets:list[ContextDataset])->None:
    validate_branch_split_isolation([sample for dataset in datasets for sample in dataset.samples])

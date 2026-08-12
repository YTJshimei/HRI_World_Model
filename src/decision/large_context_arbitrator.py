"""Safety-preserving arbitration driven by context-value estimates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np


class ContextDecisionMode(str,Enum):
    GENERIC_SAFE="GENERIC_SAFE"
    PERSONALIZED_CONTEXT_APPROVED="PERSONALIZED_CONTEXT_APPROVED"
    RULE_FALLBACK="RULE_FALLBACK"
    ABSTAIN="ABSTAIN"


@dataclass(frozen=True)
class ContextDecision:
    selected_index:int|None;selected_action:int|None;mode:ContextDecisionMode;personalization_approved:bool


def arbitrate_large_context(
    action_ids:np.ndarray,feasible_action_mask:np.ndarray,generic_cost:np.ndarray,
    personalized_cost:np.ndarray,benefit_mean:np.ndarray,harm_probability:np.ndarray,
    benefit_threshold:float,harm_threshold:float,
)->ContextDecision:
    """Approve personalized ranking only inside the frozen Phase 4 safe set."""
    actions=np.asarray(action_ids,int);mask=np.asarray(feasible_action_mask,bool)
    arrays=[np.asarray(x,float) for x in (generic_cost,personalized_cost,benefit_mean,harm_probability)]
    if any(x.shape!=actions.shape for x in arrays) or mask.shape!=actions.shape:raise ValueError("candidate arrays must align")
    if not mask.any():return ContextDecision(None,None,ContextDecisionMode.ABSTAIN,False)
    valid=np.flatnonzero(mask);generic=int(valid[np.lexsort((actions[valid],arrays[0][valid]))][0])
    approved=mask & (arrays[2]>=benefit_threshold) & (arrays[3]<=harm_threshold)
    adjusted=arrays[1]-np.maximum(arrays[2],0.)
    if approved.any():
        candidate=int(np.flatnonzero(approved)[np.lexsort((actions[approved],adjusted[approved]))][0])
        if adjusted[candidate]<arrays[0][generic]:return ContextDecision(candidate,int(actions[candidate]),ContextDecisionMode.PERSONALIZED_CONTEXT_APPROVED,True)
    return ContextDecision(generic,int(actions[generic]),ContextDecisionMode.GENERIC_SAFE,False)

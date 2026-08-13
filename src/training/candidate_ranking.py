"""Episode-local ranking objective for the Phase 5B-1.6 intervention."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


LAMBDA_RANK = 0.25


@dataclass(frozen=True)
class RankingLossAudit:
    episode_count: int
    pair_count: int
    feasible_candidate_count: int


def pairwise_logistic_ranking_loss(
    predicted_benefit: torch.Tensor,
    target_benefit: torch.Tensor,
    episode_ids: list[str] | tuple[str, ...],
    feasible: torch.Tensor,
) -> tuple[torch.Tensor, RankingLossAudit]:
    """Mean pairwise logistic loss per episode, then across episodes.

    Only feasible candidates from the same episode may form a pair. Exact
    target ties are excluded. Targets are training-only arguments and are not
    part of the model forward/runtime input contract.
    """
    if predicted_benefit.ndim != 1 or target_benefit.ndim != 1 or feasible.ndim != 1:
        raise ValueError("predicted_benefit, target_benefit and feasible must be 1-D")
    if not (len(predicted_benefit) == len(target_benefit) == len(feasible) == len(episode_ids)):
        raise ValueError("ranking inputs must have identical lengths")
    if feasible.dtype != torch.bool:
        raise ValueError("feasible must be a boolean tensor")

    episode_losses: list[torch.Tensor] = []
    pair_count = 0
    ordered_ids = list(dict.fromkeys(episode_ids))
    for episode_id in ordered_ids:
        index = torch.tensor(
            [i for i, value in enumerate(episode_ids) if value == episode_id],
            dtype=torch.long,
            device=predicted_benefit.device,
        )
        index = index[feasible[index]]
        if len(index) < 2:
            continue
        left, right = torch.triu_indices(len(index), len(index), offset=1, device=index.device)
        target_delta = target_benefit[index[left]] - target_benefit[index[right]]
        non_tie = target_delta != 0
        if not bool(non_tie.any()):
            continue
        prediction_delta = predicted_benefit[index[left]] - predicted_benefit[index[right]]
        signed = target_delta[non_tie].sign()
        losses = F.softplus(-signed * prediction_delta[non_tie])
        episode_losses.append(losses.mean())
        pair_count += int(non_tie.sum())

    if episode_losses:
        loss = torch.stack(episode_losses).mean()
    else:
        loss = predicted_benefit.sum() * 0.0
    return loss, RankingLossAudit(
        episode_count=len(episode_losses),
        pair_count=pair_count,
        feasible_candidate_count=int(feasible.sum()),
    )

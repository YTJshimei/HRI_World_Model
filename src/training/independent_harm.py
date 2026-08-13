"""Isolated training primitives for the Phase 5B independent harm-v2 head."""
from __future__ import annotations


def harm_v2_target(sample) -> bool:
    """Return and verify GT-unsafe OR adverse-human-response semantics."""
    metadata = sample.split_metadata
    derived = bool(
        sample.targets.gt_unsafe
        or metadata["excessive_deceleration_evaluation_only"]
        or metadata["abrupt_lateral_response_evaluation_only"]
        or metadata["abrupt_heading_change_evaluation_only"]
    )
    recorded = bool(metadata["harm_v2_evaluation_only"])
    if derived != recorded:
        raise RuntimeError("harm_v2 metadata disagrees with its frozen semantic definition")
    return derived


def unweighted_harm_v2_loss(logits, target, torch):
    """Plain BCE-with-logits: no class/focal/sample weighting."""
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target.float())

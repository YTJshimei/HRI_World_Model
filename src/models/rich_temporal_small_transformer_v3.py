"""Manifest-v3 R1 model: frozen R1-v2 body with a 12D action input."""
from __future__ import annotations

from torch import nn

from src.models.rich_temporal_small_transformer import RichTemporalSmallTransformer
from src.multimodal.phase5b_v3_dataset import V3_CANDIDATE_ACTION_DIM, V3_STREAM_DIMS


class RichTemporalSmallTransformerV3(RichTemporalSmallTransformer):
    """Exact v2 architecture except for 11->12 action projection width.

    Calling ``super`` first preserves the fixed-seed initialization of every
    shared parameter.  The replacement action projection is newly initialized
    from the same deterministic RNG stream; no v2 checkpoint is expanded.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_projection = nn.Linear(V3_CANDIDATE_ACTION_DIM, self.d_model)

    def architecture_audit(self) -> dict[str, object]:
        audit = super().architecture_audit()
        audit.update(
            {
                "model": "RichTemporalSmallTransformerV3",
                "architecture_delta_from_v2": "candidate action projection input 11 -> 12 only",
                "stream_input_shapes": {name: list(shape) for name, shape in V3_STREAM_DIMS.items()},
            }
        )
        audit["projection_shapes"]["candidate_action"] = [12, 128]
        audit["parameter_count"] = sum(parameter.numel() for parameter in self.parameters())
        audit["trainable_parameter_count"] = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return audit

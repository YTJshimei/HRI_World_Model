"""Fixed Phase 5B-1 small transformer for leakage-safe temporal context."""
from __future__ import annotations

import torch
from torch import nn

from src.models.large_context_adapter import ContextValuePrediction
from src.multimodal.temporal_schema import STREAM_DIMS


HISTORY_STREAMS = (
    "skeleton_history", "human_motion_history", "robot_history",
    "functional_history", "visibility_history", "wm_diagnostic_history",
    "interaction_history",
)


class RichTemporalSmallTransformer(nn.Module):
    """The preregistered B1 architecture; no architecture-search switches."""

    def __init__(
        self, d_model: int = 128, layers: int = 2, heads: int = 4,
        ffn_dim: int = 256, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if (d_model, layers, heads, ffn_dim, dropout) != (128, 2, 4, 256, 0.1):
            raise ValueError("Phase5B-1 architecture is frozen at 128/2/4/256/0.1")
        self.d_model = d_model
        input_dims = {
            name: (51 if name == "skeleton_history" else STREAM_DIMS[name][-1])
            for name in HISTORY_STREAMS
        }
        self.history_projections = nn.ModuleDict({name: nn.Linear(width, d_model) for name, width in input_dims.items()})
        self.future_projection = nn.Linear(STREAM_DIMS["candidate_robot_future"][-1], d_model)
        self.action_projection = nn.Linear(STREAM_DIMS["candidate_action"][-1], d_model)
        self.scene_projection = nn.Linear(STREAM_DIMS["scene_context"][-1], d_model)
        self.skeleton_missing = nn.Parameter(torch.zeros(17, 3))
        self.missing_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(input_dims[name]))
            for name in HISTORY_STREAMS if name != "skeleton_history"
        })
        self.modality_embedding = nn.Parameter(torch.zeros(len(HISTORY_STREAMS) + 1, d_model))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.time_projection = nn.Sequential(nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(d_model))
        self.fusion = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.benefit = nn.Linear(d_model, 1)
        self.uncertainty = nn.Linear(d_model, 1)
        self.harm = nn.Linear(d_model, 1)

    @staticmethod
    def _require(batch: dict[str, object]) -> None:
        if not all(name in batch for name in ("streams", "masks", "timestamps")):
            raise ValueError("temporal batch requires streams, masks and timestamps")

    @staticmethod
    def _masked_pool(tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return (tokens * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)

    def _encode_with_audit(self, batch: dict[str, object]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the unchanged encoder while retaining runtime-valid intermediates.

        The returned dictionary is audit-only.  It contains only projections and
        concatenations that already exist in the frozen computation; no learned
        audit layer or target-derived value is introduced.
        """
        self._require(batch)
        streams, masks, timestamps = batch["streams"], batch["masks"], batch["timestamps"]
        history_time = timestamps["history"].float().unsqueeze(-1)
        padding = masks["history_padding_mask"].bool()
        tokens, valid_tokens, history_projected, history_valid = [], [], {}, {}
        for modality, name in enumerate(HISTORY_STREAMS):
            values = streams[name].float()
            valid = masks[name].bool()
            if name == "skeleton_history":
                missing = self.skeleton_missing.view(1, 1, 17, 3)
                values = torch.where(valid, values, missing).flatten(-2)
                token_valid = valid.flatten(-2).any(-1) & padding
            else:
                missing = self.missing_embeddings[name].view(1, 1, -1)
                values = torch.where(valid, values, missing)
                token_valid = valid.any(-1) & padding
            projected = self.history_projections[name](values)
            projected = projected + self.modality_embedding[modality] + self.time_projection(history_time)
            history_projected[name] = projected
            history_valid[name] = token_valid
            tokens.append(projected); valid_tokens.append(token_valid)

        future = streams["candidate_robot_future"].float()
        future_mask = masks["candidate_robot_future"].bool()
        future_valid = future_mask.any(-1) & masks["candidate_future_valid_mask"].bool()
        future = torch.where(future_mask, future, torch.zeros_like(future))
        future_token = self.future_projection(future)
        future_token = future_token + self.modality_embedding[-1] + self.time_projection(timestamps["candidate_future"].float().unsqueeze(-1))
        tokens.append(future_token); valid_tokens.append(future_valid)

        sequence = torch.cat(tokens, dim=1)
        valid = torch.cat(valid_tokens, dim=1)
        if (~valid).all(dim=1).any():
            raise ValueError("a temporal sample cannot have every token masked")
        encoded = self.transformer(sequence, src_key_padding_mask=~valid)
        pooled = self._masked_pool(encoded, valid)

        action_mask = masks["candidate_action"].bool()
        scene_mask = masks["scene_context"].bool()
        action = self.action_projection(torch.where(action_mask, streams["candidate_action"].float(), 0.0))
        scene = self.scene_projection(torch.where(scene_mask, streams["scene_context"].float(), 0.0))
        prefinal = torch.cat((pooled, action, scene), dim=-1)
        context = self.fusion(prefinal)
        history_pools = {name: self._masked_pool(history_projected[name], history_valid[name]) for name in HISTORY_STREAMS}
        future_pool = self._masked_pool(future_token, future_valid)
        audit = {
            "R0_FINAL_FUSED": context,
            "R1_HISTORY_CONTEXT_PREFUSION": torch.cat((*[history_pools[name] for name in HISTORY_STREAMS], scene), dim=-1),
            "R2_CANDIDATE_PREFUSION": torch.cat((future_pool, action), dim=-1),
            "R3_SKELETON_PROJECTED_POOL": history_pools["skeleton_history"],
            "R4_MOTION_PROJECTED_POOL": history_pools["human_motion_history"],
            "R5_FUNCTIONAL_INTERACTION_DIAGNOSTIC": torch.cat((history_pools["functional_history"], history_pools["interaction_history"], history_pools["wm_diagnostic_history"]), dim=-1),
            "R6_PREFINAL_FUSION_CONCAT": prefinal,
            "R6_JOINT_TEMPORAL_POOL": pooled,
        }
        return context, audit

    def encode(self, batch: dict[str, object]) -> torch.Tensor:
        return self._encode_with_audit(batch)[0]

    def audit_representations(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        """Expose frozen runtime-valid intermediates without changing ``forward``."""
        return self._encode_with_audit(batch)[1]

    def forward(self, batch: dict[str, object]) -> ContextValuePrediction:
        context = self.encode(batch)
        auxiliary = context.new_zeros((len(context), 0))
        return ContextValuePrediction(
            context_embedding=context,
            benefit_mean=self.benefit(context).squeeze(-1),
            benefit_log_variance=self.uncertainty(context).squeeze(-1).clamp(-6, 3),
            harm_logit=self.harm(context).squeeze(-1),
            auxiliary=auxiliary,
        )

    def architecture_audit(self) -> dict[str, object]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {
            "model": "RichTemporalSmallTransformer", "d_model": 128,
            "transformer_layers": 2, "attention_heads": 4, "ffn_hidden_dim": 256,
            "dropout": 0.1, "history_frames": 20, "future_frames": 10,
            "history_token_count": 7 * 20, "candidate_future_token_count": 10,
            "transformer_token_count": 150, "pooling": "masked mean",
            "parameter_count": total, "trainable_parameter_count": trainable,
            "mask_usage": {
                "element_validity": True, "history_padding": True,
                "candidate_future_validity": True, "modality_validity": True,
                "occluded_skeleton_uses_learnable_missing_token": True,
                "attention_key_padding_mask": True,
            },
            "stream_input_shapes": {name: list(shape) for name, shape in STREAM_DIMS.items()},
            "projection_shapes": {
                **{name: [input_dims, 128] for name, input_dims in {
                    key: (51 if key == "skeleton_history" else STREAM_DIMS[key][-1]) for key in HISTORY_STREAMS
                }.items()},
                "candidate_robot_future": [5, 128], "candidate_action": [11, 128], "scene_context": [8, 128],
            },
            "output_shapes": {"context_embedding": ["B", 128], "benefit_mean": ["B"],
                              "benefit_log_variance": ["B"], "harm_logit": ["B"]},
            "whole_sample_flattened": False,
        }

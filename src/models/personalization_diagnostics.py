"""Oracle and meta-conditioning diagnostic models for synthetic Phase 4B.5."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.data.personalization_diagnostics import TRUE_EFFECT_DESCRIPTOR_DIM
from src.data.skeleton_schema import NUM_JOINTS
from src.data.skeleton_schema import compute_root
from src.models.personalized_response_world_model import (
    PersonalizedInteractionPrediction,
    PersonalizedRootPoseWorldModel,
)


@dataclass(frozen=True)
class ResponseOraclePrediction:
    future_by_action: torch.Tensor
    natural_future: torch.Tensor
    root_log_std_by_action: torch.Tensor
    action_effect_root_log_std_by_action: torch.Tensor
    z_person: torch.Tensor
    predicted_response_statistics: torch.Tensor


def prediction_from_encoded(
    model: PersonalizedRootPoseWorldModel,
    encoded: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    action_ids: torch.Tensor,
) -> PersonalizedInteractionPrediction:
    future, root_log_std = model.query_actions(encoded, action_ids, True)
    keep = torch.zeros(
        (action_ids.shape[0], 1), dtype=action_ids.dtype, device=action_ids.device
    )
    natural, natural_log_std = model.query_actions(encoded, keep, True)
    effect_log_std = 0.5 * torch.logaddexp(
        2.0 * root_log_std, 2.0 * natural_log_std.expand_as(root_log_std)
    )
    return PersonalizedInteractionPrediction(
        future, natural[:, 0], root_log_std, effect_log_std, encoded[-1]
    )


class MetaPersonalizedWorldModel(PersonalizedRootPoseWorldModel):
    """Phase 4B P2 architecture with explicit generic/personalized paired queries."""

    def __init__(self, history_frames: int = 20, future_frames: int = 10) -> None:
        super().__init__("P2", history_frames=history_frames, future_frames=future_frames)

    def paired_forward(
        self,
        history: torch.Tensor,
        robot_history: torch.Tensor,
        action_ids: torch.Tensor,
        confidence: torch.Tensor,
        visibility: torch.Tensor,
        support_features: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> tuple[PersonalizedInteractionPrediction, PersonalizedInteractionPrediction]:
        encoded = self.encode_context(
            history, robot_history, confidence, visibility,
            support_features=support_features, support_mask=support_mask,
        )
        personalized = prediction_from_encoded(self, encoded, action_ids)
        null = self.personal_response_encoder.null_person[None].expand(history.shape[0], -1)
        generic_encoded = (*encoded[:-1], null)
        generic = prediction_from_encoded(self, generic_encoded, action_ids)
        return personalized, generic

    def interpolate_forward(
        self,
        encoded_without_person: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        z_person: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> PersonalizedInteractionPrediction:
        return prediction_from_encoded(self, (*encoded_without_person, z_person), action_ids)


class OracleEffectWorldModel(nn.Module):
    """Predict coordinates from true response descriptors, never GT future coordinates."""

    def __init__(self, history_frames: int = 20, future_frames: int = 10) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.natural_backbone = PersonalizedRootPoseWorldModel(
            "P0", history_frames=history_frames, future_frames=future_frames
        )
        self.effect_head = nn.Linear(
            TRUE_EFFECT_DESCRIPTOR_DIM, TRUE_EFFECT_DESCRIPTOR_DIM
        )
        with torch.no_grad():
            self.effect_head.weight.copy_(torch.eye(TRUE_EFFECT_DESCRIPTOR_DIM))
            self.effect_head.bias.zero_()
        for parameter in self.effect_head.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        history: torch.Tensor,
        robot_history: torch.Tensor,
        action_ids: torch.Tensor,
        confidence: torch.Tensor,
        visibility: torch.Tensor,
        effect_descriptors: torch.Tensor,
    ) -> PersonalizedInteractionPrediction:
        base = self.natural_backbone(
            history, robot_history, action_ids, confidence, visibility
        )
        descriptor = self.effect_head(effect_descriptors)
        natural_root = compute_root(base.natural_future)
        natural_local = base.natural_future - natural_root[..., None, :]
        response_velocity = descriptor[..., :2]
        delay_frames = torch.ceil(descriptor[..., 6] * 10.0 - 1e-6)
        frame = torch.arange(
            self.future_frames, device=history.device, dtype=history.dtype
        )[None, None]
        active_time = torch.relu((frame - delay_frames[..., None] + 1.0) * 0.1)
        adaptation = 1.0 - torch.exp(
            -descriptor[..., 8, None] * active_time
        )
        root_offset_xy = (
            active_time[..., None] * adaptation[..., None]
            * response_velocity[..., None, :]
        )
        root_offset = torch.zeros(
            (*root_offset_xy.shape[:-1], 3), device=history.device,
            dtype=history.dtype,
        )
        root_offset[..., :2] = root_offset_xy
        yaw = descriptor[..., 5, None] * adaptation
        local = natural_local[:, None].expand(
            -1, action_ids.shape[1], -1, -1, -1
        ).clone()
        x, y = local[..., 0].clone(), local[..., 1].clone()
        cosine, sine = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
        local[..., 0] = cosine * x - sine * y
        local[..., 1] = sine * x + cosine * y
        active = (action_ids != 0).to(history.dtype)[..., None, None]
        future = (
            natural_root[:, None, :, None, :]
            + root_offset[..., None, :] * active[..., None]
            + local
        )
        return PersonalizedInteractionPrediction(
            future,
            base.natural_future,
            base.root_log_std_by_action,
            base.action_effect_root_log_std_by_action,
            base.z_person,
        )


class ResponseOracleWorldModel(nn.Module):
    """Explicit profile -> response-statistics -> conditioning oracle."""

    def __init__(self, history_frames: int = 20, future_frames: int = 10) -> None:
        super().__init__()
        self.backbone = PersonalizedRootPoseWorldModel(
            "P0", history_frames=history_frames, future_frames=future_frames
        )
        self.profile_to_response_statistics = nn.Sequential(
            nn.Linear(7, 48), nn.GELU(), nn.Linear(48, 7)
        )
        self.response_statistics_to_person = nn.Sequential(
            nn.Linear(7, 48), nn.GELU(), nn.Linear(48, 32), nn.Tanh()
        )

    def forward(
        self,
        history: torch.Tensor,
        robot_history: torch.Tensor,
        action_ids: torch.Tensor,
        confidence: torch.Tensor,
        visibility: torch.Tensor,
        profile_parameters: torch.Tensor,
    ) -> ResponseOraclePrediction:
        encoded = self.backbone.encode_context(
            history, robot_history, confidence, visibility
        )
        statistics = self.profile_to_response_statistics(profile_parameters)
        z_person = self.response_statistics_to_person(statistics)
        output = prediction_from_encoded(
            self.backbone, (*encoded[:-1], z_person), action_ids
        )
        return ResponseOraclePrediction(
            output.future_by_action,
            output.natural_future,
            output.root_log_std_by_action,
            output.action_effect_root_log_std_by_action,
            z_person,
            statistics,
        )

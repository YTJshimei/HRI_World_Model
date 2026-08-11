"""Phase 4A synthetic-interaction baselines with encode-once action queries."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.data.skeleton_schema import NUM_JOINTS, hip_joints
from src.data.synthetic_interaction import VIRTUAL_PERSON_PROFILES
from src.models.skeleton_baselines import masked_centered_input
from src.models.skeleton_transformer import SpatialTemporalSkeletonEncoder


@dataclass(frozen=True)
class InteractionPrediction:
    future_by_action: torch.Tensor
    natural_future: torch.Tensor


class HumanLSTMEncoder(nn.Module):
    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.missing_token = nn.Parameter(torch.zeros(NUM_JOINTS, 3))
        self.encoder = nn.LSTM(NUM_JOINTS * 5, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(
        self,
        history: torch.Tensor,
        confidence: torch.Tensor,
        visibility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, root = masked_centered_input(
            history, confidence, visibility, self.missing_token
        )
        _, (hidden, _) = self.encoder(features.flatten(start_dim=2))
        return hidden[-1], root


class ActionConditioning(nn.Module):
    """Encode robot/profile once and produce lightweight per-action FiLM vectors."""

    def __init__(self, output_size: int, action_dim: int = 16, profile_dim: int = 8) -> None:
        super().__init__()
        self.robot_encoder = nn.GRU(7, 32, batch_first=True)
        self.action_embedding = nn.Embedding(7, action_dim)
        profile_features = torch.tensor(
            [
                (
                    profile.preferred_distance,
                    profile.distance_sensitivity,
                    profile.speed_response_gain,
                    profile.response_delay,
                    profile.lateral_avoidance_gain,
                    profile.turn_sensitivity,
                    profile.adaptation_rate,
                )
                for profile in VIRTUAL_PERSON_PROFILES
            ],
            dtype=torch.float32,
        )
        feature_mean = profile_features.mean(dim=0, keepdim=True)
        feature_std = profile_features.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.register_buffer("profile_features", (profile_features - feature_mean) / feature_std)
        self.profile_projection = nn.Linear(7, profile_dim)
        self.film = nn.Linear(32 + action_dim + profile_dim, output_size * 2)
        self.output_size = output_size

    def encode_robot_profile(
        self, robot_history: torch.Tensor, profile_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.robot_encoder(robot_history)
        return hidden[-1], self.profile_projection(self.profile_features[profile_ids])

    def query(
        self,
        robot_context: torch.Tensor,
        profile_context: torch.Tensor,
        action_ids: torch.Tensor,
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_context = self.action_embedding(action_ids)
        if not enabled:
            action_context = self.action_embedding(torch.zeros_like(action_ids))
        robot = robot_context[:, None].expand(-1, action_ids.shape[1], -1)
        profile = profile_context[:, None].expand(-1, action_ids.shape[1], -1)
        scale, shift = self.film(
            torch.cat((robot, profile, action_context), dim=-1)
        ).chunk(2, dim=-1)
        return scale, shift


def _decode_global(
    relative: torch.Tensor, root: torch.Tensor, future_frames: int
) -> torch.Tensor:
    return relative.view(
        relative.shape[0], relative.shape[1], future_frames, NUM_JOINTS, 3
    ) + root[:, None, None, None, :]


class ActionAgnosticHumanModel(nn.Module):
    """W0: predict natural motion once and repeat it for every candidate action."""

    def __init__(self, hidden_size: int = 128, future_frames: int = 10) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.human_encoder = HumanLSTMEncoder(hidden_size)
        self.head = nn.Linear(hidden_size, future_frames * NUM_JOINTS * 3)

    def forward(
        self,
        history: torch.Tensor,
        robot_history: torch.Tensor,
        action_ids: torch.Tensor,
        confidence: torch.Tensor,
        visibility: torch.Tensor,
        profile_ids: torch.Tensor,
        action_conditioning: bool = True,
    ) -> InteractionPrediction:
        del robot_history, profile_ids, action_conditioning
        context, root = self.human_encoder(history, confidence, visibility)
        natural = self.head(context).view(
            history.shape[0], self.future_frames, NUM_JOINTS, 3
        ) + root[:, None, None, :]
        return InteractionPrediction(
            natural[:, None].expand(-1, action_ids.shape[1], -1, -1, -1), natural
        )


class ActionConditionedLSTM(nn.Module):
    """W1: S1-style temporal context with FiLM action queries."""

    def __init__(self, hidden_size: int = 128, future_frames: int = 10) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.human_encoder = HumanLSTMEncoder(hidden_size)
        self.conditioning = ActionConditioning(hidden_size)
        self.decoder = nn.Linear(hidden_size, future_frames * NUM_JOINTS * 3)

    def _query(
        self,
        human_context: torch.Tensor,
        root: torch.Tensor,
        robot_context: torch.Tensor,
        profile_context: torch.Tensor,
        action_ids: torch.Tensor,
        enabled: bool,
    ) -> torch.Tensor:
        scale, shift = self.conditioning.query(
            robot_context, profile_context, action_ids, enabled
        )
        conditioned = human_context[:, None] * (1.0 + 0.1 * torch.tanh(scale)) + shift
        return _decode_global(self.decoder(conditioned), root, self.future_frames)

    def forward(
        self,
        history: torch.Tensor,
        robot_history: torch.Tensor,
        action_ids: torch.Tensor,
        confidence: torch.Tensor,
        visibility: torch.Tensor,
        profile_ids: torch.Tensor,
        action_conditioning: bool = True,
    ) -> InteractionPrediction:
        human_context, root = self.human_encoder(history, confidence, visibility)
        robot_context, profile_context = self.conditioning.encode_robot_profile(
            robot_history, profile_ids
        )
        future = self._query(
            human_context, root, robot_context, profile_context, action_ids,
            action_conditioning,
        )
        keep_ids = torch.zeros(
            (history.shape[0], 1), dtype=action_ids.dtype, device=action_ids.device
        )
        natural = (
            self._query(
                human_context, root, robot_context, profile_context, keep_ids, True
            )[:, 0]
            if action_conditioning
            else future[:, 0]
        )
        return InteractionPrediction(future, natural)


class ActionConditionedRootPoseModel(nn.Module):
    """W2: S1-style root context plus S2-style local-pose auxiliary branch."""

    def __init__(
        self,
        history_frames: int = 20,
        future_frames: int = 10,
        root_hidden_size: int = 96,
        d_model: int = 48,
    ) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.root_encoder = HumanLSTMEncoder(root_hidden_size)
        self.local_encoder = SpatialTemporalSkeletonEncoder(
            d_model=d_model,
            nhead=4,
            spatial_layers=1,
            temporal_layers=2,
            history_frames=history_frames,
        )
        self.root_conditioning = ActionConditioning(root_hidden_size)
        self.local_film = nn.Linear(32 + 16 + 8, d_model * 2)
        self.root_head = nn.Linear(root_hidden_size, future_frames * 3)
        self.local_head = nn.Linear(d_model, future_frames * 3)

    def _query(
        self,
        root_context: torch.Tensor,
        local_context: torch.Tensor,
        anchor: torch.Tensor,
        robot_context: torch.Tensor,
        profile_context: torch.Tensor,
        action_ids: torch.Tensor,
        enabled: bool,
    ) -> torch.Tensor:
        root_scale, root_shift = self.root_conditioning.query(
            robot_context, profile_context, action_ids, enabled
        )
        conditioned_root = (
            root_context[:, None] * (1.0 + 0.1 * torch.tanh(root_scale)) + root_shift
        )
        root_future = self.root_head(conditioned_root).view(
            root_context.shape[0], action_ids.shape[1], self.future_frames, 3
        ) + anchor[:, None, None, :]

        action_context = self.root_conditioning.action_embedding(action_ids)
        if not enabled:
            action_context = self.root_conditioning.action_embedding(
                torch.zeros_like(action_ids)
            )
        robot = robot_context[:, None].expand(-1, action_ids.shape[1], -1)
        profile = profile_context[:, None].expand(-1, action_ids.shape[1], -1)
        scale, shift = self.local_film(
            torch.cat((robot, profile, action_context), dim=-1)
        ).chunk(2, dim=-1)
        conditioned_local = (
            local_context[:, None]
            * (1.0 + 0.1 * torch.tanh(scale)[:, :, None, :])
            + shift[:, :, None, :]
        )
        local = self.local_head(conditioned_local).view(
            root_context.shape[0], action_ids.shape[1], NUM_JOINTS,
            self.future_frames, 3,
        ).permute(0, 1, 3, 2, 4)
        pelvis = local[..., list(hip_joints), :].mean(dim=-2)
        local = local - pelvis[..., None, :]
        return root_future[..., None, :] + local

    def forward(
        self,
        history: torch.Tensor,
        robot_history: torch.Tensor,
        action_ids: torch.Tensor,
        confidence: torch.Tensor,
        visibility: torch.Tensor,
        profile_ids: torch.Tensor,
        action_conditioning: bool = True,
    ) -> InteractionPrediction:
        root_context, anchor = self.root_encoder(history, confidence, visibility)
        local_context, _ = self.local_encoder(history, confidence, visibility)
        robot_context, profile_context = self.root_conditioning.encode_robot_profile(
            robot_history, profile_ids
        )
        future = self._query(
            root_context, local_context, anchor, robot_context, profile_context,
            action_ids, action_conditioning,
        )
        keep_ids = torch.zeros(
            (history.shape[0], 1), dtype=action_ids.dtype, device=action_ids.device
        )
        natural = (
            self._query(
                root_context, local_context, anchor, robot_context, profile_context,
                keep_ids, True,
            )[:, 0]
            if action_conditioning
            else future[:, 0]
        )
        return InteractionPrediction(future, natural)


class ActionConditionedResidualModel(nn.Module):
    """W3: stable natural human forecast plus a learned robot-response residual."""

    def __init__(
        self, hidden_size: int = 128, effect_size: int = 64, future_frames: int = 10
    ) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.human_encoder = HumanLSTMEncoder(hidden_size)
        self.natural_head = nn.Linear(hidden_size, future_frames * NUM_JOINTS * 3)
        self.conditioning = ActionConditioning(effect_size)
        self.effect_context = nn.Linear(hidden_size, effect_size)
        self.effect_head = nn.Linear(effect_size, future_frames * NUM_JOINTS * 3)
        nn.init.normal_(self.effect_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.effect_head.bias)

    def forward(
        self,
        history: torch.Tensor,
        robot_history: torch.Tensor,
        action_ids: torch.Tensor,
        confidence: torch.Tensor,
        visibility: torch.Tensor,
        profile_ids: torch.Tensor,
        action_conditioning: bool = True,
    ) -> InteractionPrediction:
        human_context, root = self.human_encoder(history, confidence, visibility)
        natural = self.natural_head(human_context).view(
            history.shape[0], self.future_frames, NUM_JOINTS, 3
        ) + root[:, None, None, :]
        robot_context, profile_context = self.conditioning.encode_robot_profile(
            robot_history, profile_ids
        )
        scale, shift = self.conditioning.query(
            robot_context, profile_context, action_ids, action_conditioning
        )
        base_effect = self.effect_context(human_context)[:, None]
        conditioned = base_effect * (1.0 + 0.1 * torch.tanh(scale)) + shift
        residual = self.effect_head(conditioned).view(
            history.shape[0], action_ids.shape[1], self.future_frames, NUM_JOINTS, 3
        )
        active = (action_ids != 0).to(history.dtype)[..., None, None, None]
        if not action_conditioning:
            active = torch.zeros_like(active)
        residual = residual * active
        return InteractionPrediction(natural[:, None] + residual, natural)

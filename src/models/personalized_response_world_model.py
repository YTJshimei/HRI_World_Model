"""Leakage-controlled W2-style personal response world models for Phase 4B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from src.data.skeleton_schema import NUM_JOINTS, hip_joints
from src.models.action_conditioned_world_models import HumanLSTMEncoder
from src.models.personal_response_encoder import PersonalResponseEncoder
from src.models.skeleton_transformer import SpatialTemporalSkeletonEncoder


PersonalizationMode = Literal["P0", "P1", "P2", "P3"]


@dataclass(frozen=True)
class PersonalizedInteractionPrediction:
    future_by_action: torch.Tensor
    natural_future: torch.Tensor
    root_log_std_by_action: torch.Tensor
    action_effect_root_log_std_by_action: torch.Tensor
    z_person: torch.Tensor


class PersonalizedRootPoseWorldModel(nn.Module):
    """S1-root/S2-local backbone with controlled person-conditioning sources.

    P0 uses a generic null latent; P1 uses an identity lookup (seen-person only);
    P2 uses observable support; P3 uses explicit simulator parameters as oracle.
    """

    def __init__(
        self,
        mode: PersonalizationMode,
        history_frames: int = 20,
        future_frames: int = 10,
        number_of_seen_people: int = 0,
        root_hidden_size: int = 96,
        d_model: int = 48,
        person_dim: int = 32,
    ) -> None:
        super().__init__()
        if mode not in ("P0", "P1", "P2", "P3"):
            raise ValueError(f"unknown Phase 4B model mode: {mode}")
        if mode == "P1" and number_of_seen_people <= 0:
            raise ValueError("P1 requires number_of_seen_people")
        self.mode = mode
        self.future_frames = future_frames
        self.person_dim = person_dim
        self.root_encoder = HumanLSTMEncoder(root_hidden_size)
        self.local_encoder = SpatialTemporalSkeletonEncoder(
            d_model=d_model,
            nhead=4,
            spatial_layers=1,
            temporal_layers=2,
            history_frames=history_frames,
        )
        self.robot_encoder = nn.GRU(7, 32, batch_first=True)
        self.action_embedding = nn.Embedding(7, 16)
        self.generic_person = nn.Parameter(torch.zeros(person_dim))
        if mode == "P1":
            self.person_id_embedding = nn.Embedding(number_of_seen_people, person_dim)
        elif mode == "P2":
            self.personal_response_encoder = PersonalResponseEncoder(latent_size=person_dim)
        elif mode == "P3":
            self.oracle_encoder = nn.Sequential(
                nn.Linear(7, 48), nn.GELU(), nn.Linear(48, person_dim), nn.Tanh()
            )
        conditioning_size = 32 + 16 + person_dim
        self.root_film = nn.Linear(conditioning_size, root_hidden_size * 2)
        self.local_film = nn.Linear(conditioning_size, d_model * 2)
        self.root_head = nn.Linear(root_hidden_size, future_frames * 3)
        self.local_head = nn.Linear(d_model, future_frames * 3)
        self.root_log_std_head = nn.Linear(root_hidden_size, future_frames * 3)

    def _person_context(
        self,
        batch_size: int,
        support_features: torch.Tensor | None,
        support_mask: torch.Tensor | None,
        person_indices: torch.Tensor | None,
        oracle_parameters: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.mode == "P0":
            return self.generic_person[None].expand(batch_size, -1)
        if self.mode == "P1":
            if person_indices is None:
                raise ValueError("P1 requires seen-person indices")
            return self.person_id_embedding(person_indices)
        if self.mode == "P2":
            if support_features is None or support_mask is None:
                raise ValueError("P2 requires observable support features and mask")
            return self.personal_response_encoder(support_features, support_mask)
        if oracle_parameters is None:
            raise ValueError("P3 requires oracle simulator parameters")
        return self.oracle_encoder(oracle_parameters)

    def encode_context(
        self,
        history: torch.Tensor,
        robot_history: torch.Tensor,
        confidence: torch.Tensor,
        visibility: torch.Tensor,
        support_features: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        person_indices: torch.Tensor | None = None,
        oracle_parameters: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        root_context, anchor = self.root_encoder(history, confidence, visibility)
        local_context, _ = self.local_encoder(history, confidence, visibility)
        _, robot_hidden = self.robot_encoder(robot_history)
        person = self._person_context(
            history.shape[0], support_features, support_mask, person_indices,
            oracle_parameters,
        )
        return root_context, local_context, anchor, robot_hidden[-1], person

    def query_actions(
        self,
        encoded: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        action_ids: torch.Tensor,
        action_conditioning: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        root_context, local_context, anchor, robot_context, person = encoded
        query_actions = action_ids if action_conditioning else torch.zeros_like(action_ids)
        action = self.action_embedding(query_actions)
        robot = robot_context[:, None].expand(-1, action_ids.shape[1], -1)
        personal = person[:, None].expand(-1, action_ids.shape[1], -1)
        condition = torch.cat((robot, action, personal), dim=-1)

        root_scale, root_shift = self.root_film(condition).chunk(2, dim=-1)
        conditioned_root = (
            root_context[:, None] * (1.0 + 0.1 * torch.tanh(root_scale)) + root_shift
        )
        root_future = self.root_head(conditioned_root).view(
            root_context.shape[0], action_ids.shape[1], self.future_frames, 3
        ) + anchor[:, None, None, :]
        root_log_std = self.root_log_std_head(conditioned_root).view_as(root_future)
        root_log_std = root_log_std.clamp(min=-5.0, max=1.0)

        local_scale, local_shift = self.local_film(condition).chunk(2, dim=-1)
        conditioned_local = (
            local_context[:, None]
            * (1.0 + 0.1 * torch.tanh(local_scale)[:, :, None, :])
            + local_shift[:, :, None, :]
        )
        local = self.local_head(conditioned_local).view(
            root_context.shape[0], action_ids.shape[1], NUM_JOINTS,
            self.future_frames, 3,
        ).permute(0, 1, 3, 2, 4)
        pelvis = local[..., list(hip_joints), :].mean(dim=-2)
        local = local - pelvis[..., None, :]
        return root_future[..., None, :] + local, root_log_std

    def forward(
        self,
        history: torch.Tensor,
        robot_history: torch.Tensor,
        action_ids: torch.Tensor,
        confidence: torch.Tensor,
        visibility: torch.Tensor,
        support_features: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        person_indices: torch.Tensor | None = None,
        oracle_parameters: torch.Tensor | None = None,
        action_conditioning: bool = True,
    ) -> PersonalizedInteractionPrediction:
        encoded = self.encode_context(
            history, robot_history, confidence, visibility, support_features,
            support_mask, person_indices, oracle_parameters,
        )
        future, root_log_std = self.query_actions(
            encoded, action_ids, action_conditioning=action_conditioning
        )
        keep = torch.zeros(
            (history.shape[0], 1), dtype=action_ids.dtype, device=action_ids.device
        )
        natural, natural_root_log_std = self.query_actions(
            encoded, keep, action_conditioning=True
        )
        effect_root_log_std = 0.5 * torch.logaddexp(
            2.0 * root_log_std,
            2.0 * natural_root_log_std.expand_as(root_log_std),
        )
        return PersonalizedInteractionPrediction(
            future_by_action=future,
            natural_future=natural[:, 0],
            root_log_std_by_action=root_log_std,
            action_effect_root_log_std_by_action=effect_root_log_std,
            z_person=encoded[-1],
        )

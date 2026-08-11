"""Explicit action-routed natural + robot-response predictor for Phase 4B.6."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.data.functional_response_state import RESPONSE_STATE_DIM
from src.data.robot_action_schema import ACTION_DEFINITIONS, RobotAction
from src.data.skeleton_schema import compute_root
from src.models.personalized_response_world_model import PersonalizedRootPoseWorldModel
from src.models.functional_response_estimator import FunctionalResponseEstimator


@dataclass(frozen=True)
class FunctionalResponsePrediction:
    future_by_action: torch.Tensor
    natural_future: torch.Tensor
    action_effect_by_action: torch.Tensor
    theta_response: torch.Tensor
    theta_log_std: torch.Tensor | None


class FunctionalResponseDecoder(nn.Module):
    """Route each action only through relevant functional state dimensions."""

    def __init__(self, history_frames: int = 20, future_frames: int = 10) -> None:
        super().__init__()
        self.future_frames = future_frames
        self.natural_backbone = PersonalizedRootPoseWorldModel(
            "P0", history_frames=history_frames, future_frames=future_frames
        )
        speed = torch.zeros(7); distance = torch.zeros(7); lateral = torch.zeros(7)
        for action in RobotAction:
            definition = ACTION_DEFINITIONS[action]
            speed[int(action)] = definition.speed_scale_delta
            distance[int(action)] = definition.distance_offset_m
            lateral[int(action)] = definition.lateral_offset_m
        self.register_buffer("action_speed_delta", speed)
        self.register_buffer("action_distance_offset", distance)
        self.register_buffer("action_lateral_offset", lateral)

    def freeze_natural_backbone(self) -> None:
        for parameter in self.natural_backbone.parameters():
            parameter.requires_grad_(False)

    def decode_response(
        self,
        natural_future: torch.Tensor,
        human_history: torch.Tensor,
        robot_history: torch.Tensor,
        action_ids: torch.Tensor,
        theta_response: torch.Tensor,
    ) -> torch.Tensor:
        if theta_response.shape != (human_history.shape[0], RESPONSE_STATE_DIM):
            raise ValueError("theta_response must have shape [B,6]")
        speed_delta = self.action_speed_delta[action_ids]
        distance_offset = self.action_distance_offset[action_ids]
        history_root = compute_root(human_history)
        velocity = (history_root[:, -1, :2] - history_root[:, -2, :2]) * 10.0
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        fallback = torch.stack(
            (torch.cos(robot_history[:, -1, 2]), torch.sin(robot_history[:, -1, 2])),
            dim=-1,
        )
        forward = torch.where(
            (speed > 1e-6)[:, None], velocity / speed[:, None].clamp_min(1e-6), fallback
        )
        robot_to_human = history_root[:, -1, :2] - robot_history[:, -1, :2]
        distance = torch.linalg.vector_norm(robot_to_human, dim=-1)
        away = torch.where(
            (distance > 1e-6)[:, None],
            robot_to_human / distance[:, None].clamp_min(1e-6), forward,
        )
        lateral_direction = torch.stack((-away[:, 1], away[:, 0]), dim=-1)
        bearing_sign = torch.where(
            robot_history[:, -1, 6] >= 0,
            torch.ones_like(robot_history[:, -1, 6]),
            -torch.ones_like(robot_history[:, -1, 6]),
        )
        theta = theta_response[:, None, :]
        pressure = -distance_offset
        speed_scalar = theta[..., 0] * speed_delta * speed[:, None].clamp_min(0.35)
        distance_scalar = theta[..., 1] * pressure
        lateral_scalar = theta[..., 2] * pressure * bearing_sign[:, None] * 0.45
        response_velocity = (
            speed_scalar[..., None] * forward[:, None]
            + distance_scalar[..., None] * away[:, None]
            + lateral_scalar[..., None] * lateral_direction[:, None]
        )
        yaw_scalar = theta[..., 4] * (speed_delta + pressure) * 0.20
        delay_frames = torch.ceil(theta[..., 3] * 10.0 - 1e-6)
        frame = torch.arange(
            self.future_frames, device=human_history.device, dtype=human_history.dtype
        )[None, None]
        active_time = torch.relu((frame - delay_frames[..., None] + 1.0) * 0.1)
        adaptation = 1.0 - torch.exp(-theta[..., 5, None] * active_time)
        root_offset_xy = active_time[..., None] * adaptation[..., None] * response_velocity[..., None, :]
        root_offset = torch.zeros(
            (*root_offset_xy.shape[:-1], 3), device=human_history.device,
            dtype=human_history.dtype,
        )
        root_offset[..., :2] = root_offset_xy
        yaw = yaw_scalar[..., None] * adaptation
        natural_root = compute_root(natural_future)
        local = natural_future - natural_root[..., None, :]
        local = local[:, None].expand(-1, action_ids.shape[1], -1, -1, -1).clone()
        x, y = local[..., 0].clone(), local[..., 1].clone()
        cosine, sine = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
        local[..., 0] = cosine * x - sine * y
        local[..., 1] = sine * x + cosine * y
        active = (action_ids != int(RobotAction.KEEP)).to(human_history.dtype)[..., None, None, None]
        return (
            natural_root[:, None, :, None, :]
            + root_offset[..., None, :] * active
            + local
        )

    def forward(
        self,
        history: torch.Tensor,
        robot_history: torch.Tensor,
        action_ids: torch.Tensor,
        confidence: torch.Tensor,
        visibility: torch.Tensor,
        theta_response: torch.Tensor,
        theta_log_std: torch.Tensor | None = None,
    ) -> FunctionalResponsePrediction:
        natural = self.natural_backbone(
            history, robot_history, action_ids, confidence, visibility
        ).natural_future
        future = self.decode_response(
            natural, history, robot_history, action_ids, theta_response
        )
        return FunctionalResponsePrediction(
            future, natural, future - natural[:, None], theta_response,
            theta_log_std,
        )


class FunctionalResponseWorldModel(nn.Module):
    """F2: observable support -> functional state -> structured response."""

    def __init__(self, history_frames: int = 20, future_frames: int = 10) -> None:
        super().__init__()
        self.estimator = FunctionalResponseEstimator()
        self.decoder = FunctionalResponseDecoder(history_frames, future_frames)

    def forward(
        self,
        history: torch.Tensor,
        robot_history: torch.Tensor,
        action_ids: torch.Tensor,
        confidence: torch.Tensor,
        visibility: torch.Tensor,
        response_statistics: torch.Tensor,
        support_mask: torch.Tensor,
        response_state_mask: torch.Tensor,
    ) -> FunctionalResponsePrediction:
        estimate = self.estimator(
            response_statistics, support_mask, response_state_mask
        )
        return self.decoder(
            history, robot_history, action_ids, confidence, visibility,
            estimate.theta_mean, estimate.theta_log_std,
        )

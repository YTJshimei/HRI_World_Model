"""Batched counterfactual rollout with one shared human-context encoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.functional_response_state import RESPONSE_STATE_SCALE
from src.data.robot_action_schema import ACTION_DEFINITIONS, HOLD_ACTION_ID
from src.data.skeleton_schema import compute_root
from src.decision.decision_state import DecisionState
from src.models.functional_response_decoder import FunctionalResponseDecoder


@dataclass(frozen=True)
class CounterfactualRollout:
    action_ids: np.ndarray
    natural_future: np.ndarray
    predicted_root: np.ndarray
    predicted_local: np.ndarray
    predicted_global: np.ndarray
    predicted_robot_xy: np.ndarray
    predicted_human_robot_distance: np.ndarray
    predicted_action_effect: np.ndarray
    prediction_uncertainty: np.ndarray
    context_encoding_count: int


def _robot_future(
    human_history: np.ndarray, robot_history: np.ndarray,
    action_ids: np.ndarray, future_frames: int, sample_rate_hz: float = 10.0,
) -> np.ndarray:
    history_root = compute_root(human_history)
    robot_to_human = history_root[-1, :2] - robot_history[-1, :2]
    norm = float(np.linalg.norm(robot_to_human))
    away = robot_to_human / norm if norm > 1e-8 else np.asarray((1.0, 0.0))
    lateral = np.asarray((-away[1], away[0]))
    dt = 1.0 / sample_rate_hz
    results = []
    for action_id in action_ids:
        if int(action_id) == HOLD_ACTION_ID:
            from src.data.hold_candidate import hold_robot_rollout
            results.append(
                hold_robot_rollout(
                    robot_history, future_frames, sample_rate_hz
                ).xy
            )
            continue
        definition = ACTION_DEFINITIONS[int(action_id)]
        position = robot_history[-1, :2].astype(np.float64).copy()
        yaw = float(robot_history[-1, 2])
        speed = float(robot_history[-1, 3]) * (1.0 + definition.speed_scale_delta)
        angular = float(robot_history[-1, 4])
        trajectory = []
        for frame in range(future_frames):
            yaw += angular * dt
            position += speed * dt * np.asarray((np.cos(yaw), np.sin(yaw)))
            progress = (frame + 1) / future_frames
            trajectory.append(
                position - away * definition.distance_offset_m * progress
                + lateral * definition.lateral_offset_m * progress
            )
        results.append(trajectory)
    return np.asarray(results, dtype=np.float32)


class CounterfactualRolloutEngine:
    """Encode natural human context once, then query all actions/sigma points."""

    def __init__(self, decoder: FunctionalResponseDecoder, device: torch.device) -> None:
        self.decoder = decoder.to(device).eval()
        self.device = device

    @classmethod
    def from_phase4b6_checkpoint(
        cls, checkpoint_path: str | Path, device: str | torch.device = "cpu"
    ) -> "CounterfactualRolloutEngine":
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(path)
        decoder = FunctionalResponseDecoder()
        state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
        prefix = "decoder."
        decoder.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)})
        decoder.freeze_natural_backbone()
        return cls(decoder, torch.device(device))

    @torch.inference_mode()
    def rollout(self, state: DecisionState, uncertainty_aware: bool = True) -> CounterfactualRollout:
        history = torch.from_numpy(np.asarray(state.human_history, dtype=np.float32))[None].to(self.device)
        robot = torch.from_numpy(np.asarray(state.robot_history, dtype=np.float32))[None].to(self.device)
        confidence = torch.from_numpy(np.asarray(state.confidence, dtype=np.float32))[None].to(self.device)
        visibility = torch.from_numpy(np.asarray(state.visibility_mask, dtype=bool))[None].to(self.device)
        action_ids_np = np.asarray([int(item.action) for item in state.candidates], dtype=np.int64)
        action_ids = torch.from_numpy(action_ids_np)[None].to(self.device)

        # Exactly one expensive context encoding, independent of candidate count.
        natural = self.decoder.natural_backbone(
            history, robot, action_ids, confidence, visibility
        ).natural_future
        theta = torch.from_numpy(np.asarray(state.belief.theta_hat, dtype=np.float32))[None].to(self.device)
        future = self.decoder.decode_response(natural, history, robot, action_ids, theta)
        effect = future - natural[:, None]

        if uncertainty_aware:
            uncertainty = np.asarray(state.belief.theta_uncertainty, dtype=np.float32).copy()
            # Adaptation is poorly identified; marginalize conservatively without
            # trusting its point estimate as a strong action-specific signal.
            uncertainty[5] *= 0.35
            scale = np.asarray(RESPONSE_STATE_SCALE, dtype=np.float32)
            sigma = np.minimum(uncertainty, 0.75 * scale)
            theta_samples = [np.asarray(state.belief.theta_hat, dtype=np.float32)]
            for dimension in range(6):
                low = theta_samples[0].copy(); high = theta_samples[0].copy()
                low[dimension] = max(0.0, low[dimension] - sigma[dimension])
                high[dimension] += sigma[dimension]
                theta_samples.extend((low, high))
            theta_batch = torch.from_numpy(np.stack(theta_samples)).to(self.device)
            sample_count = len(theta_samples)
            sample_history = history.expand(sample_count, -1, -1, -1)
            sample_robot = robot.expand(sample_count, -1, -1)
            sample_natural = natural.expand(sample_count, -1, -1, -1)
            sample_actions = action_ids.expand(sample_count, -1)
            sampled = self.decoder.decode_response(
                sample_natural, sample_history, sample_robot,
                sample_actions, theta_batch,
            )
            sampled_effect = sampled - sample_natural[:, None]
            response_uncertainty = sampled_effect.std(dim=0, unbiased=False)
        else:
            response_uncertainty = torch.zeros_like(effect[0])

        global_future = future[0].cpu().numpy()
        root = compute_root(global_future)
        local = global_future - root[:, :, None]
        robot_future = _robot_future(
            state.human_history, state.robot_history, action_ids_np,
            global_future.shape[1],
        )
        distance = np.linalg.norm(root[..., :2] - robot_future, axis=-1)
        return CounterfactualRollout(
            action_ids=action_ids_np,
            natural_future=natural[0].cpu().numpy(),
            predicted_root=root.astype(np.float32),
            predicted_local=local.astype(np.float32),
            predicted_global=global_future.astype(np.float32),
            predicted_robot_xy=robot_future,
            predicted_human_robot_distance=distance.astype(np.float32),
            predicted_action_effect=effect[0].cpu().numpy().astype(np.float32),
            prediction_uncertainty=response_uncertainty.cpu().numpy().astype(np.float32),
            context_encoding_count=1,
        )

"""Training-only objectives for VLA outputs consumed by frozen controllers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


PARKOUR_LATENT_DIM = 32
PARKOUR_YAW_DIM = 2
PARKOUR_OUTPUT_DIM = PARKOUR_LATENT_DIM + PARKOUR_YAW_DIM
PARKOUR_OBSERVATION_DIM = 753
PARKOUR_ACTION_DIM = 12
PARKOUR_YAW_SCALE = 1.5
PARKOUR_MTS_THRESHOLD_RAD = 0.6

HAIC_LATENT_DIM = 256
HAIC_INPUT_DIM = 605
HAIC_ACTION_DIM = 23


class _ParkourHistoryEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        activation = nn.ELU()
        self.encoder = nn.Sequential(nn.Linear(53, 30), activation)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(30, 20, kernel_size=4, stride=2),
            activation,
            nn.Conv1d(20, 10, kernel_size=2, stride=1),
            activation,
            nn.Flatten(),
        )
        self.linear_output = nn.Sequential(nn.Linear(30, 20), activation)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        projection = self.encoder(obs.reshape(batch_size * 10, -1))
        output = self.conv_layers(projection.reshape(batch_size, 10, -1).permute(0, 2, 1))
        return self.linear_output(output)


class _ParkourActor(nn.Module):
    """The released Go2 teacher actor's fixed inference topology."""

    def __init__(self) -> None:
        super().__init__()
        activation = nn.ELU()
        self.priv_encoder = nn.Sequential(
            nn.Linear(29, 64), activation, nn.Linear(64, 20), activation
        )
        self.history_encoder = _ParkourHistoryEncoder()
        self.scan_encoder = nn.Sequential(
            nn.Linear(132, 128),
            activation,
            nn.Linear(128, 64),
            activation,
            nn.Linear(64, 32),
            nn.Tanh(),
        )
        self.actor_backbone = nn.Sequential(
            nn.Linear(53 + 32 + 9 + 20, 512),
            activation,
            nn.Linear(512, 256),
            activation,
            nn.Linear(256, 128),
            activation,
            nn.Linear(128, 12),
        )

    def infer_hist_latent(self, obs: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(obs[:, -10 * 53 :].view(-1, 10, 53))

    def forward(
        self,
        obs: torch.Tensor,
        *,
        scandots_latent: torch.Tensor,
    ) -> torch.Tensor:
        obs_prop_scan = torch.cat([obs[:, :53], scandots_latent], dim=1)
        obs_priv_explicit = obs[:, 53 + 132 : 53 + 132 + 9]
        latent = self.infer_hist_latent(obs)
        return self.actor_backbone(torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1))


class _HaicHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_mean = nn.Linear(HAIC_LATENT_DIM, HAIC_ACTION_DIM)
        self.actor_std = nn.Parameter(torch.ones(HAIC_ACTION_DIM))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(features)


class _HaicActor(nn.Module):
    """Pure state-dict-compatible copy of HAIC ``actor_adapt``."""

    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(HAIC_INPUT_DIM + HAIC_LATENT_DIM, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Mish(),
        )
        self.head = _HaicHead()

    def forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature(actor_input))


def _load_actor_checkpoint(path: str | Path, objective: str) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if objective == "parkour":
        actor = _ParkourActor()
        state_dict = {
            key.removeprefix("actor."): value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("actor.")
        }
    elif objective == "haic":
        actor = _HaicActor()
        state_dict = checkpoint
    else:
        raise ValueError(f"Unknown action objective: {objective}")
    actor.load_state_dict(state_dict)
    actor.requires_grad_(False)
    actor.eval()
    return actor


def _valid_substeps(termination: torch.Tensor) -> torch.Tensor:
    """Keep the terminal action and mask only substeps after termination."""
    done = termination.to(dtype=torch.int64)
    return (done.cumsum(dim=-1) - done) == 0


def _apply_parkour_mts(observation: torch.Tensor, predicted_yaw: torch.Tensor) -> torch.Tensor:
    predicted_yaw_rad = PARKOUR_YAW_SCALE * predicted_yaw
    current_yaw_error = torch.atan2(
        torch.sin(predicted_yaw_rad[..., 0] - observation[..., 6]),
        torch.cos(predicted_yaw_rad[..., 0] - observation[..., 6]),
    )
    use_prediction = current_yaw_error.abs() < PARKOUR_MTS_THRESHOLD_RAD
    result = observation.clone()
    result[..., 6:8] = torch.where(
        use_prediction[..., None], predicted_yaw_rad, observation[..., 6:8]
    )
    return result


class ActionObjective:
    """A non-module objective with a frozen, training-only downstream actor."""

    def __init__(
        self,
        objective: str,
        checkpoint_path: str | Path,
        output_min: torch.Tensor,
        output_max: torch.Tensor,
    ) -> None:
        self.objective = objective
        self.actor = _load_actor_checkpoint(checkpoint_path, objective)
        self.output_min = output_min.detach().cpu().float()
        self.output_max = output_max.detach().cpu().float()

    def _decode(self, prediction: torch.Tensor) -> torch.Tensor:
        output_min = self.output_min.to(device=prediction.device, dtype=prediction.dtype)
        output_max = self.output_max.to(device=prediction.device, dtype=prediction.dtype)
        prediction = prediction.clamp(-1.0, 1.0)
        return (prediction + 1.0) * 0.5 * (output_max - output_min) + output_min

    def __call__(self, action_outputs: dict[str, Any], action_inputs: dict[str, Any]):
        action_prediction = action_outputs["action_prediction"]
        prediction = self._decode(action_prediction[..., : self.output_min.numel()])
        self.actor.to(device=prediction.device, dtype=prediction.dtype)
        if self.objective == "parkour":
            return self._parkour(prediction, action_inputs)
        return self._haic(prediction, action_inputs)

    def _parkour(
        self, prediction: torch.Tensor, action_inputs: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        observation = action_inputs["actor_observation"]
        target = action_inputs["action_target"][..., :PARKOUR_OUTPUT_DIM]
        target_latent = target[..., :PARKOUR_LATENT_DIM]
        target_yaw = target[..., PARKOUR_LATENT_DIM:PARKOUR_OUTPUT_DIM]
        batch_size = prediction.shape[0]
        horizon = prediction.shape[1]
        with torch.no_grad():
            teacher_action = self.actor(
                observation.reshape(batch_size * horizon, PARKOUR_OBSERVATION_DIM),
                scandots_latent=target_latent.reshape(batch_size * horizon, PARKOUR_LATENT_DIM),
            ).view(batch_size, horizon, PARKOUR_ACTION_DIM)
        predicted_yaw = prediction[..., PARKOUR_LATENT_DIM:PARKOUR_OUTPUT_DIM]
        student_observation = _apply_parkour_mts(
            observation, predicted_yaw
        )
        predicted_latent = prediction[..., :PARKOUR_LATENT_DIM]
        student_action = self.actor(
            student_observation.reshape(batch_size * horizon, PARKOUR_OBSERVATION_DIM),
            scandots_latent=predicted_latent.reshape(batch_size * horizon, PARKOUR_LATENT_DIM),
        ).view(batch_size, horizon, PARKOUR_ACTION_DIM)
        termination = action_inputs["termination"]
        action_mask = _valid_substeps(termination).to(student_action.dtype)[..., None]
        action_loss = (
            torch.linalg.vector_norm(student_action - teacher_action, ord=2, dim=-1)
            * action_mask[..., 0]
        ).sum() / action_mask.sum()
        yaw_error = torch.linalg.vector_norm(
            PARKOUR_YAW_SCALE * (predicted_yaw - target_yaw), ord=2, dim=-1
        )
        yaw_loss = (yaw_error * action_mask[..., 0]).sum() / action_mask.sum()
        return {
            "task_loss": action_loss + yaw_loss,
            "actor_loss": action_loss,
            "yaw_loss": yaw_loss,
        }

    def _haic(
        self,
        prediction: torch.Tensor,
        action_inputs: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        actor_input = action_inputs["actor_input"]
        target_action = action_inputs["teacher_action"]
        batch_size = prediction.shape[0]
        horizon = prediction.shape[1]
        latent = prediction[..., :HAIC_LATENT_DIM]
        student_action = self.actor(
            torch.cat([actor_input, latent], dim=-1).reshape(
                batch_size * horizon, HAIC_INPUT_DIM + HAIC_LATENT_DIM
            )
        ).view(batch_size, horizon, HAIC_ACTION_DIM)
        action_mask = _valid_substeps(action_inputs["termination"]).to(student_action.dtype)[
            ..., None
        ]
        action_error = F.mse_loss(student_action, target_action, reduction="none")
        actor_loss = (action_error * action_mask).sum() / (action_mask.sum() * HAIC_ACTION_DIM)
        return {
            "task_loss": actor_loss,
            "actor_loss": actor_loss,
        }


class TaskActionObjective:
    """Apply the existing actor objective to a single predicted task command."""

    def __init__(self, config) -> None:
        self.name = config["name"]
        self.objective = ActionObjective(
            self.name,
            config["actor_checkpoint"],
            torch.tensor(config["output_min"]),
            torch.tensor(config["output_max"]),
        )

    def __call__(self, prediction: torch.Tensor, examples: list[dict]) -> torch.Tensor:
        def tensor(key: str) -> torch.Tensor:
            return torch.as_tensor(
                np.stack([example[key] for example in examples]),
                device=prediction.device,
                dtype=prediction.dtype,
            )

        if self.name == "parkour":
            # A 10 Hz H1 command is consumed with five fresh 50 Hz actor observations.
            prediction = prediction.expand(-1, 5, -1)
            inputs = {
                "actor_observation": tensor("actor_observation"),
                "action_target": tensor("action_target").expand(-1, 5, -1),
                "termination": tensor("termination"),
            }
        else:
            inputs = {
                "actor_input": tensor("actor_input").reshape(-1, 1, 605),
                "teacher_action": tensor("teacher_action").reshape(-1, 1, 23),
                "termination": tensor("termination").reshape(-1, 1),
            }
        return self.objective({"action_prediction": prediction}, inputs)["task_loss"]

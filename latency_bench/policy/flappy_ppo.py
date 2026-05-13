from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Action, Observation, PolicyOutput
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy.factory import register_policy
from modeling.flappy_ppo import (
    FlappyPpoAgent,
    FlappyPpoObservationConfig,
    FlappyPpoTorchObservationPipeline,
    build_agent_from_checkpoint,
    load_flappy_ppo_checkpoint,
    observation_config_from_checkpoint,
)


class FlappyPpoPolicyRunner(PolicyRunner):
    def __init__(
        self,
        agent: FlappyPpoAgent,
        observation_config: FlappyPpoObservationConfig,
        action_by_raw_id: Mapping[int, Action],
        device: str | torch.device = "cpu",
        deterministic: bool = True,
        checkpoint_global_step: int = 0,
    ):
        self.device = torch.device(device)
        self.agent = agent.to(self.device)
        self.agent.eval()
        self.observation_config = observation_config
        self.preprocessor = FlappyPpoTorchObservationPipeline(
            observation_config=observation_config,
            n_envs=1,
            device=self.device,
        )
        self.action_by_raw_id = dict(action_by_raw_id)
        self.deterministic = bool(deterministic)
        self.checkpoint_global_step = int(checkpoint_global_step)
        self._has_observation = False

    @property
    def current_stacked_observation(self):
        obs = self.preprocessor.current_observation
        if obs is None:
            return None
        if obs.ndim == 4 and obs.shape[0] == 1:
            return obs[0].copy()
        return obs

    def predict(self, observation: Observation) -> PolicyOutput:
        frame_skip = _extract_frame_skip(observation)
        prebuilt = _prebuilt_observation_tensor(
            observation.data,
            observation_config=self.observation_config,
            frame_skip=frame_skip,
            device=self.device,
        )
        if prebuilt is not None:
            obs_tensor, obs_nhwc = prebuilt
            self.preprocessor._last_observation = obs_nhwc
            self._has_observation = True
        else:
            obs_tensor = self._preprocess_raw_frame(observation, frame_skip)

        with torch.no_grad():
            if self.deterministic:
                logits = self.agent.get_logits(obs_tensor)
                raw_action = int(torch.argmax(logits, dim=1).item())
            else:
                action, _, _, _ = self.agent.get_action_and_value(obs_tensor)
                raw_action = int(action.item())

        action = self.action_by_raw_id.get(raw_action)
        if action is None:
            raise ValueError(f"Flappy PPO emitted unmapped action id: {raw_action}")
        return PolicyOutput(
            action=action,
            raw_output=raw_action,
            metadata={
                "policy_type": "flappy_ppo",
                "checkpoint_global_step": self.checkpoint_global_step,
            },
        )

    def _preprocess_raw_frame(self, observation: Observation, frame_skip: int | float) -> torch.Tensor:
        raw_rgb = torch.as_tensor(
            observation.data,
            dtype=torch.uint8,
            device=self.device,
        ).unsqueeze(0)
        if int(observation.env_step) == 0 or not self._has_observation:
            obs_tensor = self.preprocessor.reset(raw_rgb, frame_skips=[frame_skip])
        else:
            obs_tensor = self.preprocessor.step(
                raw_rgb,
                dones=[False],
                frame_skips=[frame_skip],
            )
        self._has_observation = True
        return obs_tensor


@register_policy("flappy_ppo")
def _build_flappy_ppo_policy(config: dict[str, Any], action_resolver: ActionResolver) -> PolicyRunner:
    policy_cfg = config["policy"]
    checkpoint_path = Path(policy_cfg["checkpoint_path"])
    device = _resolve_device(policy_cfg.get("device", "auto"))
    checkpoint = load_flappy_ppo_checkpoint(checkpoint_path, map_location=device)
    observation_config = observation_config_from_checkpoint(checkpoint)
    agent = build_agent_from_checkpoint(checkpoint, device=device)
    action_refs = policy_cfg.get("actions", action_resolver.default_action_refs())
    actions = action_resolver.resolve_many(action_refs)
    action_by_raw_id = _build_action_map(actions, action_resolver)
    return FlappyPpoPolicyRunner(
        agent=agent,
        observation_config=observation_config,
        action_by_raw_id=action_by_raw_id,
        device=device,
        deterministic=bool(policy_cfg.get("deterministic", True)),
        checkpoint_global_step=int(checkpoint.get("global_step", 0)),
    )


def _resolve_device(device: Any) -> str:
    device_name = str(device)
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def _build_action_map(actions: list[Action], action_resolver: ActionResolver) -> dict[int, Action]:
    action_by_raw_id: dict[int, Action] = {}
    for action in actions:
        try:
            raw_id = int(action.value)
        except (TypeError, ValueError):
            continue
        action_by_raw_id[raw_id] = action

    for raw_id in (0, 1):
        if raw_id not in action_by_raw_id:
            action_by_raw_id[raw_id] = action_resolver.resolve(raw_id)

    return action_by_raw_id


def _extract_frame_skip(observation: Observation) -> int | float:
    metadata = getattr(observation, "metadata", None)
    if isinstance(metadata, Mapping):
        return metadata.get("frame_skip", metadata.get("latency_frame_skip", 0))
    return 0


def _prebuilt_observation_tensor(
    data: Any,
    *,
    observation_config: FlappyPpoObservationConfig,
    frame_skip: int | float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    del frame_skip
    arr = np.asarray(data)
    if arr.ndim != 3:
        return None
    channel_options = {observation_config.stacked_channels}
    if arr.shape[-1] in channel_options:
        obs_nhwc_np = arr
    elif arr.shape[0] in channel_options:
        obs_nhwc_np = np.transpose(arr, (1, 2, 0))
    else:
        return None
    if obs_nhwc_np.shape[:2] != (observation_config.image_size, observation_config.image_size):
        return None
    if obs_nhwc_np.shape[-1] != observation_config.in_channels:
        return None
    obs_nhwc = torch.as_tensor(
        np.ascontiguousarray(obs_nhwc_np),
        dtype=torch.uint8,
        device=device,
    ).unsqueeze(0)
    obs_tensor = obs_nhwc.permute(0, 3, 1, 2).contiguous()
    return obs_tensor, obs_nhwc


__all__ = ["FlappyPpoPolicyRunner"]

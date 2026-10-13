"""Adapt the native batched MIKASA simulator to benchmark env-step slots."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from latency_bench.core.types import Action, Observation
from latency_bench.envs.tasks.mikasa import (
    MIKASA_IMAGE_VIEWS_INFO_KEY,
    MIKASA_STATE_INFO_KEY,
    make_vla_env,
)
from latency_bench.executors.gpu_batched_env_step_backend import (
    GpuBatchedEnvStepBackendBase,
    SlotStepOutcome,
)


class MikasaEnvStepBackend(GpuBatchedEnvStepBackendBase):
    """Own the native MIKASA simulator and its 7D action contract."""

    backend_name = "mikasa_gpu_batched"

    def __init__(self, *, config: dict[str, Any], num_slots: int, env=None):
        noop_action = Action(
            value=np.asarray(config["env"]["noop_action"], dtype=np.float32),
            name="noop",
            is_noop=True,
        )
        super().__init__(
            config=config,
            noop_action=noop_action,
            num_slots=num_slots,
            action_space=gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
        )
        self.env = (
            make_vla_env(
                num_envs=num_slots,
                simulator_device=config["env"]["simulator_device"],
            )
            if env is None
            else env
        )
        self._episode_seeds = [0] * num_slots
        self._success = np.zeros(num_slots, dtype=np.bool_)
        self._observation, _ = self.env.reset(seed=self._episode_seeds)

    def _reset_slot_observation(self, slot_id: int, *, seed: int | None) -> Observation:
        # The inference interpreter does not need the native task suite at spawn.
        from mikasa_robo_suite.seed_reset import reset_seeded_slot

        if seed is not None:
            self._episode_seeds[slot_id] = int(seed)
        self._observation, _ = reset_seeded_slot(
            self.env,
            slot_id=slot_id,
            seed=self._episode_seeds[slot_id],
        )
        self._env_steps[slot_id] = 0
        self._success[slot_id] = False
        return self._observation_for_slot(slot_id)

    def _observe_slot_observations(
        self,
        slot_ids: list[int],
    ) -> dict[int, Observation]:
        return {slot_id: self._observation_for_slot(slot_id) for slot_id in slot_ids}

    def _capture_rgb_frames(self, slot_ids: list[int]) -> dict[int, np.ndarray]:
        return {
            slot_id: np.concatenate(
                self._observation_for_slot(slot_id).metadata[MIKASA_IMAGE_VIEWS_INFO_KEY], axis=1,
            )
            for slot_id in slot_ids
        }

    def _step_cores(
        self,
        slot_ids: list[int],
        *,
        actions: np.ndarray,
        active_mask: np.ndarray,
    ) -> Any:
        del slot_ids, active_mask
        tensor_actions = torch.as_tensor(
            actions,
            dtype=torch.float32,
            device=self.env.unwrapped.device,
        )
        self._observation, reward, terminated, truncated, info = self.env.step(
            tensor_actions
        )
        return reward, terminated, truncated, info

    def _slot_step_outcome(self, state: Any, slot_id: int) -> SlotStepOutcome:
        reward, terminated, truncated, info = state
        success = bool(_slot_value(info["success"], slot_id))
        self._success[slot_id] |= success
        return SlotStepOutcome(
            reward=float(_slot_value(reward, slot_id)),
            done=bool(_slot_value(terminated, slot_id)),
            truncated=bool(_slot_value(truncated, slot_id)),
            info={
                "success": success,
                "task_metrics": {"success": float(self._success[slot_id])},
            },
        )

    def _observation_for_slot(self, slot_id: int) -> Observation:
        rgb = self._observation["rgb"]
        if torch.is_tensor(rgb):
            rgb = rgb.detach().cpu().numpy()
        rgb = np.asarray(rgb)
        views = np.stack(
            [
                np.asarray(rgb[slot_id, :, :, :3], dtype=np.uint8),
                np.asarray(rgb[slot_id, :, :, 3:6], dtype=np.uint8),
            ]
        )
        metadata = {
            MIKASA_IMAGE_VIEWS_INFO_KEY: views,
            MIKASA_STATE_INFO_KEY: self._observation["proprio"][slot_id].detach().cpu().numpy(),
            "slot_id": slot_id,
        }
        if "action_prefix_state_key" in self.config["env"]:
            metadata["action_prefix_state_key"] = self.config["env"]["action_prefix_state_key"]
        if "returned_action_context" in self.config["env"]:
            context = self.config["env"]["returned_action_context"]
            metadata["returned_action_context"] = {
                **context,
                "order": np.asarray(context["order"]),
                "low": np.asarray(context["low"], dtype=np.float32),
                "high": np.asarray(context["high"], dtype=np.float32),
            }
        return Observation(
            data=None,
            env_step=int(self._env_steps[slot_id]),
            sim_time_ms=float(self._env_steps[slot_id]) * self._frame_ms,
            metadata=metadata,
        )

    def close(self) -> None:
        if not self.closed:
            self.env.close()
        super().close()


def _slot_value(value: Any, slot_id: int) -> Any:
    if torch.is_tensor(value):
        return value.detach().reshape(-1)[slot_id].cpu().item()
    return np.asarray(value).reshape(-1)[slot_id].item()

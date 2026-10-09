"""Shared skeleton for in-process GPU-batched env step backends.

latency_bench owns observation cadence, latency sampling, and action
persistence through one ActionScheduler per slot; these backends only
advance active raw frames in a batched core. Subclasses own the core
stepping, rendering, and observation contract; this base owns the slot
bookkeeping and the reset/observe/step response plumbing that is
identical across every backend.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.executors.env_step_backend import EnvStepResponse, RemoteEnvSlotHandle


def config_policy_type(config: dict[str, Any]) -> str:
    return str(config.get("policy", {}).get("type", "")).strip().lower()


def uses_starvla_policy(config: dict[str, Any]) -> bool:
    return config_policy_type(config) == "starvla"


@dataclass
class SlotStepOutcome:
    """One slot's result of a batched core step, before the shared step_info keys."""

    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any]
    # Backend-specific keys appended after the shared step_info keys.
    extra_info: dict[str, Any] = field(default_factory=dict)


class GpuBatchedEnvStepBackendBase(ABC):
    backend_name: str

    def __init__(self, *, config: dict[str, Any], noop_action: Action, num_slots: int, action_space: gym.Space):
        self.config = config
        self.noop_action = noop_action
        self.action_space = action_space
        self._num_slots = int(num_slots)
        self._env_fps = float(config["env"]["env_fps"])
        self._frame_ms = 1000.0 / self._env_fps
        self._env_steps = np.zeros(self._num_slots, dtype=np.int64)
        self._last_infos: list[dict[str, Any]] = [{} for _slot in range(self._num_slots)]
        self.slot_handles = [
            RemoteEnvSlotHandle(
                slot_id=slot_id,
                env_fps=self._env_fps,
                noop_action=noop_action,
                action_space=action_space,
            )
            for slot_id in range(self._num_slots)
        ]
        self.last_step_metadata_by_slot: dict[int, dict[str, Any]] = {}
        self.collection_frames: dict[int, np.ndarray] = {}
        self.closed = False

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def env_fps(self) -> float:
        return self._env_fps

    def worker_pids(self) -> list[int]:
        return []

    def reset_slot(self, slot_id: int, *, episode_id: int, seed: int | None) -> Observation:
        self._raise_if_closed()
        slot_id = int(slot_id)
        start = time.perf_counter()
        observation = self._reset_slot_observation(slot_id, seed=seed)
        metadata = {
            "backend": self.backend_name,
            "slot_id": slot_id,
            "episode_id": int(episode_id),
            "worker_pid": os.getpid(),
            "worker_reset_wall_sec": time.perf_counter() - start,
        }
        self.slot_handles[slot_id].update_observation(observation, metadata)
        return observation

    def observe_slots(self, slot_ids: Sequence[int], *, capture_rgb: bool = False) -> dict[int, Observation]:
        self._raise_if_closed()
        observations = self._observe_slot_observations([int(slot_id) for slot_id in slot_ids])
        if capture_rgb:
            missing_rgb = [slot for slot, obs in observations.items()
                           if ENV_RAW_RGB_FRAME_STACK_INFO_KEY not in obs.metadata]
            frames = self._capture_rgb_frames(missing_rgb) if missing_rgb else {}
            self.collection_frames = {
                slot: np.array(obs.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY][-1]
                               if ENV_RAW_RGB_FRAME_STACK_INFO_KEY in obs.metadata else frames[slot], copy=True)
                for slot, obs in observations.items()
            }
        for slot_id, observation in observations.items():
            self.slot_handles[slot_id].update_observation(observation, None)
        return observations

    def step_slots(self, actions_by_slot: Mapping[int, Action]) -> dict[int, EnvStepResponse]:
        self._raise_if_closed()
        start = time.perf_counter()
        slot_ids = [int(slot_id) for slot_id in actions_by_slot]
        actions = np.zeros((self._num_slots, *self.action_space.shape), dtype=self.action_space.dtype)
        active_mask = np.zeros(self._num_slots, dtype=np.bool_)
        action_names: dict[int, str | None] = {}
        for slot_id, action in actions_by_slot.items():
            slot_id = int(slot_id)
            actions[slot_id] = action.value
            active_mask[slot_id] = True
            action_names[slot_id] = action.name

        state = self._step_cores(slot_ids, actions=actions, active_mask=active_mask)

        responses: dict[int, EnvStepResponse] = {}
        for slot_id in slot_ids:
            self._env_steps[slot_id] += 1
            outcome = self._slot_step_outcome(state, slot_id)
            step_info = outcome.info
            step_info.update(
                {
                    "env_step": int(self._env_steps[slot_id]),
                    "sim_time_ms": int(self._env_steps[slot_id]) * self._frame_ms,
                    "applied_action": actions[slot_id].copy(),
                    "applied_action_name": action_names[slot_id],
                }
            )
            step_info.update(outcome.extra_info)
            result = StepResult(
                observation=None,
                reward=outcome.reward,
                done=outcome.done,
                truncated=outcome.truncated,
                info=step_info,
            )
            metadata = {
                "backend": self.backend_name,
                "slot_id": slot_id,
                "worker_pid": os.getpid(),
                "worker_step_wall_sec": time.perf_counter() - start,
            }
            self.last_step_metadata_by_slot[slot_id] = metadata
            responses[slot_id] = EnvStepResponse(
                result=result,
                worker_pid=os.getpid(),
                metadata=metadata,
            )
        return responses

    def close(self) -> None:
        self.collection_frames.clear()
        self.closed = True

    @abstractmethod
    def _reset_slot_observation(self, slot_id: int, *, seed: int | None) -> Observation:
        """Reset the slot's core state and return its first observation."""

    @abstractmethod
    def _observe_slot_observations(self, slot_ids: list[int]) -> dict[int, Observation]:
        """Render/collect the current observation for each slot."""

    @abstractmethod
    def _step_cores(self, slot_ids: list[int], *, actions: np.ndarray, active_mask: np.ndarray) -> Any:
        """Advance the batched core one raw frame; returns state for _slot_step_outcome."""

    @abstractmethod
    def _slot_step_outcome(self, state: Any, slot_id: int) -> SlotStepOutcome:
        """Extract one slot's step outcome; runs after the slot's env_step counter advanced."""

    def _capture_rgb_frames(self, slot_ids: list[int]) -> dict[int, np.ndarray]:
        """Capture current student RGB without changing policy frame history."""
        raise NotImplementedError(f"{self.backend_name} does not provide collection RGB")

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise RuntimeError("env step backend is closed")

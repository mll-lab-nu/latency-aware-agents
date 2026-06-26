from __future__ import annotations

import os
import time
from typing import Any
from collections.abc import Mapping, Sequence

import gymnasium as gym
import numpy as np
import torch

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.executors.env_step_backend import EnvStepResponse, RemoteEnvSlotHandle
from training.common.raw_rgb_export import ENV_RAW_RGB_FRAME_STACK_INFO_KEY, RawRgbFrameStackBuffer
from training.demon_attack_sf.gpu_core import CudaDemonAttackCore


class DemonAttackGpuBatchedEnvStepBackend:
    """Batch Demon Attack CUDA stepping behind latency_bench's simulated scheduler.

    latency_bench owns observation cadence, latency sampling, and action
    persistence. This backend only advances active raw frames and exposes the
    same policy observation contract as DemonAttackEnvAdapter.
    """

    def __init__(self, *, config: dict[str, Any], noop_action: Action, num_slots: int):
        self.config = config
        self.noop_action = noop_action
        self._num_slots = int(num_slots)
        env_cfg = config["env"]
        if env_cfg["reset_mode"] != "fire":
            raise ValueError("demon_attack_gpu_batched currently supports reset_mode='fire'")
        self._env_fps = float(env_cfg["env_fps"])
        self._frame_ms = 1000.0 / self._env_fps
        self.frame_stack = env_cfg["frame_stack"]
        self.clip_reward = env_cfg["clip_reward"]
        self.export_env_raw_rgb_frames = bool(env_cfg.get("export_env_raw_rgb_frames", False)) or _uses_starvla_policy(config)
        self.device = _resolve_device(env_cfg["gpu_env_device"])
        self.core = CudaDemonAttackCore(
            batch_size=self._num_slots,
            device=self.device,
            noop_max=env_cfg["noop_max"],
            terminal_on_life_loss=env_cfg["terminal_on_life_loss"],
            obs_stride=1,
            frame_stack=1,
        )
        self._stack = torch.empty((self._num_slots, self.frame_stack, 84, 84), dtype=torch.uint8, device=self.device)
        self._next_stack = torch.empty_like(self._stack)
        self._env_steps = np.zeros(self._num_slots, dtype=np.int64)
        self._last_infos: list[dict[str, Any]] = [{} for _slot in range(self._num_slots)]
        self._raw_rgb_frames = [
            RawRgbFrameStackBuffer(num_frames=self.frame_stack)
            for _slot in range(self._num_slots)
        ]
        self.slot_handles = [
            RemoteEnvSlotHandle(
                slot_id=slot_id,
                env_fps=self._env_fps,
                noop_action=noop_action,
                action_space=gym.spaces.Discrete(6),
            )
            for slot_id in range(self._num_slots)
        ]
        self.last_reset_metadata_by_slot: dict[int, dict[str, Any]] = {}
        self.last_step_metadata_by_slot: dict[int, dict[str, Any]] = {}
        self._core_reset_initialized = False
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
        if self._core_reset_initialized:
            obs, infos = self.core.reset_batch(seeds=[seed], indices=[slot_id])
        else:
            full_reset_seeds = None if seed is None else [seed for _slot in range(self._num_slots)]
            obs, infos = self.core.reset_batch(seeds=full_reset_seeds)
            self._core_reset_initialized = True
        self._env_steps[slot_id] = 0
        self._stack[slot_id] = obs[slot_id].repeat(self.frame_stack, 1, 1)
        self._last_infos[slot_id] = self._info(slot_id, infos[slot_id])
        if self.export_env_raw_rgb_frames:
            self._reset_raw_rgb_frame(slot_id)
        observation = self._observe_slot(slot_id)
        metadata = {
            "backend": "demon_attack_gpu_batched",
            "slot_id": slot_id,
            "episode_id": int(episode_id),
            "worker_pid": os.getpid(),
            "worker_reset_wall_sec": time.perf_counter() - start,
        }
        self.last_reset_metadata_by_slot[slot_id] = metadata
        self.slot_handles[slot_id].update_observation(observation, metadata)
        return observation

    def observe_slots(self, slot_ids: Sequence[int]) -> dict[int, Observation]:
        self._raise_if_closed()
        observations = {int(slot_id): self._observe_slot(int(slot_id)) for slot_id in slot_ids}
        for slot_id, observation in observations.items():
            self.slot_handles[slot_id].update_observation(observation, None)
        return observations

    def step_slots(self, actions_by_slot: Mapping[int, Action]) -> dict[int, EnvStepResponse]:
        self._raise_if_closed()
        start = time.perf_counter()
        slot_ids = [int(slot_id) for slot_id in actions_by_slot]
        active_mask = np.zeros(self._num_slots, dtype=np.bool_)
        actions = np.zeros(self._num_slots, dtype=np.int64)
        action_names: dict[int, str | None] = {}
        for slot_id, action in actions_by_slot.items():
            slot_id = int(slot_id)
            active_mask[slot_id] = True
            actions[slot_id] = int(action.value)
            action_names[slot_id] = action.name

        step_result = self.core.step_batch(actions, active_mask=active_mask)
        rewards = step_result.rewards.detach().cpu().numpy().astype(np.float32, copy=False)
        terminated = step_result.terminated.detach().cpu().numpy().astype(np.bool_, copy=False)
        truncated = step_result.truncated.detach().cpu().numpy().astype(np.bool_, copy=False)
        scores = step_result.scores.detach().cpu().numpy().astype(np.int64, copy=False)
        lives = step_result.lives.detach().cpu().numpy().astype(np.int64, copy=False)
        if self.clip_reward:
            rewards_out = np.sign(rewards).astype(np.float32, copy=False)
        else:
            rewards_out = rewards

        self._append_observation_stack(step_result.observations, active_mask)
        if self.export_env_raw_rgb_frames:
            self._append_raw_rgb_frames(active_mask=active_mask, done_mask=np.logical_or(terminated, truncated))

        responses: dict[int, EnvStepResponse] = {}
        for slot_id in slot_ids:
            self._env_steps[slot_id] += 1
            done = bool(terminated[slot_id])
            is_truncated = bool(truncated[slot_id])
            info = self._info(
                slot_id,
                {
                    "score": int(scores[slot_id]),
                    "ale.lives": int(lives[slot_id]),
                    "lives": int(lives[slot_id]),
                },
            )
            self._last_infos[slot_id] = info
            step_info = dict(info)
            step_info.update(
                {
                    "env_step": int(self._env_steps[slot_id]),
                    "sim_time_ms": int(self._env_steps[slot_id]) * self._frame_ms,
                    "applied_action": int(actions[slot_id]),
                    "applied_action_name": action_names[slot_id],
                    "is_true_episode_end": bool(done or is_truncated),
                }
            )
            result = StepResult(
                observation=None,
                reward=float(rewards_out[slot_id]),
                done=done,
                truncated=is_truncated,
                info=step_info,
            )
            metadata = {
                "backend": "demon_attack_gpu_batched",
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
        self.closed = True

    def _observe_slot(self, slot_id: int) -> Observation:
        return Observation(
            data=self._policy_frame_stack(slot_id),
            env_step=int(self._env_steps[slot_id]),
            sim_time_ms=int(self._env_steps[slot_id]) * self._frame_ms,
            metadata=self._metadata(slot_id),
        )

    def _policy_frame_stack(self, slot_id: int) -> np.ndarray:
        return self._stack[int(slot_id)].detach().cpu().numpy().copy()

    def _metadata(self, slot_id: int) -> dict[str, Any]:
        metadata = dict(self._last_infos[int(slot_id)])
        metadata.update(
            {
                "env_step": int(self._env_steps[int(slot_id)]),
                "sim_time_ms": int(self._env_steps[int(slot_id)]) * self._frame_ms,
                "observation": "atari_frame_v1",
            }
        )
        if self.export_env_raw_rgb_frames:
            metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = self._raw_rgb_frames[int(slot_id)].stack()
        return metadata

    def _info(self, slot_id: int, info: dict[str, Any]) -> dict[str, Any]:
        out = dict(info)
        out["env_step"] = int(self._env_steps[int(slot_id)])
        out["sim_time_ms"] = int(self._env_steps[int(slot_id)]) * self._frame_ms
        out["observation"] = "atari_frame_v1"
        return out

    def _append_observation_stack(self, obs: torch.Tensor, active_mask: np.ndarray) -> None:
        active_indices = np.flatnonzero(active_mask)
        index_t = torch.as_tensor(active_indices, dtype=torch.int64, device=self.device)
        self._next_stack[index_t] = self._stack[index_t]
        self._next_stack[index_t, :-1, :, :] = self._stack[index_t, 1:, :, :].clone()
        self._next_stack[index_t, -1:, :, :] = obs[index_t]
        self._stack[index_t] = self._next_stack[index_t]

    def _reset_raw_rgb_frame(self, slot_id: int) -> None:
        frames = self.core.native_rgb_frames().detach().cpu().numpy()
        self._raw_rgb_frames[int(slot_id)].reset(frames[int(slot_id)])

    def _append_raw_rgb_frames(self, *, active_mask: np.ndarray, done_mask: np.ndarray) -> None:
        frames = self.core.native_rgb_frames().detach().cpu().numpy()
        for slot_id in np.flatnonzero(active_mask):
            slot_id = int(slot_id)
            if not bool(done_mask[slot_id]):
                self._raw_rgb_frames[slot_id].append(frames[slot_id])

    def _raise_if_closed(self) -> None:
        if self.closed:
            raise RuntimeError("env step backend is closed")


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda")
    return torch.device(device)


def _uses_starvla_policy(config: dict[str, Any]) -> bool:
    policy_type = str(config.get("policy", {}).get("type", "")).strip().lower()
    return policy_type == "starvla"

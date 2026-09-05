from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from latency_bench.core.types import Action, Observation
from latency_bench.envs.atari import TRUE_EPISODE_END_INFO_KEY
from latency_bench.executors.gpu_batched_env_step_backend import (
    GpuBatchedEnvStepBackendBase,
    SlotStepOutcome,
    uses_starvla_policy,
)
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY, RawRgbFrameStackBuffer
from training.demon_attack_sf.gpu_core import CudaDemonAttackCore


class DemonAttackGpuBatchedEnvStepBackend(GpuBatchedEnvStepBackendBase):
    """Batch Demon Attack CUDA stepping behind latency_bench's simulated scheduler.

    latency_bench owns observation cadence, latency sampling, and action
    persistence. This backend only advances active raw frames and exposes the
    same policy observation contract as DemonAttackEnvAdapter.
    """
    backend_name = "demon_attack_gpu_batched"

    def __init__(self, *, config: dict[str, Any], noop_action: Action, num_slots: int):
        env_cfg = config["env"]
        if env_cfg["reset_mode"] != "fire":
            raise ValueError("demon_attack_gpu_batched currently supports reset_mode='fire'")
        super().__init__(config=config, noop_action=noop_action, num_slots=num_slots, action_space=gym.spaces.Discrete(6))
        self.frame_stack = env_cfg["frame_stack"]
        self.clip_reward = env_cfg["clip_reward"]
        self.export_env_raw_rgb_frames = bool(env_cfg.get("export_env_raw_rgb_frames", False)) or uses_starvla_policy(config)
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
        self._raw_rgb_frames = [
            RawRgbFrameStackBuffer(num_frames=self.frame_stack)
            for _slot in range(self._num_slots)
        ]
        self._core_reset_initialized = False

    def _reset_slot_observation(self, slot_id: int, *, seed: int | None) -> Observation:
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
        return self._observe_slot(slot_id)

    def _observe_slot_observations(self, slot_ids: list[int]) -> dict[int, Observation]:
        return {slot_id: self._observe_slot(slot_id) for slot_id in slot_ids}

    def _step_cores(self, slot_ids: list[int], *, actions: np.ndarray, active_mask: np.ndarray) -> Any:
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
        return rewards_out, terminated, truncated, scores, lives

    def _slot_step_outcome(self, state: Any, slot_id: int) -> SlotStepOutcome:
        rewards_out, terminated, truncated, scores, lives = state
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
        return SlotStepOutcome(
            reward=float(rewards_out[slot_id]),
            done=done,
            truncated=is_truncated,
            info=dict(info),
            extra_info={TRUE_EPISODE_END_INFO_KEY: bool(done or is_truncated)},
        )

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


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda")
    return torch.device(device)

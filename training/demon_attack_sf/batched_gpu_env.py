from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict as GymDict, Discrete

from latency_bench.core.clock import EnvClock
from training.common.latency_config import (
    sample_factory_latency_params_from_flat_config,
    sample_factory_latency_seed_from_flat_config,
)
from training.common.batched_metrics import BatchedEpisodeMetrics
from training.common.gpu_latency import CudaLatencyScheduler
from training.demon_attack_sf.gpu_core import CudaDemonAttackCore


class DemonAttackBatchedGpuEnv(gym.Env):
    is_multiagent = True

    def __init__(
        self,
        *,
        cfg: Any,
        env_config: Any | None,
        gpu_env_device: str | torch.device,
        gpu_env_batch_size: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.env_config = env_config
        self.num_agents = gpu_env_batch_size
        self.device = self._resolve_device(gpu_env_device)
        self.clock = EnvClock(env_fps=cfg.env_fps, obs_fps=cfg.obs_fps)
        self.frame_stack = cfg.frame_stack
        self.action_space = Discrete(6)
        self.observation_space = GymDict(
            {
                "obs": Box(
                    low=0,
                    high=255,
                    shape=(self.frame_stack, 84, 84),
                    dtype=np.uint8,
                )
            }
        )
        self.core = CudaDemonAttackCore(
            batch_size=self.num_agents,
            device=self.device,
            noop_max=cfg.noop_max,
            terminal_on_life_loss=cfg.terminal_on_life_loss,
        )
        sub_seeds = [self._sub_seed(agent_idx) for agent_idx in range(self.num_agents)]
        latency_seeds = [
            sample_factory_latency_seed_from_flat_config(
                self.cfg,
                fallback_seed=sub_seeds[agent_idx],
                worker_index=self._worker_index,
                vector_index=self._vector_index,
                env_id=self._sub_env_id(agent_idx),
            )
            for agent_idx in range(self.num_agents)
        ]
        latency_params = sample_factory_latency_params_from_flat_config(self.cfg, env_fps=self.clock.env_fps)
        self.latency = CudaLatencyScheduler(
            batch_size=self.num_agents,
            fixed_latency_raw_frames=latency_params.fixed_latency_raw_frames,
            env_fps=self.clock.env_fps,
            obs_fps=self.clock.obs_fps,
            hold_policy=self.cfg.hold_policy,
            ordering_policy=self.cfg.ordering_policy,
            latency_distribution_spec=latency_params.latency_distribution_spec,
            seeds=latency_seeds,
            env_reset_seed_bases=[seed + 104729 for seed in sub_seeds],
            add_latency_info=self.cfg.add_latency_info,
            max_pending_actions=self.cfg.max_pending_actions,
            device=self.device,
        )
        self.metrics = BatchedEpisodeMetrics(
            batch_size=self.num_agents,
            frames_per_decision=self.clock.obs_stride_raw_frames,
            metrics_path=self.cfg.episode_metrics_path,
            worker_index=self._worker_index,
            vector_index=self._vector_index,
            env_ids=[self._sub_env_id(agent_idx) for agent_idx in range(self.num_agents)],
            include_score=False,
        )
        self._stack = torch.empty((self.num_agents, self.frame_stack, 84, 84), dtype=torch.uint8, device=self.device)
        self._next_stack = torch.empty_like(self._stack)
        self._raw_episode_steps = np.zeros(self.num_agents, dtype=np.int64)
        self._raw_episode_steps_gpu = torch.zeros(self.num_agents, dtype=torch.int64, device=self.device)
        self._frame_numbers = np.zeros(self.num_agents, dtype=np.int64)
        self._max_episode_steps = self.cfg.max_episode_steps
        self._last_frame_skips = np.zeros(self.num_agents, dtype=np.int64)
        self._active_mask_gpu = torch.ones(self.num_agents, dtype=torch.bool, device=self.device)
        self.sample_factory_step_cpu_tensors: dict[str, torch.Tensor] = {}
        self._has_reset = False

    @property
    def _worker_index(self) -> int:
        return self.env_config.worker_index

    @property
    def _vector_index(self) -> int:
        return self.env_config.vector_index

    @property
    def _env_id(self) -> int:
        return self.env_config.env_id

    def _sub_seed(self, agent_idx: int) -> int:
        return self.cfg.seed + 100000 * self._worker_index + 1000 * self._vector_index + int(agent_idx)

    def _sub_env_id(self, agent_idx: int) -> int:
        return self._env_id * self.num_agents + int(agent_idx)

    def reset(self, *, seed=None, options=None):
        if seed is None:
            reset_seeds, latency_infos = self.latency.reset()
        else:
            reset_seeds, latency_infos = self.latency.reset(
                seeds=[int(seed) + int(agent_idx) for agent_idx in range(self.num_agents)]
            )
        self.metrics.reset()
        self._raw_episode_steps[:] = 0
        self._raw_episode_steps_gpu.zero_()
        obs, _ = self.core.reset_batch(seeds=reset_seeds)
        self._reset_frame_numbers()
        self._stack.copy_(obs.repeat(1, self.frame_stack, 1, 1))
        infos = self._reset_infos(latency_infos)
        self._last_frame_skips[:] = self._latency_raw_frames(latency_infos)
        self._has_reset = True
        return self._obs(), infos

    def step(self, action):
        if not self._has_reset:
            self.reset()
        actions = self._normalize_cuda_actions(action)
        issued_actions = actions.detach().cpu().numpy().astype(np.int64, copy=False)
        result = self.core.step_decision(
            self.latency,
            actions,
            active_mask=self._active_mask_gpu,
            episode_step_idx=self._raw_episode_steps_gpu,
            max_episode_steps=self._max_episode_steps,
        )
        rewards_raw = result.rewards
        terminated = result.terminated
        truncated = result.truncated
        raw_frames_advanced = result.raw_frames_advanced
        self._raw_episode_steps += raw_frames_advanced
        self._frame_numbers += raw_frames_advanced
        latency_raw_frames = result.latency_raw_frames
        self._last_frame_skips[:] = latency_raw_frames
        if self.cfg.clip_reward:
            # Clip per raw frame, then sum across the decision window (sum-of-signs),
            # matching the CPU path (RewardTraceWrapper inside ActionLatencyWrapper).
            # The CUDA kernel computes this in result.rewards_clipped{,_gpu}; signing
            # the accumulated reward here (sign-of-sum) would saturate the learner
            # signal to {-1,0,1} and slow learning relative to CPU.
            rewards = result.rewards_clipped.astype(np.float32, copy=False)
            rewards_out = result.rewards_clipped_gpu
        else:
            rewards = rewards_raw.astype(np.float32, copy=False)
            rewards_out = result.rewards_gpu
        done_infos = self.metrics.update(
            issued_actions=issued_actions,
            rewards=rewards_raw,
            terminated=terminated,
            truncated=truncated,
            scores=result.scores,
            active_mask=np.ones(self.num_agents, dtype=np.bool_),
            latency_raw_frames=latency_raw_frames,
            lives=result.lives,
        )
        done_mask = np.logical_or(terminated, truncated)
        core_reset_mask = truncated | (terminated & (result.lives <= 0))
        step_frame_numbers = self._frame_numbers.copy()
        infos = self._infos(
            rewards_raw=rewards_raw,
            terminated=terminated,
            truncated=truncated,
            lives=result.lives,
            frame_numbers=step_frame_numbers,
            done_infos=done_infos,
            latency_result=result,
        )
        reset_obs, reset_infos = self._reset_episode_bookkeeping(
            np.flatnonzero(done_mask),
            core_reset_indices=np.flatnonzero(core_reset_mask),
        )
        self._append_stack(result.observations, done_mask)
        if reset_obs is not None:
            reset_indices = np.flatnonzero(core_reset_mask)
            index_t = torch.as_tensor(reset_indices, dtype=torch.int64, device=self.device)
            self._stack[index_t] = reset_obs[index_t].repeat(1, self.frame_stack, 1, 1)
        for agent_idx, reset_info in reset_infos.items():
            infos[int(agent_idx)]["reset_info"] = reset_info
        self.sample_factory_step_cpu_tensors = {
            "rewards": torch.from_numpy(rewards),
            "terminated": torch.from_numpy(terminated),
            "truncated": torch.from_numpy(truncated),
            "dones": torch.from_numpy(done_mask),
        }
        return self._obs(), rewards_out, result.terminated_gpu, result.truncated_gpu, infos

    def _append_stack(self, obs: torch.Tensor, dones: np.ndarray) -> None:
        if self.frame_stack == 1:
            self._stack.copy_(obs)
        else:
            self._next_stack[:, : -1, :, :].copy_(self._stack[:, 1:, :, :])
            self._next_stack[:, -1:, :, :].copy_(obs)
            self._stack.copy_(self._next_stack)
        done_indices = np.flatnonzero(dones)
        if done_indices.size:
            index_t = torch.as_tensor(done_indices, dtype=torch.int64, device=self.device)
            self._stack[index_t] = obs[index_t].repeat(1, self.frame_stack, 1, 1)

    def _infos(
        self,
        *,
        rewards_raw: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        lives: np.ndarray,
        frame_numbers: np.ndarray,
        done_infos: dict[int, dict[str, Any]],
        latency_result: Any,
    ) -> list[dict[str, Any]]:
        infos: list[dict[str, Any]] = []
        for agent_idx in range(self.num_agents):
            info = {
                "lives": int(lives[agent_idx]),
                "episode_frame_number": int(frame_numbers[agent_idx]),
                "frame_number": int(frame_numbers[agent_idx]),
                "raw_reward": float(rewards_raw[agent_idx]),
            }
            if self.latency.add_latency_info:
                info.update(self.latency.core_decision_info_for_agent(agent_idx, latency_result))
            if int(agent_idx) in done_infos:
                done_info = dict(done_infos[int(agent_idx)])
                if "raw_reward" in done_info:
                    del done_info["raw_reward"]
                if "score" in done_info:
                    del done_info["score"]
                info.update(done_info)
            infos.append(info)
        return infos

    def _reset_episode_bookkeeping(
        self,
        done_indices: Sequence[int],
        *,
        core_reset_indices: Sequence[int],
    ) -> tuple[torch.Tensor | None, dict[int, dict[str, Any]]]:
        if len(done_indices) == 0:
            return None, {}
        reset_indices = [int(agent_idx) for agent_idx in done_indices]
        reset_seeds, reset_latency_infos = self.latency.reset(indices=reset_indices)
        self.metrics.reset(indices=reset_indices)
        self._raw_episode_steps[reset_indices] = 0
        self._raw_episode_steps_gpu[torch.as_tensor(reset_indices, dtype=torch.int64, device=self.device)] = 0
        reset_infos = {
            agent_idx: self._reset_info_for_latency_info(reset_latency_infos[offset])
            for offset, agent_idx in enumerate(reset_indices)
        }
        core_indices = [int(agent_idx) for agent_idx in core_reset_indices]
        if len(core_indices) == 0:
            return None, reset_infos
        seed_by_agent = {agent_idx: int(reset_seeds[offset]) for offset, agent_idx in enumerate(reset_indices)}
        core_seeds = [seed_by_agent[agent_idx] for agent_idx in core_indices]
        reset_obs, _ = self.core.reset_batch(seeds=core_seeds, indices=core_indices)
        self._reset_frame_numbers(indices=core_indices)
        return reset_obs, reset_infos

    def _obs(self) -> dict[str, torch.Tensor]:
        return {"obs": self._stack}

    def _reset_infos(self, latency_infos: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._reset_info_for_latency_info(info) for info in latency_infos]

    def _reset_info_for_latency_info(self, latency_info: dict[str, Any]) -> dict[str, Any]:
        if self.latency.add_latency_info:
            return dict(latency_info)
        return {}

    def _reset_frame_numbers(self, indices: Sequence[int] | np.ndarray | None = None) -> None:
        base_frames = self.core.cache_index.detach().cpu().numpy().astype(np.int64) + 3
        if indices is None:
            self._frame_numbers[:] = base_frames
        else:
            target_indices = [int(agent_idx) for agent_idx in indices]
            self._frame_numbers[target_indices] = base_frames[target_indices]

    def _normalize_cuda_actions(self, action: Any) -> torch.Tensor:
        if isinstance(action, torch.Tensor):
            actions = action.to(device=self.device, dtype=torch.int64)
        else:
            actions = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        if actions.ndim == 0:
            actions = actions.reshape(1)
        if actions.ndim == 2 and int(actions.shape[1]) == 1:
            actions = actions[:, 0]
        return actions.reshape(-1).contiguous()

    @staticmethod
    def _latency_raw_frames(infos: Sequence[dict[str, Any]]) -> np.ndarray:
        return np.asarray([int(info["latency_raw_frames"]) for info in infos], dtype=np.int64)

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if str(device) == "auto":
            return torch.device("cuda")
        return torch.device(device)

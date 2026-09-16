from __future__ import annotations

from collections import deque
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict as GymDict, Discrete

from latency_bench.core.clock import EnvClock
from training.common.latency_config import (
    build_iid_profile_sampler,
    profile_sampler_max_latency_raw_frames,
    sample_factory_latency_params_from_flat_config,
    sample_factory_latency_seed_from_flat_config,
)
from training.common.batched_metrics import BatchedEpisodeMetrics
from training.common.gpu_latency import CudaLatencyScheduler
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
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
        self.export_env_raw_rgb_frames = cfg.export_env_raw_rgb_frames
        self._env_raw_rgb_frames: list[deque[np.ndarray]] = [
            deque(maxlen=int(self.frame_stack)) for _ in range(self.num_agents)
        ]
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
            obs_stride=self.clock.obs_stride_raw_frames,
            frame_stack=self.frame_stack,
        )
        sub_seeds = [self._sub_seed(agent_idx) for agent_idx in range(self.num_agents)]
        latency_seeds = [
            sample_factory_latency_seed_from_flat_config(
                self.cfg,
                fallback_seed=sub_seeds[agent_idx],
                env_id=self._sub_env_id(agent_idx),
            )
            for agent_idx in range(self.num_agents)
        ]
        latency_params = sample_factory_latency_params_from_flat_config(self.cfg, env_fps=self.clock.env_fps)
        profile_samplers = None
        profile_max_latency_raw_frames = None
        if latency_params.profile_path is not None:
            profile_samplers = [
                build_iid_profile_sampler(
                    latency_params.profile_path,
                    seed=latency_seeds[agent_idx],
                    env_fps=self.clock.env_fps,
                    obs_fps=self.clock.obs_fps,
                )
                for agent_idx in range(self.num_agents)
            ]
            profile_max_latency_raw_frames = profile_sampler_max_latency_raw_frames(
                profile_samplers[0],
                worker_slot=latency_params.profile_worker_slot,
                env_fps=self.clock.env_fps,
            )
        self.latency = CudaLatencyScheduler(
            batch_size=self.num_agents,
            fixed_latency_raw_frames=latency_params.fixed_latency_raw_frames,
            env_fps=self.clock.env_fps,
            obs_fps=self.clock.obs_fps,
            hold_policy=self.cfg.hold_policy,
            ordering_policy=self.cfg.ordering_policy,
            latency_distribution_spec=latency_params.latency_distribution_spec,
            sample_distribution_per_episode=latency_params.sample_distribution_per_episode,
            seeds=latency_seeds,
            env_reset_seed_bases=[seed + 104729 for seed in sub_seeds],
            add_latency_info=self.cfg.add_latency_info,
            max_pending_actions=self.cfg.max_pending_actions,
            latency_samplers=profile_samplers,
            profile_worker_slot=latency_params.profile_worker_slot,
            profile_max_latency_raw_frames=profile_max_latency_raw_frames,
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
        self._active_mask_cpu = np.ones(self.num_agents, dtype=np.bool_)
        self._issued_actions_cpu = np.zeros(self.num_agents, dtype=np.int64)
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
        return self.cfg.seed + 100000 * self._worker_index + self._worker_local_agent_id(agent_idx)

    def _sub_env_id(self, agent_idx: int) -> int:
        return self._env_id * self.num_agents + int(agent_idx)

    def _worker_local_agent_id(self, agent_idx: int) -> int:
        return self._vector_index * self.num_agents + int(agent_idx)

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
        if self.export_env_raw_rgb_frames:
            raw_rgb = self.core.native_rgb_frames()
            self._reset_env_raw_rgb_frames(raw_rgb)
            self._attach_env_raw_rgb_infos(infos)
        return self._obs(), infos

    def step(self, action):
        if not self._has_reset:
            self.reset()
        actions = self._normalize_cuda_actions(action)
        result = self.core.step_decision(
            self.latency,
            actions,
            active_mask=self._active_mask_gpu,
            episode_step_idx=self._raw_episode_steps_gpu,
            max_episode_steps=self._max_episode_steps,
        )
        return self._finish_decision_step(
            actions=actions,
            result=result,
            rewards_raw=result.rewards,
            rewards_clipped=result.rewards_clipped,
            terminated=result.terminated,
            truncated=result.truncated,
            scores=result.scores,
            lives=result.lives,
            raw_frames_advanced=result.raw_frames_advanced,
            latency_raw_frames=result.latency_raw_frames,
            full_infos=True,
        )

    def sample_factory_gpu_step(self, action):
        if not self._has_reset:
            self.reset()
        actions = self._normalize_cuda_actions(action)
        result = self.core.step_decision_gpu(
            self.latency,
            actions,
            active_mask=self._active_mask_gpu,
            episode_step_idx=self._raw_episode_steps_gpu,
            max_episode_steps=self._max_episode_steps,
        )
        rewards_raw = result.rewards_gpu.detach().cpu().numpy().astype(np.float32, copy=False)
        rewards_clipped = result.rewards_clipped_gpu.detach().cpu().numpy().astype(np.float32, copy=False)
        terminated = result.terminated_gpu.detach().cpu().numpy().astype(np.bool_, copy=False)
        truncated = result.truncated_gpu.detach().cpu().numpy().astype(np.bool_, copy=False)
        scores = result.scores_gpu.detach().cpu().numpy().astype(np.int64, copy=False)
        lives = result.lives_gpu.detach().cpu().numpy().astype(np.int64, copy=False)
        raw_frames_advanced = result.raw_frames_advanced_gpu.detach().cpu().numpy().astype(np.int64, copy=False)
        latency_raw_frames = result.latency_raw_frames_gpu.detach().cpu().numpy().astype(np.int64, copy=False)
        return self._finish_decision_step(
            actions=actions,
            result=result,
            rewards_raw=rewards_raw,
            rewards_clipped=rewards_clipped,
            terminated=terminated,
            truncated=truncated,
            scores=scores,
            lives=lives,
            raw_frames_advanced=raw_frames_advanced,
            latency_raw_frames=latency_raw_frames,
            full_infos=False,
        )

    def _finish_decision_step(
        self,
        *,
        actions: torch.Tensor,
        result: Any,
        rewards_raw: np.ndarray,
        rewards_clipped: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        scores: np.ndarray,
        lives: np.ndarray,
        raw_frames_advanced: np.ndarray,
        latency_raw_frames: np.ndarray,
        full_infos: bool,
    ):
        self._raw_episode_steps += raw_frames_advanced
        self._frame_numbers += raw_frames_advanced
        self._last_frame_skips[:] = latency_raw_frames
        if self.cfg.clip_reward:
            # Clip per raw frame, then sum across the decision window (sum-of-signs),
            # matching the CPU path (RewardTraceWrapper inside ActionLatencyWrapper).
            # The CUDA kernel computes this in result.rewards_clipped{,_gpu}; signing
            # the accumulated reward here (sign-of-sum) would saturate the learner
            # signal to {-1,0,1} and slow learning relative to CPU.
            rewards = rewards_clipped.astype(np.float32, copy=False)
            rewards_out = result.rewards_clipped_gpu
        else:
            rewards = rewards_raw.astype(np.float32, copy=False)
            rewards_out = result.rewards_gpu
        done_infos = self.metrics.update(
            issued_actions=self._issued_actions_for_metrics(actions),
            rewards=rewards_raw,
            terminated=terminated,
            truncated=truncated,
            scores=scores,
            active_mask=self._active_mask_cpu,
            latency_raw_frames=latency_raw_frames,
            lives=lives,
        )
        done_mask = np.logical_or(terminated, truncated)
        core_reset_mask = truncated | (terminated & (lives <= 0))
        life_loss_reset_mask = terminated & (lives > 0) & ~truncated
        step_frame_numbers = self._frame_numbers.copy()
        if full_infos:
            infos = self._infos(
                rewards_raw=rewards_raw,
                terminated=terminated,
                truncated=truncated,
                lives=lives,
                frame_numbers=step_frame_numbers,
                done_infos=done_infos,
                latency_result=result,
            )
        else:
            infos = self._sparse_infos(
                rewards_raw=rewards_raw,
                lives=lives,
                frame_numbers=step_frame_numbers,
                done_infos=done_infos,
                latency_result=result,
            )
        reset_obs, reset_infos = self._reset_episode_bookkeeping(
            np.flatnonzero(done_mask),
            core_reset_indices=np.flatnonzero(core_reset_mask),
            life_loss_reset_indices=np.flatnonzero(life_loss_reset_mask),
        )
        self._append_stack(result.observations, done_mask)
        for agent_idx, agent_reset_obs in reset_obs.items():
            self._stack[int(agent_idx)] = agent_reset_obs.repeat(self.frame_stack, 1, 1)
        for agent_idx, reset_info in reset_infos.items():
            infos[int(agent_idx)]["reset_info"] = reset_info
        if self.export_env_raw_rgb_frames:
            raw_rgb = self.core.native_rgb_frames()
            life_loss_indices = np.flatnonzero(life_loss_reset_mask)
            if life_loss_indices.size:
                self._reset_env_raw_rgb_frames(raw_rgb, agent_indices=life_loss_indices)
            self._append_env_raw_rgb_frames(raw_rgb, done_mask=done_mask)
            self._attach_env_raw_rgb_infos(infos)
        self.sample_factory_step_cpu_tensors = {
            "rewards": torch.from_numpy(rewards),
            "terminated": torch.from_numpy(terminated),
            "truncated": torch.from_numpy(truncated),
            "dones": torch.from_numpy(done_mask),
        }
        return self._obs(), rewards_out, result.terminated_gpu, result.truncated_gpu, infos

    def set_active_agents(self, active_mask: Sequence[bool]) -> None:
        """Set which batched agents should advance on subsequent steps.

        Args:
            active_mask: Boolean mask aligned with batched agent indices.
        """
        mask = np.asarray(list(active_mask), dtype=np.bool_)
        self._active_mask_cpu[:] = mask
        self._active_mask_gpu.copy_(torch.as_tensor(mask, dtype=torch.bool, device=self.device))

    def reset_agents(
        self,
        reset_plan: dict[int, int],
        *,
        active_mask: Sequence[bool] | None = None,
    ):
        """Reset selected batched agents using explicit rollout seeds.

        Args:
            reset_plan: Mapping from agent index to reset seed.
            active_mask: Optional active-agent mask to install after reset.

        Returns:
            Full batched observation and per-agent info list.
        """
        if active_mask is not None:
            self.set_active_agents(active_mask)
        if not self._has_reset:
            first_seed = reset_plan[0]
            return self.reset(seed=first_seed)

        reset_indices = [int(agent_idx) for agent_idx in reset_plan.keys()]
        reset_seed_inputs = [int(seed) for seed in reset_plan.values()]
        reset_seeds, reset_latency_infos = self.latency.reset(indices=reset_indices, seeds=reset_seed_inputs)
        self.metrics.reset(indices=reset_indices)
        index_t = torch.as_tensor(reset_indices, dtype=torch.int64, device=self.device)
        self._raw_episode_steps[reset_indices] = 0
        self._raw_episode_steps_gpu[index_t] = 0
        reset_obs, _ = self.core.reset_batch(seeds=reset_seeds, indices=reset_indices)
        self._reset_frame_numbers(indices=reset_indices)
        for agent_idx in reset_indices:
            self._stack[int(agent_idx)] = reset_obs[int(agent_idx)].repeat(self.frame_stack, 1, 1)
        infos = [{} for _ in range(self.num_agents)]
        for offset, agent_idx in enumerate(reset_indices):
            infos[int(agent_idx)] = self._reset_info_for_latency_info(reset_latency_infos[offset])
        if self.export_env_raw_rgb_frames:
            raw_rgb = self.core.native_rgb_frames()
            self._reset_env_raw_rgb_frames(raw_rgb, agent_indices=reset_indices)
            self._attach_env_raw_rgb_infos(infos)
        return self._obs(), infos

    def _issued_actions_for_metrics(self, actions: torch.Tensor) -> np.ndarray:
        if self.metrics._action_labels:
            return actions.detach().cpu().numpy().astype(np.int64, copy=False)
        return self._issued_actions_cpu

    def _append_stack(self, obs_frames: torch.Tensor, dones: np.ndarray) -> None:
        # obs_frames: (B, k, 84, 84) — the k=obs_stride raw frames of this decision window,
        # oldest..newest. The policy stack is the LAST frame_stack RAW frames (matches ALE).
        k = obs_frames.shape[1]
        if self.frame_stack <= k:
            self._stack.copy_(obs_frames[:, -self.frame_stack:, :, :])
        else:
            # frame_stack spans multiple decisions: roll old frames left, append the k new ones.
            self._next_stack[:, : -k, :, :].copy_(self._stack[:, k:, :, :])
            self._next_stack[:, -k:, :, :].copy_(obs_frames)
            self._stack.copy_(self._next_stack)
        done_indices = np.flatnonzero(dones)
        if done_indices.size:
            # On episode end, fill the stack with the newest (terminal) frame; the subsequent
            # reset_obs overwrite in step() replaces it with the post-reset frame.
            index_t = torch.as_tensor(done_indices, dtype=torch.int64, device=self.device)
            self._stack[index_t] = obs_frames[index_t, -1:, :, :].repeat(1, self.frame_stack, 1, 1)

    def _reset_env_raw_rgb_frames(
        self,
        raw_rgb: torch.Tensor,
        agent_indices: Sequence[int] | None = None,
    ) -> None:
        """Initialize exported raw RGB stacks from current native frames.

        Args:
            raw_rgb: Batched native RGB tensor from the CUDA core.
            agent_indices: Optional subset of agents to reset.
        """
        if not self.export_env_raw_rgb_frames:
            return
        frames = raw_rgb.detach().cpu().numpy()
        indices = range(self.num_agents) if agent_indices is None else [int(item) for item in agent_indices]
        for agent_idx in indices:
            self._env_raw_rgb_frames[agent_idx].clear()
            frame = np.array(frames[agent_idx], copy=True)
            for _ in range(self.frame_stack):
                self._env_raw_rgb_frames[agent_idx].append(frame.copy())

    def _append_env_raw_rgb_frames(self, raw_rgb: torch.Tensor, *, done_mask: np.ndarray) -> None:
        """Append latest native RGB frames for non-terminal active agents.

        Args:
            raw_rgb: Batched native RGB tensor from the CUDA core.
            done_mask: Per-agent done mask for the decision step.
        """
        if not self.export_env_raw_rgb_frames:
            return
        frames = raw_rgb.detach().cpu().numpy()
        append_mask = self._active_mask_cpu & ~done_mask
        for agent_idx in np.flatnonzero(append_mask):
            self._env_raw_rgb_frames[int(agent_idx)].append(np.array(frames[int(agent_idx)], copy=True))

    def _attach_env_raw_rgb_infos(self, infos: list[dict[str, Any]]) -> None:
        """Attach raw RGB stacks to per-agent infos when export mode is enabled.

        Args:
            infos: Mutable per-agent info list.
        """
        if not self.export_env_raw_rgb_frames:
            return
        for agent_idx, info in enumerate(infos):
            updated = dict(info)
            updated[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = np.stack(tuple(self._env_raw_rgb_frames[agent_idx]), axis=0)
            infos[agent_idx] = updated

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
                done_info = {
                    key: value
                    for key, value in done_infos[int(agent_idx)].items()
                    if key not in ("raw_reward", "score")
                }
                info.update(done_info)
            infos.append(info)
        return infos

    def _sparse_infos(
        self,
        *,
        rewards_raw: np.ndarray,
        lives: np.ndarray,
        frame_numbers: np.ndarray,
        done_infos: dict[int, dict[str, Any]],
        latency_result: Any,
    ) -> list[dict[str, Any]]:
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_agents)]
        for agent_idx, done_info_src in done_infos.items():
            info = {
                "lives": int(lives[agent_idx]),
                "episode_frame_number": int(frame_numbers[agent_idx]),
                "frame_number": int(frame_numbers[agent_idx]),
                "raw_reward": float(rewards_raw[agent_idx]),
            }
            if self.latency.add_latency_info:
                info.update(self._gpu_decision_info_for_agent(agent_idx, latency_result))
            done_info = {
                key: value
                for key, value in done_info_src.items()
                if key not in ("raw_reward", "score")
            }
            info.update(done_info)
            infos[int(agent_idx)] = info
        return infos

    def _gpu_decision_info_for_agent(self, agent_idx: int, decision: Any) -> dict[str, Any]:
        latency_raw_frames = int(decision.latency_raw_frames_gpu[agent_idx].item())
        info = {
            "episode_idx": int(self.latency.episode_idx_cpu[agent_idx]),
            "decision_step_idx": int(decision.decision_step_indices_gpu[agent_idx].item() - 1),
            "env_step_idx": int(decision.env_step_indices_gpu[agent_idx].item()),
            "raw_frame": int(decision.env_step_indices_gpu[agent_idx].item()),
            "applied_action": int(decision.applied_actions_gpu[agent_idx].item()),
            "pending_action_count": int(decision.pending_counts_gpu[agent_idx].item()),
            "latency_raw_frames": latency_raw_frames,
            "latency_ms": float(latency_raw_frames) * self.latency.clock.frame_ms,
            "env_fps": float(self.latency.clock.env_fps),
            "obs_fps": float(self.latency.clock.obs_fps),
            "obs_stride_raw_frames": int(self.latency.clock.obs_stride_raw_frames),
            "hold_policy": self.latency.hold_policy.value,
            "ordering_policy": self.latency.ordering_policy.value,
            "queued_action": int(decision.queued_actions_gpu[agent_idx].item()),
            "ready_raw_frame": int(decision.ready_raw_frames_gpu[agent_idx].item()),
            "raw_frames_advanced": int(decision.raw_frames_advanced_gpu[agent_idx].item()),
        }
        if self.latency.latency_distribution_spec is not None:
            info["latency_distribution"] = self.latency.latency_distribution_spec.to_dict()
            info["latency_distribution_label"] = str(self.latency.latency_distribution_label)
        return info

    def _reset_episode_bookkeeping(
        self,
        done_indices: Sequence[int],
        *,
        core_reset_indices: Sequence[int],
        life_loss_reset_indices: Sequence[int],
    ) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, Any]]]:
        if len(done_indices) == 0:
            return {}, {}
        reset_indices = [int(agent_idx) for agent_idx in done_indices]
        reset_seeds, reset_latency_infos = self.latency.reset(indices=reset_indices)
        self.metrics.reset(indices=reset_indices)
        self._raw_episode_steps[reset_indices] = 0
        self._raw_episode_steps_gpu[torch.as_tensor(reset_indices, dtype=torch.int64, device=self.device)] = 0
        reset_infos = {
            agent_idx: self._reset_info_for_latency_info(reset_latency_infos[offset])
            for offset, agent_idx in enumerate(reset_indices)
        }
        reset_obs_by_agent: dict[int, torch.Tensor] = {}
        core_indices = [int(agent_idx) for agent_idx in core_reset_indices]
        if len(core_indices) > 0:
            seed_by_agent = {agent_idx: int(reset_seeds[offset]) for offset, agent_idx in enumerate(reset_indices)}
            core_seeds = [seed_by_agent[agent_idx] for agent_idx in core_indices]
            core_reset_obs, _ = self.core.reset_batch(seeds=core_seeds, indices=core_indices)
            self._reset_frame_numbers(indices=core_indices)
            for agent_idx in core_indices:
                reset_obs_by_agent[int(agent_idx)] = core_reset_obs[int(agent_idx)].clone()
        life_loss_indices = [int(agent_idx) for agent_idx in life_loss_reset_indices]
        if len(life_loss_indices) > 0:
            life_loss_reset_obs = self.core.soft_reset_life_loss_batch(indices=life_loss_indices)
            self._frame_numbers[life_loss_indices] += 3
            for agent_idx in life_loss_indices:
                reset_obs_by_agent[int(agent_idx)] = life_loss_reset_obs[int(agent_idx)].clone()
        return reset_obs_by_agent, reset_infos

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

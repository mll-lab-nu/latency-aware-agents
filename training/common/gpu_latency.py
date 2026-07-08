from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from latency_bench.core.clock import EnvClock
from latency_bench.core.latency_distribution import (
    LatencyDistributionSpec,
    derive_latency_seed,
    distribution_max_frame_latency,
    latency_distribution_label,
    make_latency_distribution_sampler,
)
from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy
from training.common.latency_config import build_latency_info
from training.common.utils import batch_indices, seed_values
from training.flappy_sf.flappy_cuda_extension import load_flappy_cuda_extension


@dataclass
class CudaLatencyDecisionResult:
    queued_actions: torch.Tensor
    applied_actions: torch.Tensor
    latency_raw_frames: torch.Tensor
    ready_raw_frames: torch.Tensor
    raw_frames_advanced: torch.Tensor
    pending_counts: torch.Tensor


@dataclass
class CudaCoreDecisionResult:
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    scores: np.ndarray
    raw_frames_advanced: np.ndarray
    latency_raw_frames: np.ndarray
    queued_actions: np.ndarray
    applied_actions: np.ndarray
    ready_raw_frames: np.ndarray
    pending_counts: np.ndarray
    env_step_indices: np.ndarray
    decision_step_indices: np.ndarray
    timeout_mask: np.ndarray
    rewards_gpu: torch.Tensor
    terminated_gpu: torch.Tensor
    truncated_gpu: torch.Tensor


class CudaLatencyScheduler:
    def __init__(
        self,
        *,
        batch_size: int,
        fixed_latency_raw_frames: int,
        env_fps: float,
        obs_fps: float,
        hold_policy: str | ActionHoldPolicy,
        ordering_policy: str | OrderingPolicy,
        seeds: Sequence[int],
        env_reset_seed_bases: Sequence[int],
        add_latency_info: bool,
        device: str | torch.device,
        latency_distribution_spec: LatencyDistributionSpec | None = None,
        sample_distribution_per_episode: bool = False,
        max_pending_actions: int | None = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.fixed_latency_raw_frames = int(fixed_latency_raw_frames)
        self.latency_distribution_spec = latency_distribution_spec
        self.sample_distribution_per_episode = bool(sample_distribution_per_episode)
        self.latency_distribution_label = (
            None if self.latency_distribution_spec is None else latency_distribution_label(self.latency_distribution_spec)
        )
        self.max_frame_latency = (
            distribution_max_frame_latency(self.latency_distribution_spec)
            if self.latency_distribution_spec is not None
            else int(self.fixed_latency_raw_frames)
        )
        self.max_pending_actions = None if max_pending_actions is None else int(max_pending_actions)
        self.clock = EnvClock(env_fps=float(env_fps), obs_fps=float(obs_fps))
        self.hold_policy = (
            hold_policy
            if isinstance(hold_policy, ActionHoldPolicy)
            else ActionHoldPolicy(str(hold_policy).strip().lower())
        )
        self.ordering_policy = (
            ordering_policy
            if isinstance(ordering_policy, OrderingPolicy)
            else OrderingPolicy(str(ordering_policy).strip().lower())
        )
        self.base_latency_seeds = np.asarray([int(seed) for seed in seeds], dtype=np.int64)
        self.env_reset_seed_bases = np.asarray([int(seed) for seed in env_reset_seed_bases], dtype=np.int64)
        self.add_latency_info = bool(add_latency_info)
        self.device = torch.device(device)
        self.ext = load_flappy_cuda_extension()
        self.latency_distribution_type = self._latency_distribution_type()
        self.latency_distribution_mean = (
            0.0 if self.latency_distribution_spec is None or self.latency_distribution_spec.mean is None
            else float(self.latency_distribution_spec.mean)
        )
        self.latency_distribution_std = (
            0.0 if self.latency_distribution_spec is None or self.latency_distribution_spec.std is None
            else float(self.latency_distribution_spec.std)
        )
        self.latency_distribution_min = (
            0.0 if self.latency_distribution_spec is None else float(self.latency_distribution_spec.min)
        )
        self.latency_distribution_max = (
            0.0 if self.latency_distribution_spec is None else float(self.latency_distribution_spec.max)
        )

        self.episode_idx_cpu = np.zeros(self.batch_size, dtype=np.int64)
        self.episode_idx = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.env_step_idx = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.decision_step_idx = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.next_action_id = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.current_latency_raw_frames = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.current_applied_action = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.persistence_remaining = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)

        self.queue_capacity = int(self.max_pending_actions or (self.max_frame_latency + self.clock.obs_stride_raw_frames + 2))
        self.pending_actions = torch.zeros(
            (self.batch_size, self.queue_capacity),
            dtype=torch.int64,
            device=self.device,
        )
        self.pending_ready_raw_frames = torch.zeros_like(self.pending_actions)
        self.pending_action_ids = torch.zeros_like(self.pending_actions)
        self.scratch_actions = torch.zeros_like(self.pending_actions)
        self.scratch_ready_raw_frames = torch.zeros_like(self.pending_actions)
        self.scratch_action_ids = torch.zeros_like(self.pending_actions)
        self.pending_head = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.pending_count = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self.latency_rng_state = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)

        self._decision_latency_raw_frames = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self._decision_ready_raw_frames = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self._decision_queued_actions = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self._decision_timeout = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        self._raw_actions = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        reset_latency = 0 if self.latency_distribution_spec is not None else int(self.fixed_latency_raw_frames)
        self._latency_raw_frames_cpu = np.full(self.batch_size, reset_latency, dtype=np.int64)

    def reset(
        self,
        *,
        indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        seeds: Sequence[int | None] | int | None = None,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        target_indices = self._indices(indices)
        seed_values = self._seed_values(seeds, len(target_indices))
        if not self.add_latency_info:
            reset_seeds: list[int] = []
            infos: list[dict[str, Any]] = []
            reset_latencies = self._reset_latencies_for_indices(target_indices)
            for offset, agent_idx in enumerate(target_indices):
                reset_latency = reset_latencies[offset]
                seed = seed_values[offset]
                if seed is None:
                    seed = int(self.env_reset_seed_bases[agent_idx] + int(self.episode_idx_cpu[agent_idx]))
                reset_seeds.append(int(seed))
                infos.append(
                    {
                        "episode_idx": int(self.episode_idx_cpu[agent_idx]),
                        "decision_step_idx": 0,
                        "env_step_idx": 0,
                        "raw_frame": 0,
                        "applied_action": 0,
                        "pending_action_count": 0,
                        "latency_raw_frames": int(reset_latency),
                        "latency_ms": float(reset_latency) * self.clock.frame_ms,
                        "env_fps": float(self.clock.env_fps),
                        "obs_fps": float(self.clock.obs_fps),
                        "obs_stride_raw_frames": int(self.clock.obs_stride_raw_frames),
                        "hold_policy": self.hold_policy.value,
                        "ordering_policy": self.ordering_policy.value,
                    }
                )
                if self.latency_distribution_spec is not None:
                    infos[-1]["latency_distribution"] = self.latency_distribution_spec.to_dict()
                    infos[-1]["latency_distribution_label"] = str(self.latency_distribution_label)
            index_t = torch.as_tensor(target_indices, dtype=torch.int64, device=self.device).contiguous()
            self._reset_latency_rng_states(target_indices)
            self.ext.flappy_latency_reset(
                index_t,
                self.pending_head,
                self.pending_count,
                self.current_applied_action,
                self.env_step_idx,
                self.decision_step_idx,
                self.next_action_id,
                self.persistence_remaining,
                self.current_latency_raw_frames,
                self.episode_idx,
                int(reset_latency),
                int(len(target_indices)),
            )
            self.episode_idx_cpu[np.asarray(target_indices, dtype=np.int64)] += 1
            self.current_latency_raw_frames[index_t] = torch.as_tensor(
                reset_latencies,
                dtype=torch.int64,
                device=self.device,
            )
            self._latency_raw_frames_cpu[np.asarray(target_indices, dtype=np.int64)] = reset_latencies
            return reset_seeds, infos
        reset_seeds: list[int] = []
        infos: list[dict[str, Any]] = []
        for offset, agent_idx in enumerate(target_indices):
            self.pending_head[agent_idx] = 0
            self.pending_count[agent_idx] = 0
            self.current_applied_action[agent_idx] = 0
            self.env_step_idx[agent_idx] = 0
            self.decision_step_idx[agent_idx] = 0
            self.next_action_id[agent_idx] = 0
            self.persistence_remaining[agent_idx] = 0
            self.current_latency_raw_frames[agent_idx] = self._reset_latency_for_agent(agent_idx)
            seed = seed_values[offset]
            if seed is None:
                seed = int(self.env_reset_seed_bases[agent_idx] + int(self.episode_idx[agent_idx].item()))
            reset_seeds.append(int(seed))
            infos.append(self._latency_info(agent_idx, int(self.current_latency_raw_frames[agent_idx].item())))
            self.episode_idx[agent_idx] += 1
            self.episode_idx_cpu[agent_idx] += 1
        self._reset_latency_rng_states(target_indices, episode_offset=-1)
        self._latency_raw_frames_cpu[np.asarray(target_indices, dtype=np.int64)] = (
            self.current_latency_raw_frames[
                torch.as_tensor(target_indices, dtype=torch.int64, device=self.device)
            ]
            .detach()
            .cpu()
            .numpy()
        )
        return reset_seeds, infos

    def begin_decision(self, actions: torch.Tensor, active_mask: torch.Tensor) -> None:
        action_t = actions.to(device=self.device, dtype=torch.int64).reshape(-1).contiguous()
        active = active_mask.to(device=self.device, dtype=torch.bool).reshape(-1).contiguous()
        self.ext.flappy_latency_begin(
            action_t,
            active,
            self.env_step_idx,
            self.next_action_id,
            self.current_latency_raw_frames,
            self.pending_actions,
            self.pending_ready_raw_frames,
            self.pending_action_ids,
            self.pending_head,
            self.pending_count,
            self._decision_queued_actions,
            self._decision_latency_raw_frames,
            self._decision_ready_raw_frames,
            self.latency_rng_state,
            int(self.fixed_latency_raw_frames),
            int(self.latency_distribution_type),
            float(self.latency_distribution_mean),
            float(self.latency_distribution_std),
            float(self.latency_distribution_min),
            float(self.latency_distribution_max),
            int(self.batch_size),
            int(self.queue_capacity),
        )

    def raw_actions(self, active_mask: torch.Tensor) -> torch.Tensor:
        active = active_mask.to(device=self.device, dtype=torch.bool).reshape(-1).contiguous()
        self.ext.flappy_latency_raw_actions(
            active,
            self.env_step_idx,
            self.current_applied_action,
            self.persistence_remaining,
            self.pending_actions,
            self.pending_ready_raw_frames,
            self.pending_action_ids,
            self.pending_head,
            self.pending_count,
            self.scratch_actions,
            self.scratch_ready_raw_frames,
            self.scratch_action_ids,
            self._raw_actions,
            int(self.clock.obs_stride_raw_frames if self.hold_policy == ActionHoldPolicy.HOLD else 1),
            int(0 if self.ordering_policy == OrderingPolicy.ISSUE_ORDER_FIFO else 1),
            int(self.batch_size),
            int(self.queue_capacity),
        )
        return self._raw_actions

    def record_raw_step(self, advanced_mask: torch.Tensor) -> None:
        advanced = advanced_mask.to(device=self.device, dtype=torch.bool).reshape(-1).contiguous()
        self.ext.flappy_latency_record(advanced, self.env_step_idx, int(self.batch_size))

    def step_core_decision_numpy(
        self,
        core: Any,
        actions: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        episode_step_idx: torch.Tensor,
        max_episode_steps: int,
    ) -> CudaCoreDecisionResult:
        action_t = actions.to(device=self.device, dtype=torch.int64).reshape(-1).contiguous()
        active = active_mask.to(device=self.device, dtype=torch.bool).reshape(-1).contiguous()
        episode_steps = episode_step_idx.view(-1)
        reset_pipes_in_kernel = self.clock.obs_stride_raw_frames > 1
        if reset_pipes_in_kernel:
            reset_upper_pipe_xy, reset_lower_pipe_xy, reset_pipe_slot_capacity = core.prepare_pipe_resets_for_decision(
                active,
                int(self.clock.obs_stride_raw_frames),
            )
        else:
            reset_upper_pipe_xy = core._pipe_reset_upper_xy
            reset_lower_pipe_xy = core._pipe_reset_lower_xy
            reset_pipe_slot_capacity = int(core._pipe_reset_slot_capacity)
        self.ext.flappy_latency_core_decision(
            core.player_x,
            core.player_y,
            core.player_vel_y,
            core.player_rot,
            core.player_idx,
            core._player_idx_cycle_pos,
            core.loop_iter,
            core.score,
            core.score_limit,
            core.ground_x,
            core.ground_y,
            core.upper_pipe_xy,
            core.lower_pipe_xy,
            action_t,
            active,
            self.env_step_idx,
            self.decision_step_idx,
            self.next_action_id,
            self.current_latency_raw_frames,
            self.current_applied_action,
            self.persistence_remaining,
            self.pending_actions,
            self.pending_ready_raw_frames,
            self.pending_action_ids,
            self.pending_head,
            self.pending_count,
            self.scratch_actions,
            self.scratch_ready_raw_frames,
            self.scratch_action_ids,
            self._decision_queued_actions,
            self._decision_latency_raw_frames,
            self._decision_ready_raw_frames,
            self._decision_timeout,
            core._packed_decision,
            core._decision_rewards,
            core._decision_terminated,
            core._decision_truncated,
            episode_steps,
            reset_upper_pipe_xy,
            reset_lower_pipe_xy,
            self.latency_rng_state,
            int(self.fixed_latency_raw_frames),
            int(self.latency_distribution_type),
            float(self.latency_distribution_mean),
            float(self.latency_distribution_std),
            float(self.latency_distribution_min),
            float(self.latency_distribution_max),
            int(self.clock.obs_stride_raw_frames),
            int(self.clock.obs_stride_raw_frames if self.hold_policy == ActionHoldPolicy.HOLD else 1),
            int(0 if self.ordering_policy == OrderingPolicy.ISSUE_ORDER_FIFO else 1),
            int(max_episode_steps),
            int(reset_pipes_in_kernel),
            int(reset_pipe_slot_capacity),
            int(self.batch_size),
            int(self.queue_capacity),
            int(core._base_shift),
        )
        packed = core._packed_decision.detach().cpu().numpy()
        if not reset_pipes_in_kernel:
            core._reset_offscreen_pipes_from_numpy(packed[:, 5:8])
        self._latency_raw_frames_cpu[:] = packed[:, 8].astype(np.int64)
        return CudaCoreDecisionResult(
            rewards=packed[:, 0].copy(),
            terminated=packed[:, 1].astype(np.bool_),
            truncated=packed[:, 2].astype(np.bool_),
            scores=packed[:, 3].astype(np.int64),
            raw_frames_advanced=packed[:, 4].astype(np.int64),
            latency_raw_frames=self._latency_raw_frames_cpu.copy(),
            queued_actions=self._decision_queued_actions.detach().cpu().numpy().copy(),
            applied_actions=self.current_applied_action.detach().cpu().numpy().copy(),
            ready_raw_frames=self._decision_ready_raw_frames.detach().cpu().numpy().copy(),
            pending_counts=self.pending_count.detach().cpu().numpy().copy(),
            env_step_indices=self.env_step_idx.detach().cpu().numpy().copy(),
            decision_step_indices=self.decision_step_idx.detach().cpu().numpy().copy(),
            timeout_mask=self._decision_timeout.detach().cpu().numpy().copy(),
            rewards_gpu=core._decision_rewards,
            terminated_gpu=core._decision_terminated,
            truncated_gpu=core._decision_truncated,
        )

    def finish_decision(
        self,
        active_mask: torch.Tensor,
        raw_frames_advanced: torch.Tensor,
    ) -> CudaLatencyDecisionResult:
        active = active_mask.to(device=self.device, dtype=torch.bool).reshape(-1).contiguous()
        advanced = raw_frames_advanced.to(device=self.device, dtype=torch.int64).reshape(-1)
        result = CudaLatencyDecisionResult(
            queued_actions=self._decision_queued_actions.clone(),
            applied_actions=self.current_applied_action.clone(),
            latency_raw_frames=self._decision_latency_raw_frames.clone(),
            ready_raw_frames=self._decision_ready_raw_frames.clone(),
            raw_frames_advanced=advanced.clone(),
            pending_counts=self.pending_count.clone(),
        )
        self.ext.flappy_latency_finish(active, self.decision_step_idx, int(self.batch_size))
        self._latency_raw_frames_cpu[:] = result.latency_raw_frames.detach().cpu().numpy()
        return result

    def fixed_latency_raw_frames_numpy(self) -> np.ndarray:
        return self._latency_raw_frames_cpu

    def latency_info_for_reset(self, indices: Sequence[int] | np.ndarray | torch.Tensor | None = None) -> list[dict[str, Any]]:
        target_indices = self._indices(indices)
        return [
            self._latency_info(agent_idx, int(self.current_latency_raw_frames[agent_idx].item()))
            if self.add_latency_info
            else {}
            for agent_idx in target_indices
        ]

    def decision_info_for_agent(
        self,
        agent_idx: int,
        decision: CudaLatencyDecisionResult,
    ) -> dict[str, Any]:
        info = self._latency_info(agent_idx, int(decision.latency_raw_frames[agent_idx].item()))
        info["decision_step_idx"] = int(self.decision_step_idx[agent_idx].item() - 1)
        info["queued_action"] = int(decision.queued_actions[agent_idx].item())
        info["ready_raw_frame"] = int(decision.ready_raw_frames[agent_idx].item())
        info["raw_frames_advanced"] = int(decision.raw_frames_advanced[agent_idx].item())
        info["obs_stride_raw_frames"] = int(self.clock.obs_stride_raw_frames)
        return info

    def core_decision_info_for_agent(
        self,
        agent_idx: int,
        decision: CudaCoreDecisionResult,
    ) -> dict[str, Any]:
        latency_raw_frames = int(decision.latency_raw_frames[agent_idx])
        info = {
            "episode_idx": int(self.episode_idx_cpu[agent_idx]),
            "decision_step_idx": int(decision.decision_step_indices[agent_idx] - 1),
            "env_step_idx": int(decision.env_step_indices[agent_idx]),
            "raw_frame": int(decision.env_step_indices[agent_idx]),
            "applied_action": int(decision.applied_actions[agent_idx]),
            "pending_action_count": int(decision.pending_counts[agent_idx]),
            "latency_raw_frames": latency_raw_frames,
            "latency_ms": float(latency_raw_frames) * self.clock.frame_ms,
            "env_fps": float(self.clock.env_fps),
            "obs_fps": float(self.clock.obs_fps),
            "obs_stride_raw_frames": int(self.clock.obs_stride_raw_frames),
            "hold_policy": self.hold_policy.value,
            "ordering_policy": self.ordering_policy.value,
            "queued_action": int(decision.queued_actions[agent_idx]),
            "ready_raw_frame": int(decision.ready_raw_frames[agent_idx]),
            "raw_frames_advanced": int(decision.raw_frames_advanced[agent_idx]),
        }
        if self.latency_distribution_spec is not None:
            info["latency_distribution"] = self.latency_distribution_spec.to_dict()
            info["latency_distribution_label"] = str(self.latency_distribution_label)
        return info

    def _latency_info(self, agent_idx: int, latency_value: int) -> dict[str, Any]:
        return build_latency_info(
            episode_idx=self.episode_idx[agent_idx].item(),
            decision_step_idx=self.decision_step_idx[agent_idx].item(),
            env_step_idx=self.env_step_idx[agent_idx].item(),
            applied_action=int(self.current_applied_action[agent_idx].item()),
            pending_action_count=self.pending_count[agent_idx].item(),
            latency_raw_frames=int(latency_value),
            clock=self.clock,
            hold_policy=self.hold_policy,
            ordering_policy=self.ordering_policy,
            latency_distribution_spec=self.latency_distribution_spec,
            latency_distribution_label=self.latency_distribution_label,
        )

    def _reset_latency_rng_states(self, target_indices: Sequence[int], *, episode_offset: int = 0) -> None:
        if self.latency_distribution_spec is None:
            return
        states = np.asarray(
            [
                derive_latency_seed(
                    int(self.base_latency_seeds[int(agent_idx)]),
                    episode_idx=int(self.episode_idx_cpu[int(agent_idx)]) + int(episode_offset),
                )
                for agent_idx in target_indices
            ],
            dtype=np.int64,
        )
        self.latency_rng_state[torch.as_tensor(target_indices, dtype=torch.int64, device=self.device)] = torch.as_tensor(
            states,
            dtype=torch.int64,
            device=self.device,
        )

    def _latency_distribution_type(self) -> int:
        if self.latency_distribution_spec is None:
            return 0
        if self.sample_distribution_per_episode:
            return 3
        return {"uniform": 1, "normal": 2}[self.latency_distribution_spec.type]

    def _reset_latencies_for_indices(self, target_indices: Sequence[int]) -> np.ndarray:
        return np.asarray(
            [self._reset_latency_for_agent(int(agent_idx)) for agent_idx in target_indices],
            dtype=np.int64,
        )

    def _reset_latency_for_agent(self, agent_idx: int) -> int:
        if self.latency_distribution_spec is None:
            return int(self.fixed_latency_raw_frames)
        if self.sample_distribution_per_episode:
            seed = derive_latency_seed(
                int(self.base_latency_seeds[int(agent_idx)]),
                episode_idx=int(self.episode_idx_cpu[int(agent_idx)]),
            )
            return int(make_latency_distribution_sampler(self.latency_distribution_spec, seed=seed).sample_frame())
        return 0

    def _indices(self, indices: Sequence[int] | np.ndarray | torch.Tensor | None) -> list[int]:
        return batch_indices(indices, batch_size=self.batch_size)

    @staticmethod
    def _seed_values(seeds: Sequence[int | None] | int | None, count: int) -> list[int | None]:
        return seed_values(seeds, count)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

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


@dataclass
class BatchedLatencyDecisionResult:
    queued_actions: np.ndarray
    applied_actions: np.ndarray
    latency_raw_frames: np.ndarray
    ready_raw_frames: np.ndarray
    raw_frames_advanced: np.ndarray
    pending_counts: np.ndarray


class BatchedFlappyLatencyScheduler:
    def __init__(
        self,
        *,
        batch_size: int,
        fixed_latency_raw_frames: int,
        env_fps: float,
        obs_fps: float,
        hold_policy: str | ActionHoldPolicy,
        ordering_policy: str | OrderingPolicy,
        latency_distribution_spec: LatencyDistributionSpec | None,
        sample_distribution_per_episode: bool = False,
        seeds: Sequence[int],
        env_reset_seed_bases: Sequence[int],
        add_latency_info: bool,
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

        self.episode_idx = np.zeros(self.batch_size, dtype=np.int64)
        self.env_step_idx = np.zeros(self.batch_size, dtype=np.int64)
        self.decision_step_idx = np.zeros(self.batch_size, dtype=np.int64)
        self.next_action_id = np.zeros(self.batch_size, dtype=np.int64)
        self.current_latency_raw_frames = np.zeros(self.batch_size, dtype=np.int64)
        self.current_applied_action = np.zeros(self.batch_size, dtype=np.int64)
        self.persistence_remaining = np.zeros(self.batch_size, dtype=np.int64)
        queue_capacity = int(self.max_pending_actions or (self.max_frame_latency + self.clock.obs_stride_raw_frames + 2))
        self.pending_actions = np.zeros((self.batch_size, queue_capacity), dtype=np.int64)
        self.pending_ready_raw_frames = np.zeros((self.batch_size, queue_capacity), dtype=np.int64)
        self.pending_action_ids = np.zeros((self.batch_size, queue_capacity), dtype=np.int64)
        self.pending_head = np.zeros(self.batch_size, dtype=np.int64)
        self.pending_count = np.zeros(self.batch_size, dtype=np.int64)
        self.latency_distribution_samplers = [None for _ in range(self.batch_size)]
        self._decision_latency_raw_frames = np.zeros(self.batch_size, dtype=np.int64)
        self._decision_ready_raw_frames = np.zeros(self.batch_size, dtype=np.int64)
        self._decision_queued_actions = np.zeros(self.batch_size, dtype=np.int64)

    def reset(
        self,
        *,
        indices: Sequence[int] | np.ndarray | None = None,
        seeds: Sequence[int | None] | int | None = None,
    ) -> tuple[list[int], list[dict[str, Any]]]:
        target_indices = self._indices(indices)
        seed_values = self._seed_values(seeds, len(target_indices))
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
            self._reset_latency_distribution_sampler(agent_idx)
            if self.latency_distribution_spec is None:
                self.current_latency_raw_frames[agent_idx] = self.fixed_latency_raw_frames
            elif self.sample_distribution_per_episode:
                self.current_latency_raw_frames[agent_idx] = self.latency_distribution_samplers[agent_idx].sample_frame()
            else:
                self.current_latency_raw_frames[agent_idx] = 0
            seed = seed_values[offset]
            if seed is None:
                seed = int(self.env_reset_seed_bases[agent_idx] + self.episode_idx[agent_idx])
            reset_seeds.append(int(seed))
            infos.append(self._latency_info(agent_idx, int(self.current_latency_raw_frames[agent_idx])))
            self.episode_idx[agent_idx] += 1
        return reset_seeds, infos

    def begin_decision(self, actions: Sequence[int] | np.ndarray, active_mask: Sequence[bool] | np.ndarray) -> None:
        action_arr = np.asarray(actions, dtype=np.int64).reshape(-1)
        active = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        active_indices = np.flatnonzero(active)
        latency_raw_frames = np.full(self.batch_size, int(self.fixed_latency_raw_frames), dtype=np.int64)
        if self.latency_distribution_spec is not None:
            for agent_idx in active_indices:
                latency_raw_frames[agent_idx] = self._next_latency_raw_frames(int(agent_idx))
        self.current_latency_raw_frames[active] = latency_raw_frames[active]
        ready_raw_frames = self.env_step_idx + latency_raw_frames
        self._decision_latency_raw_frames[active] = latency_raw_frames[active]
        self._decision_ready_raw_frames[active] = ready_raw_frames[active]
        self._decision_queued_actions[active] = action_arr[active]

        tail = (self.pending_head + self.pending_count) % self.pending_actions.shape[1]
        self.pending_actions[active_indices, tail[active_indices]] = action_arr[active_indices]
        self.pending_ready_raw_frames[active_indices, tail[active_indices]] = ready_raw_frames[active_indices]
        self.pending_action_ids[active_indices, tail[active_indices]] = self.next_action_id[active_indices]
        self.pending_count[active] += 1
        self.next_action_id[active] += 1
        if self.max_pending_actions is not None and np.any(self.pending_count[active] > self.max_pending_actions):
            raise RuntimeError(
                "pending action queue exceeded latency.max_pending_actions="
                f"{self.max_pending_actions}"
            )

    def raw_actions(self, active_mask: Sequence[bool] | np.ndarray) -> np.ndarray:
        active = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        applied = np.zeros(self.batch_size, dtype=np.int64)
        selected_mask, selected_actions = self._select_ready_actions(active)
        selected_mask &= active
        self.current_applied_action[selected_mask] = selected_actions[selected_mask]
        self.persistence_remaining[selected_mask] = (
            self.clock.obs_stride_raw_frames if self.hold_policy == ActionHoldPolicy.HOLD else 1
        )
        persistent = active & (self.persistence_remaining > 0)
        applied[persistent] = self.current_applied_action[persistent]
        self.persistence_remaining[persistent] -= 1
        self.current_applied_action[active] = applied[active]
        return applied

    def record_raw_step(self, advanced_mask: Sequence[bool] | np.ndarray) -> None:
        advanced = np.asarray(advanced_mask, dtype=np.bool_).reshape(-1)
        self.env_step_idx[advanced] += 1

    def finish_decision(
        self,
        active_mask: Sequence[bool] | np.ndarray,
        raw_frames_advanced: Sequence[int] | np.ndarray,
    ) -> BatchedLatencyDecisionResult:
        active = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        advanced = np.asarray(raw_frames_advanced, dtype=np.int64).reshape(-1)
        result = BatchedLatencyDecisionResult(
            queued_actions=self._decision_queued_actions.copy(),
            applied_actions=self.current_applied_action.copy(),
            latency_raw_frames=self._decision_latency_raw_frames.copy(),
            ready_raw_frames=self._decision_ready_raw_frames.copy(),
            raw_frames_advanced=advanced.copy(),
            pending_counts=self.pending_count.copy(),
        )
        self.decision_step_idx[active] += 1
        return result

    def latency_info_for_reset(self, indices: Sequence[int] | np.ndarray | None = None) -> list[dict[str, Any]]:
        target_indices = self._indices(indices)
        return [
            self._latency_info(agent_idx, int(self.current_latency_raw_frames[agent_idx]))
            if self.add_latency_info
            else {}
            for agent_idx in target_indices
        ]

    def decision_info_for_agent(
        self,
        agent_idx: int,
        decision: BatchedLatencyDecisionResult,
    ) -> dict[str, Any]:
        info = self._latency_info(agent_idx, int(decision.latency_raw_frames[agent_idx]))
        info["decision_step_idx"] = int(self.decision_step_idx[agent_idx] - 1)
        info["queued_action"] = int(decision.queued_actions[agent_idx])
        info["ready_raw_frame"] = int(decision.ready_raw_frames[agent_idx])
        info["raw_frames_advanced"] = int(decision.raw_frames_advanced[agent_idx])
        info["obs_stride_raw_frames"] = int(self.clock.obs_stride_raw_frames)
        return info

    def _select_ready_actions(self, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.ordering_policy == OrderingPolicy.ISSUE_ORDER_FIFO:
            return self._select_ready_fifo(active)
        return self._select_latest_ready(active)

    def _select_ready_fifo(self, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.arange(self.batch_size)
        has_pending = self.pending_count > 0
        selected = active & has_pending & (self.pending_ready_raw_frames[rows, self.pending_head] <= self.env_step_idx)
        actions = np.zeros(self.batch_size, dtype=np.int64)
        actions[selected] = self.pending_actions[rows[selected], self.pending_head[selected]]
        self.pending_head[selected] = (self.pending_head[selected] + 1) % self.pending_actions.shape[1]
        self.pending_count[selected] -= 1
        return selected, actions

    def _select_latest_ready(self, active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        capacity = self.pending_actions.shape[1]
        offsets = np.arange(capacity, dtype=np.int64)
        rows = np.arange(self.batch_size)[:, None]
        positions = (self.pending_head[:, None] + offsets[None, :]) % capacity
        valid = offsets[None, :] < self.pending_count[:, None]
        ready = active[:, None] & valid & (self.pending_ready_raw_frames[rows, positions] <= self.env_step_idx[:, None])
        selected = np.any(ready, axis=1)
        selected_offsets = np.maximum.accumulate(np.where(ready, offsets[None, :], -1), axis=1)[:, -1]
        selected_positions = (self.pending_head + np.maximum(selected_offsets, 0)) % capacity
        actions = np.zeros(self.batch_size, dtype=np.int64)
        actions[selected] = self.pending_actions[np.arange(self.batch_size)[selected], selected_positions[selected]]

        keep = valid & ~ready
        keep_counts = np.sum(keep, axis=1).astype(np.int64)
        order_key = np.where(keep, offsets[None, :], capacity + offsets[None, :])
        order = np.argsort(order_key, axis=1)
        compact_positions = np.take_along_axis(positions, order, axis=1)
        self.pending_actions[:, :] = np.take_along_axis(self.pending_actions, compact_positions, axis=1)
        self.pending_ready_raw_frames[:, :] = np.take_along_axis(
            self.pending_ready_raw_frames,
            compact_positions,
            axis=1,
        )
        self.pending_action_ids[:, :] = np.take_along_axis(self.pending_action_ids, compact_positions, axis=1)
        self.pending_head[:] = 0
        self.pending_count[selected] = keep_counts[selected]
        return selected, actions

    def _next_latency_raw_frames(self, agent_idx: int) -> int:
        if self.latency_distribution_spec is not None:
            if self.sample_distribution_per_episode:
                return self.current_latency_raw_frames[agent_idx]
            if self.latency_distribution_samplers[agent_idx] is None:
                self._reset_latency_distribution_sampler(agent_idx)
            return int(self.latency_distribution_samplers[agent_idx].sample_frame())
        return int(self.fixed_latency_raw_frames)

    def _reset_latency_distribution_sampler(self, agent_idx: int) -> None:
        if self.latency_distribution_spec is None:
            self.latency_distribution_samplers[agent_idx] = None
            return
        seed = derive_latency_seed(
            int(self.base_latency_seeds[agent_idx]),
            episode_idx=int(self.episode_idx[agent_idx]),
        )
        self.latency_distribution_samplers[agent_idx] = make_latency_distribution_sampler(
            self.latency_distribution_spec,
            seed=seed,
        )

    def _latency_info(self, agent_idx: int, latency_value: int) -> dict[str, Any]:
        return build_latency_info(
            episode_idx=self.episode_idx[agent_idx],
            decision_step_idx=self.decision_step_idx[agent_idx],
            env_step_idx=self.env_step_idx[agent_idx],
            applied_action=int(self.current_applied_action[agent_idx]),
            pending_action_count=self.pending_count[agent_idx],
            latency_raw_frames=int(latency_value),
            clock=self.clock,
            hold_policy=self.hold_policy,
            ordering_policy=self.ordering_policy,
            latency_distribution_spec=self.latency_distribution_spec,
            latency_distribution_label=self.latency_distribution_label,
        )

    def _indices(self, indices: Sequence[int] | np.ndarray | None) -> list[int]:
        return batch_indices(indices, batch_size=self.batch_size)

    def _seed_values(self, seeds: Sequence[int | None] | int | None, count: int) -> list[int | None]:
        return seed_values(seeds, count)

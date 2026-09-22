from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from latency_bench.core.clock import EnvClock
from latency_bench.core.latency_distribution import (
    LatencyDistributionSpec,
    derive_seed,
    distribution_max_frame_latency,
    latency_distribution_label,
    make_latency_distribution_sampler,
)
from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy
from training.common.latency_config import build_latency_info
from training.common.latency_config import (
    build_profile_sampler,
    profile_sampler_max_latency_raw_frames_for_workers,
)
from training.common.utils import batch_indices, seed_values


@dataclass
class BatchedLatencyDecisionResult:
    queued_actions: np.ndarray
    applied_actions: np.ndarray
    latency_raw_frames: np.ndarray
    ready_raw_frames: np.ndarray
    raw_frames_advanced: np.ndarray
    pending_counts: np.ndarray
    command_admitted: np.ndarray
    observation_dropped: np.ndarray
    worker_ids: np.ndarray
    worker_service_raw_frames: np.ndarray


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
        profile_path: str | None = None,
        profile_worker_slot: int | None = None,
        profile_method: str | None = None,
        simulated_worker_capacity: int | None = None,
        sample_distribution_per_episode: bool = False,
        seeds: Sequence[int],
        env_reset_seed_bases: Sequence[int],
        add_latency_info: bool,
        max_pending_actions: int | None = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.fixed_latency_raw_frames = int(fixed_latency_raw_frames)
        self.latency_distribution_spec = latency_distribution_spec
        self.profile_path = profile_path
        self.profile_worker_slot = profile_worker_slot
        self.profile_method = profile_method
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
        self.profile_samplers = [
            build_profile_sampler(
                profile_path,
                method=profile_method,
                seed=int(seed),
                env_fps=self.clock.env_fps,
                obs_fps=self.clock.obs_fps,
            )
            for seed in self.base_latency_seeds
        ] if profile_path is not None else [None for _ in range(self.batch_size)]
        if profile_path is not None:
            self.max_frame_latency = profile_sampler_max_latency_raw_frames_for_workers(
                self.profile_samplers[0],
                profile_worker_slot=profile_worker_slot,
                simulated_worker_capacity=simulated_worker_capacity,
                env_fps=self.clock.env_fps,
            )

        self.episode_idx = np.zeros(self.batch_size, dtype=np.int64)
        self.env_step_idx = np.zeros(self.batch_size, dtype=np.int64)
        self.decision_step_idx = np.zeros(self.batch_size, dtype=np.int64)
        self.next_action_id = np.zeros(self.batch_size, dtype=np.int64)
        self.current_latency_raw_frames = np.zeros(self.batch_size, dtype=np.int64)
        self.current_applied_action = np.zeros(self.batch_size, dtype=np.int64)
        queue_capacity = int(self.max_pending_actions or (self.max_frame_latency + self.clock.obs_stride_raw_frames + 2))
        self.pending_actions = np.zeros((self.batch_size, queue_capacity), dtype=np.int64)
        self.pending_ready_raw_frames = np.zeros((self.batch_size, queue_capacity), dtype=np.int64)
        self.pending_action_ids = np.zeros((self.batch_size, queue_capacity), dtype=np.int64)
        self.pending_head = np.zeros(self.batch_size, dtype=np.int64)
        self.pending_count = np.zeros(self.batch_size, dtype=np.int64)
        self.worker_busy_until = (
            None
            if simulated_worker_capacity is None
            else np.zeros((self.batch_size, simulated_worker_capacity), dtype=np.int64)
        )
        self.submitted_observations = np.zeros(self.batch_size, dtype=np.int64)
        self.dropped_observations = np.zeros(self.batch_size, dtype=np.int64)
        self.latency_distribution_samplers = [None for _ in range(self.batch_size)]
        self._decision_latency_raw_frames = np.zeros(self.batch_size, dtype=np.int64)
        self._decision_ready_raw_frames = np.zeros(self.batch_size, dtype=np.int64)
        self._decision_queued_actions = np.zeros(self.batch_size, dtype=np.int64)
        self._decision_admitted = np.zeros(self.batch_size, dtype=np.bool_)
        self._decision_dropped = np.zeros(self.batch_size, dtype=np.bool_)
        self._decision_worker_ids = np.full(self.batch_size, -1, dtype=np.int64)
        self._decision_worker_service_raw_frames = np.zeros(self.batch_size, dtype=np.int64)

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
            if self.worker_busy_until is not None:
                self.worker_busy_until[agent_idx] = 0
            self.submitted_observations[agent_idx] = 0
            self.dropped_observations[agent_idx] = 0
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
        admitted = active.copy()
        worker_ids = np.full(self.batch_size, -1, dtype=np.int64)
        if self.worker_busy_until is not None:
            idle = self.worker_busy_until <= self.env_step_idx[:, None]
            admitted &= idle.any(axis=1)
            worker_ids[admitted] = idle[admitted].argmax(axis=1)
        dropped = active & ~admitted
        admitted_indices = np.flatnonzero(admitted)
        latency_raw_frames = np.zeros(self.batch_size, dtype=np.int64)
        worker_service_raw_frames = np.zeros(self.batch_size, dtype=np.int64)
        for agent_idx in admitted_indices:
            latency, worker_service = self._next_latency_sample(
                int(agent_idx),
                int(worker_ids[agent_idx]),
            )
            latency_raw_frames[agent_idx] = latency
            worker_service_raw_frames[agent_idx] = worker_service
        self.current_latency_raw_frames[active] = latency_raw_frames[active]
        ready_raw_frames = self.env_step_idx + latency_raw_frames
        self._decision_latency_raw_frames.fill(0)
        self._decision_ready_raw_frames.fill(-1)
        self._decision_queued_actions.fill(0)
        self._decision_admitted[:] = admitted
        self._decision_dropped[:] = dropped
        self._decision_worker_ids[:] = worker_ids
        self._decision_worker_service_raw_frames[:] = worker_service_raw_frames
        self._decision_latency_raw_frames[admitted] = latency_raw_frames[admitted]
        self._decision_ready_raw_frames[admitted] = ready_raw_frames[admitted]
        self._decision_queued_actions[admitted] = action_arr[admitted]

        tail = (self.pending_head + self.pending_count) % self.pending_actions.shape[1]
        self.pending_actions[admitted_indices, tail[admitted_indices]] = action_arr[admitted_indices]
        self.pending_ready_raw_frames[admitted_indices, tail[admitted_indices]] = ready_raw_frames[admitted_indices]
        self.pending_action_ids[admitted_indices, tail[admitted_indices]] = self.next_action_id[admitted_indices]
        self.pending_count[admitted] += 1
        self.next_action_id[admitted] += 1
        self.submitted_observations[admitted] += 1
        self.dropped_observations[dropped] += 1
        if self.worker_busy_until is not None:
            self.worker_busy_until[admitted_indices, worker_ids[admitted_indices]] = (
                self.env_step_idx[admitted] + worker_service_raw_frames[admitted]
            )
        if self.max_pending_actions is not None and np.any(self.pending_count[admitted] > self.max_pending_actions):
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
        if self.hold_policy != ActionHoldPolicy.HOLD:
            self.current_applied_action[active & ~selected_mask] = 0
        applied[active] = self.current_applied_action[active]
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
            command_admitted=self._decision_admitted.copy(),
            observation_dropped=self._decision_dropped.copy(),
            worker_ids=self._decision_worker_ids.copy(),
            worker_service_raw_frames=self._decision_worker_service_raw_frames.copy(),
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
        info["command_admitted"] = bool(decision.command_admitted[agent_idx])
        info["observation_dropped"] = bool(decision.observation_dropped[agent_idx])
        info["worker_id"] = int(decision.worker_ids[agent_idx])
        info["worker_service_raw_frames"] = int(decision.worker_service_raw_frames[agent_idx])
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

        # A newer result also supersedes older requests that have not arrived.
        keep = valid & (offsets[None, :] > selected_offsets[:, None])
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

    def _next_latency_sample(self, agent_idx: int, worker_id: int) -> tuple[int, int]:
        if self.profile_samplers[agent_idx] is not None:
            sample = self.profile_samplers[agent_idx].sample(
                {
                    "episode_id": int(self.episode_idx[agent_idx]),
                    "worker_slot": self.profile_worker_slot if worker_id < 0 else worker_id,
                }
            )
            return int(sample.action_ready_raw_frames), int(sample.worker_service_raw_frames)
        if self.latency_distribution_spec is not None:
            if self.sample_distribution_per_episode:
                latency = int(self.current_latency_raw_frames[agent_idx])
                return latency, latency
            if self.latency_distribution_samplers[agent_idx] is None:
                self._reset_latency_distribution_sampler(agent_idx)
            latency = int(self.latency_distribution_samplers[agent_idx].sample_frame())
            return latency, latency
        latency = int(self.fixed_latency_raw_frames)
        return latency, latency

    def _reset_latency_distribution_sampler(self, agent_idx: int) -> None:
        if self.latency_distribution_spec is None:
            self.latency_distribution_samplers[agent_idx] = None
            return
        seed = derive_seed(
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

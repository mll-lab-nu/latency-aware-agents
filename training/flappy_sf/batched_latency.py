from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from latency_bench.core.clock import EnvClock
from latency_bench.core.latency_distribution import (
    derive_latency_seed,
    distribution_max_frame_latency,
    latency_distribution_label,
    make_latency_distribution_sampler,
    parse_latency_distribution,
)
from latency_bench.core.types import ActionHoldPolicy, OrderingPolicy


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
        latency_distribution_spec: Any,
        seeds: Sequence[int],
        env_reset_seed_bases: Sequence[int],
        add_latency_info: bool,
        max_pending_actions: int | None = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.fixed_latency_raw_frames = int(fixed_latency_raw_frames)
        self.latency_distribution_spec = (
            None if latency_distribution_spec is None else parse_latency_distribution(latency_distribution_spec)
        )
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
        self.pending_events: list[list[dict[str, int]]] = [[] for _ in range(self.batch_size)]
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
            self.pending_events[agent_idx].clear()
            self.current_applied_action[agent_idx] = 0
            self.env_step_idx[agent_idx] = 0
            self.decision_step_idx[agent_idx] = 0
            self.next_action_id[agent_idx] = 0
            self.persistence_remaining[agent_idx] = 0
            self._reset_latency_distribution_sampler(agent_idx)
            self.current_latency_raw_frames[agent_idx] = (
                0 if self.latency_distribution_spec is not None else int(self.fixed_latency_raw_frames)
            )
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
        for agent_idx in np.flatnonzero(active):
            latency_raw_frames = self._next_latency_raw_frames(int(agent_idx))
            self.current_latency_raw_frames[agent_idx] = latency_raw_frames
            ready_raw_frame = int(self.env_step_idx[agent_idx]) + int(latency_raw_frames)
            action = int(action_arr[agent_idx])
            self._decision_latency_raw_frames[agent_idx] = latency_raw_frames
            self._decision_ready_raw_frames[agent_idx] = ready_raw_frame
            self._decision_queued_actions[agent_idx] = action
            self.pending_events[agent_idx].append(
                {
                    "action_id": int(self.next_action_id[agent_idx]),
                    "obs_id": int(self.decision_step_idx[agent_idx]),
                    "action": action,
                    "ready_raw_frame": int(ready_raw_frame),
                }
            )
            self.next_action_id[agent_idx] += 1
            if self.max_pending_actions is not None and len(self.pending_events[agent_idx]) > self.max_pending_actions:
                raise RuntimeError(
                    "pending action queue exceeded latency.max_pending_actions="
                    f"{self.max_pending_actions}"
                )

    def raw_actions(self, active_mask: Sequence[bool] | np.ndarray) -> np.ndarray:
        active = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        applied = np.zeros(self.batch_size, dtype=np.int64)
        for agent_idx in np.flatnonzero(active):
            selected = self._select_event(int(agent_idx))
            if selected is not None:
                self.current_applied_action[agent_idx] = int(selected["action"])
                self.persistence_remaining[agent_idx] = self.clock.obs_stride_raw_frames if self.hold_policy == ActionHoldPolicy.HOLD else 1
            if self.persistence_remaining[agent_idx] <= 0:
                applied[agent_idx] = 0
            else:
                applied[agent_idx] = int(self.current_applied_action[agent_idx])
                self.persistence_remaining[agent_idx] -= 1
        return applied

    def record_raw_step(self, advanced_mask: Sequence[bool] | np.ndarray) -> None:
        advanced = np.asarray(advanced_mask, dtype=np.bool_).reshape(-1)
        self.env_step_idx[advanced] += 1

    def finish_decision(
        self,
        active_mask: Sequence[bool] | np.ndarray,
        raw_frames_advanced: Sequence[int] | np.ndarray,
    ) -> list[dict[str, Any]]:
        active = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        advanced = np.asarray(raw_frames_advanced, dtype=np.int64).reshape(-1)
        infos: list[dict[str, Any]] = [{} for _ in range(self.batch_size)]
        for agent_idx in np.flatnonzero(active):
            info = self._latency_info(agent_idx, int(self._decision_latency_raw_frames[agent_idx]))
            info["queued_action"] = int(self._decision_queued_actions[agent_idx])
            info["ready_raw_frame"] = int(self._decision_ready_raw_frames[agent_idx])
            info["raw_frames_advanced"] = int(advanced[agent_idx])
            info["obs_stride_raw_frames"] = int(self.clock.obs_stride_raw_frames)
            infos[agent_idx] = info if self.add_latency_info else {}
            self.decision_step_idx[agent_idx] += 1
        return infos

    def latency_info_for_reset(self, indices: Sequence[int] | np.ndarray | None = None) -> list[dict[str, Any]]:
        target_indices = self._indices(indices)
        return [
            self._latency_info(agent_idx, int(self.current_latency_raw_frames[agent_idx]))
            if self.add_latency_info
            else {}
            for agent_idx in target_indices
        ]

    def _select_event(self, agent_idx: int) -> dict[str, int] | None:
        pending = self.pending_events[agent_idx]
        if not pending:
            return None
        current_step = int(self.env_step_idx[agent_idx])
        if self.ordering_policy == OrderingPolicy.ISSUE_ORDER_FIFO:
            head = pending[0]
            if int(head["ready_raw_frame"]) <= current_step:
                self.pending_events[agent_idx] = pending[1:]
                return head
            return None
        ready = [event for event in pending if int(event["ready_raw_frame"]) <= current_step]
        not_ready = [event for event in pending if int(event["ready_raw_frame"]) > current_step]
        if not ready:
            return None
        selected = ready[-1]
        self.pending_events[agent_idx] = not_ready
        return selected

    def _next_latency_raw_frames(self, agent_idx: int) -> int:
        if self.latency_distribution_spec is not None:
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
        latency_raw_frames = int(latency_value)
        info: dict[str, Any] = {
            "episode_idx": int(self.episode_idx[agent_idx]),
            "decision_step_idx": int(self.decision_step_idx[agent_idx]),
            "env_step_idx": int(self.env_step_idx[agent_idx]),
            "raw_frame": int(self.env_step_idx[agent_idx]),
            "applied_action": int(self.current_applied_action[agent_idx]),
            "pending_action_count": int(len(self.pending_events[agent_idx])),
            "latency_raw_frames": latency_raw_frames,
            "latency_ms": float(latency_raw_frames) * self.clock.frame_ms,
            "env_fps": float(self.clock.env_fps),
            "obs_fps": float(self.clock.obs_fps),
            "obs_stride_raw_frames": int(self.clock.obs_stride_raw_frames),
            "hold_policy": self.hold_policy.value,
            "ordering_policy": self.ordering_policy.value,
        }
        if self.latency_distribution_spec is not None:
            info["latency_distribution"] = self.latency_distribution_spec.to_dict()
            info["latency_distribution_label"] = str(self.latency_distribution_label)
        return info

    def _indices(self, indices: Sequence[int] | np.ndarray | None) -> list[int]:
        if indices is None:
            return list(range(self.batch_size))
        return [int(item) for item in np.asarray(indices).reshape(-1).tolist()]

    def _seed_values(self, seeds: Sequence[int | None] | int | None, count: int) -> list[int | None]:
        if seeds is None:
            return [None] * count
        if isinstance(seeds, (int, np.integer)):
            return [int(seeds)] * count
        return [None if item is None else int(item) for item in list(seeds)]

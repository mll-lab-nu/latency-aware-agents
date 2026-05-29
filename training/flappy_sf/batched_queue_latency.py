from __future__ import annotations

from bisect import insort_right
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class QueueLatencyDecision:
    queued_actions: np.ndarray
    applied_actions: np.ndarray
    scheduled_apply_steps: np.ndarray
    scheduled_latencies: np.ndarray
    pending_counts: np.ndarray


class BatchedQueueLatencyScheduler:
    """Batched equivalent of sample-factory-backend QueueLatencyWrapper."""

    def __init__(
        self,
        *,
        batch_size: int,
        queue_latency_values: Sequence[int],
        latency_mode: str,
        latency_sampling_mode: str,
        latency_distribution: str,
        latency_normal_mean: float,
        latency_normal_std: float,
        latency_normal_min: int,
        latency_normal_max: int,
        latency_uniform_min: int,
        latency_uniform_max: int,
        episodes_per_stage: int,
        randomize_within_stage: bool,
        within_episode_change_interval: int,
        within_episode_change_prob: float,
        seeds: Sequence[int],
        env_reset_seed_bases: Sequence[int],
        latency_info_keys: Sequence[str],
        add_latency_info: bool,
    ):
        self.batch_size = int(batch_size)
        self.queue_latency_values = [max(0, int(item)) for item in queue_latency_values]
        self.latency_mode = "within_episode_random" if latency_mode == "within_episode_change" else str(latency_mode)
        self.latency_sampling_mode = str(latency_sampling_mode)
        self.latency_distribution = str(latency_distribution)
        self.latency_normal_mean = float(latency_normal_mean)
        self.latency_normal_std = float(latency_normal_std)
        self.latency_normal_min = int(latency_normal_min)
        self.latency_normal_max = int(latency_normal_max)
        self.latency_uniform_min = int(latency_uniform_min)
        self.latency_uniform_max = int(latency_uniform_max)
        self.episodes_per_stage = max(1, int(episodes_per_stage))
        self.randomize_within_stage = bool(randomize_within_stage)
        self.within_episode_change_interval = max(1, int(within_episode_change_interval))
        self.within_episode_change_prob = float(within_episode_change_prob)
        self.latency_info_keys = tuple(str(item) for item in latency_info_keys if str(item))
        self.add_latency_info = bool(add_latency_info)
        self.rngs = [np.random.RandomState(int(seed)) for seed in seeds]
        self.env_reset_seed_bases = np.asarray([int(seed) for seed in env_reset_seed_bases], dtype=np.int64)
        self.current_latency = np.zeros(self.batch_size, dtype=np.int64)
        self.current_applied_action = np.zeros(self.batch_size, dtype=np.int64)
        self.episode_idx = np.zeros(self.batch_size, dtype=np.int64)
        self.env_step_idx = np.zeros(self.batch_size, dtype=np.int64)
        self.decision_step_idx = np.zeros(self.batch_size, dtype=np.int64)
        self.pending_actions: list[list[tuple[int, int, int, int]]] = [[] for _ in range(self.batch_size)]
        self.schedule_seq = np.zeros(self.batch_size, dtype=np.int64)
        self._decision_queued_actions = np.zeros(self.batch_size, dtype=np.int64)
        self._decision_scheduled_apply_steps = np.zeros(self.batch_size, dtype=np.int64)
        self._decision_scheduled_latencies = np.zeros(self.batch_size, dtype=np.int64)

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
            self.current_applied_action[agent_idx] = 0
            self.env_step_idx[agent_idx] = 0
            self.decision_step_idx[agent_idx] = 0
            self.pending_actions[agent_idx].clear()
            self.schedule_seq[agent_idx] = 0
            self.current_latency[agent_idx] = (
                int(self.queue_latency_values[0])
                if self.latency_mode == "within_episode_curriculum"
                else self._sample_latency(agent_idx)
            )
            seed = seed_values[offset]
            if seed is None:
                seed = int(self.env_reset_seed_bases[agent_idx] + self.episode_idx[agent_idx])
            reset_seeds.append(int(seed))
            infos.append(self._latency_info(agent_idx, int(self.current_latency[agent_idx])))
            self.episode_idx[agent_idx] += 1
        return reset_seeds, infos

    def begin_decision(self, actions: Sequence[int] | np.ndarray, active_mask: Sequence[bool] | np.ndarray) -> None:
        action_arr = np.asarray(actions, dtype=np.int64).reshape(-1)
        active = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        for agent_idx in np.flatnonzero(active):
            self._maybe_update_latency_for_decision(int(agent_idx))
            latency = int(self.current_latency[agent_idx])
            apply_step = int(self.env_step_idx[agent_idx]) + latency
            self.schedule_seq[agent_idx] += 1
            action = int(action_arr[agent_idx])
            insort_right(
                self.pending_actions[agent_idx],
                (apply_step, int(self.schedule_seq[agent_idx]), action, latency),
            )
            self._decision_queued_actions[agent_idx] = action
            self._decision_scheduled_apply_steps[agent_idx] = apply_step
            self._decision_scheduled_latencies[agent_idx] = latency

    def raw_actions(self, active_mask: Sequence[bool] | np.ndarray) -> np.ndarray:
        active = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        for agent_idx in np.flatnonzero(active):
            self._advance_pending_actions(int(agent_idx))
        return self.current_applied_action.copy()

    def record_step(self, active_mask: Sequence[bool] | np.ndarray) -> None:
        active = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        self.env_step_idx[active] += 1

    def finish_decision(self, active_mask: Sequence[bool] | np.ndarray) -> QueueLatencyDecision:
        active = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        decision = QueueLatencyDecision(
            queued_actions=self._decision_queued_actions.copy(),
            applied_actions=self.current_applied_action.copy(),
            scheduled_apply_steps=self._decision_scheduled_apply_steps.copy(),
            scheduled_latencies=self._decision_scheduled_latencies.copy(),
            pending_counts=np.asarray([len(items) for items in self.pending_actions], dtype=np.int64),
        )
        self.decision_step_idx[active] += 1
        return decision

    def decision_info_for_agent(self, agent_idx: int, decision: QueueLatencyDecision) -> dict[str, Any]:
        latency = int(decision.scheduled_latencies[agent_idx])
        info = self._latency_info(agent_idx, latency)
        info["queued_action"] = int(decision.queued_actions[agent_idx])
        info["scheduled_apply_step"] = int(decision.scheduled_apply_steps[agent_idx])
        info["applied_action"] = int(decision.applied_actions[agent_idx])
        info["pending_queue_len"] = int(decision.pending_counts[agent_idx])
        return info

    def _sample_latency(self, agent_idx: int) -> int:
        rng = self.rngs[agent_idx]
        if self.latency_distribution == "normal":
            sampled = (
                self.latency_normal_mean
                if self.latency_normal_std == 0.0
                else float(rng.normal(self.latency_normal_mean, self.latency_normal_std))
            )
            rounded = int(np.rint(sampled))
            return int(min(max(rounded, self.latency_normal_min), self.latency_normal_max))
        if self.latency_distribution == "uniform":
            return int(rng.randint(self.latency_uniform_min, self.latency_uniform_max + 1))
        if self.latency_sampling_mode == "uniform_all":
            return int(rng.choice(self.queue_latency_values))
        stage = int(self.episode_idx[agent_idx]) // self.episodes_per_stage
        stage = min(stage, len(self.queue_latency_values) - 1)
        if self.randomize_within_stage:
            return int(rng.choice(self.queue_latency_values[: stage + 1]))
        return int(self.queue_latency_values[stage])

    def _maybe_update_latency_for_decision(self, agent_idx: int) -> None:
        if self.latency_mode == "episode_fixed":
            return
        if self.latency_mode == "within_episode_random":
            if int(self.decision_step_idx[agent_idx]) == 0:
                return
            if int(self.decision_step_idx[agent_idx]) % self.within_episode_change_interval != 0:
                return
            if self.rngs[agent_idx].rand() <= self.within_episode_change_prob:
                self.current_latency[agent_idx] = self._sample_latency(agent_idx)
            return
        stage = int(self.decision_step_idx[agent_idx]) // self.within_episode_change_interval
        stage = min(stage, len(self.queue_latency_values) - 1)
        self.current_latency[agent_idx] = int(self.queue_latency_values[stage])

    def _advance_pending_actions(self, agent_idx: int) -> None:
        pending = self.pending_actions[agent_idx]
        while pending and pending[0][0] <= int(self.env_step_idx[agent_idx]):
            _apply_step, _seq, action, _latency = pending.pop(0)
            self.current_applied_action[agent_idx] = int(action)

    def _latency_info(self, agent_idx: int, latency_value: int) -> dict[str, Any]:
        info = {
            "latency_mode": self.latency_mode,
            "episode_idx": int(self.episode_idx[agent_idx]),
            "decision_step_idx": int(self.decision_step_idx[agent_idx]),
            "env_step_idx": int(self.env_step_idx[agent_idx]),
            "applied_action": int(self.current_applied_action[agent_idx]),
            "pending_queue_len": int(len(self.pending_actions[agent_idx])),
        }
        for key in self.latency_info_keys:
            info[key] = int(latency_value)
        return info

    def _indices(self, indices: Sequence[int] | np.ndarray | None) -> list[int]:
        if indices is None:
            return list(range(self.batch_size))
        return [int(item) for item in np.asarray(indices).reshape(-1).tolist()]

    @staticmethod
    def _seed_values(seeds: Sequence[int | None] | int | None, count: int) -> list[int | None]:
        if seeds is None:
            return [None] * count
        if isinstance(seeds, (int, np.integer)):
            return [int(seeds)] * count
        return [None if item is None else int(item) for item in list(seeds)]

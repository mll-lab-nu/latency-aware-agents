from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from latency_bench.envs.atari import TRUE_EPISODE_END_INFO_KEY, is_true_atari_episode_end
from training.common.episode_metrics import append_episode_metrics_event
from training.common.utils import batch_indices


class BatchedEpisodeMetrics:
    def __init__(
        self,
        *,
        batch_size: int,
        frames_per_decision: int,
        metrics_path: str,
        worker_index: int,
        vector_index: int,
        env_ids: Sequence[int],
        issued_action_labels: dict[Any, str] | None = None,
        include_score: bool = True,
    ) -> None:
        self.batch_size = int(batch_size)
        self.frames_per_decision = frames_per_decision
        self.metrics_path = metrics_path or ""
        self.worker_index = int(worker_index)
        self.vector_index = int(vector_index)
        self.env_ids = np.asarray([int(item) for item in env_ids], dtype=np.int64)
        self.issued_action_labels = dict(issued_action_labels or {})
        self.include_score = bool(include_score)
        self.full_game_return = np.zeros(self.batch_size, dtype=np.float64)
        self.full_game_raw_return = np.zeros(self.batch_size, dtype=np.float64)
        self.full_game_length = np.zeros(self.batch_size, dtype=np.int64)
        self.submitted_observation_count = np.zeros(self.batch_size, dtype=np.int64)
        self.dropped_observation_count = np.zeros(self.batch_size, dtype=np.int64)
        self.episode_idx = np.zeros(self.batch_size, dtype=np.int64)
        self._preserve_full_game_on_next_reset = np.zeros(self.batch_size, dtype=np.bool_)
        self._action_labels = sorted({str(label) for label in self.issued_action_labels.values()})
        self._action_label_to_idx = {label: idx for idx, label in enumerate(self._action_labels)}
        self._action_value_to_label_idx = {
            int(action): self._action_label_to_idx[str(label)]
            for action, label in self.issued_action_labels.items()
        }
        label_count = len(self._action_labels)
        self.issued_action_counts = np.zeros((self.batch_size, label_count), dtype=np.int64)
        self.issued_action_run_sums = np.zeros((self.batch_size, label_count), dtype=np.int64)
        self.issued_action_run_counts = np.zeros((self.batch_size, label_count), dtype=np.int64)
        self.issued_action_run_max = np.zeros((self.batch_size, label_count), dtype=np.int64)
        self._last_issued_action_label_idx = np.full(self.batch_size, -1, dtype=np.int64)
        self._last_issued_action_run_len = np.zeros(self.batch_size, dtype=np.int64)

    def reset(self, indices: Sequence[int] | np.ndarray | None = None) -> None:
        for agent_idx in self._indices(indices):
            if not bool(self._preserve_full_game_on_next_reset[agent_idx]):
                self._reset_full_game_counters(agent_idx)
            self._preserve_full_game_on_next_reset[agent_idx] = False

    def update(
        self,
        *,
        issued_actions: Sequence[int] | np.ndarray,
        rewards: Sequence[float] | np.ndarray,
        terminated: Sequence[bool] | np.ndarray,
        truncated: Sequence[bool] | np.ndarray,
        scores: Sequence[int] | np.ndarray,
        active_mask: Sequence[bool] | np.ndarray,
        latency_raw_frames: Sequence[int] | np.ndarray,
        command_admitted: Sequence[bool] | np.ndarray,
        lives: Sequence[int] | np.ndarray | None = None,
        timeout_mask: Sequence[bool] | np.ndarray | None = None,
    ) -> dict[int, dict[str, Any]]:
        actions = np.asarray(issued_actions, dtype=np.int64).reshape(-1)
        reward_arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
        term_arr = np.asarray(terminated, dtype=np.bool_).reshape(-1)
        trunc_arr = np.asarray(truncated, dtype=np.bool_).reshape(-1)
        score_arr = np.asarray(scores, dtype=np.int64).reshape(-1)
        active = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        latency_arr = np.asarray(latency_raw_frames, dtype=np.int64).reshape(-1)
        admitted_arr = np.asarray(command_admitted, dtype=np.bool_).reshape(-1)
        lives_arr = None if lives is None else np.asarray(lives, dtype=np.int64).reshape(-1)
        timeout_arr = (
            np.zeros(self.batch_size, dtype=np.bool_)
            if timeout_mask is None
            else np.asarray(timeout_mask, dtype=np.bool_).reshape(-1)
        )
        self._record_issued_actions(actions, active)
        self.full_game_return[active] += reward_arr[active]
        self.full_game_raw_return[active] += reward_arr[active]
        self.full_game_length[active] += 1
        self.submitted_observation_count[active & admitted_arr] += 1
        self.dropped_observation_count[active & ~admitted_arr] += 1

        done_mask = active & np.logical_or(term_arr, trunc_arr)
        if not np.any(done_mask):
            return {}

        out: dict[int, dict[str, Any]] = {}
        for agent_idx in np.flatnonzero(done_mask):
            info: dict[str, Any] = {"raw_reward": float(reward_arr[agent_idx])}
            if self.include_score:
                info["score"] = int(score_arr[agent_idx])
            if lives_arr is not None:
                info["lives"] = int(lives_arr[agent_idx])
            if bool(timeout_arr[agent_idx]):
                info["TimeLimit.truncated"] = True
            done = True
            true_episode_end = is_true_atari_episode_end(bool(term_arr[agent_idx]), bool(trunc_arr[agent_idx]), info)
            if done:
                info[TRUE_EPISODE_END_INFO_KEY] = bool(true_episode_end)
            if done and true_episode_end:
                self._attach_episode_stats(
                    agent_idx=int(agent_idx),
                    info=info,
                    terminated=bool(term_arr[agent_idx]),
                    truncated=bool(trunc_arr[agent_idx]),
                    latency_raw_frames=int(latency_arr[agent_idx]),
                )
                self.episode_idx[agent_idx] += 1
                self._reset_full_game_counters(int(agent_idx))
                self._preserve_full_game_on_next_reset[agent_idx] = False
            elif done:
                self._preserve_full_game_on_next_reset[agent_idx] = True
            out[int(agent_idx)] = info
        return out

    def _attach_episode_stats(
        self,
        *,
        agent_idx: int,
        info: dict[str, Any],
        terminated: bool,
        truncated: bool,
        latency_raw_frames: int,
    ) -> None:
        extra_stats = dict(info.get("episode_extra_stats") or {})
        length = int(self.full_game_length[agent_idx])
        extra_stats["episode/return"] = float(self.full_game_return[agent_idx])
        extra_stats["episode/raw_return"] = float(self.full_game_raw_return[agent_idx])
        extra_stats["episode/length"] = int(length)
        extra_stats["episode/raw_frames"] = int(length * self.frames_per_decision)
        extra_stats["episode/index"] = int(self.episode_idx[agent_idx])
        extra_stats["episode/terminated"] = float(bool(terminated))
        extra_stats["episode/truncated"] = float(bool(truncated))
        timeout = bool(truncated or info.get("TimeLimit.truncated", False))
        extra_stats["episode/timeout"] = float(timeout)
        extra_stats["episode/return_per_step"] = (
            float(self.full_game_return[agent_idx]) / float(length) if length > 0 else 0.0
        )
        if info.get("score") is not None:
            extra_stats["episode/score"] = float(info["score"])
        if info.get("pipes_passed") is not None:
            extra_stats["episode/pipes_passed"] = float(info["pipes_passed"])
        action_stats = self._issued_action_stats(agent_idx)
        extra_stats.update(action_stats)
        extra_stats["episode/latency_raw_frames"] = int(latency_raw_frames)
        submitted = int(self.submitted_observation_count[agent_idx])
        dropped = int(self.dropped_observation_count[agent_idx])
        total_observations = submitted + dropped
        extra_stats["episode/submitted_observation_count"] = submitted
        extra_stats["episode/dropped_observation_count"] = dropped
        extra_stats["episode/observation_drop_ratio"] = float(dropped) / float(total_observations)

        info["episode_extra_stats"] = extra_stats
        info["episodic_raw_return"] = float(self.full_game_raw_return[agent_idx])
        info["episodic_return"] = float(self.full_game_return[agent_idx])
        info["episodic_length"] = int(length)
        done_reason = "terminated" if bool(terminated) else "truncated" if bool(truncated) else "unknown"
        time_limit_truncated = bool(info.get("TimeLimit.truncated", truncated))
        event = {
            "episode_idx": int(self.episode_idx[agent_idx]),
            "return": float(self.full_game_return[agent_idx]),
            "raw_return": float(self.full_game_raw_return[agent_idx]),
            "length": int(length),
            "raw_frames": int(length * self.frames_per_decision),
            "latency_raw_frames": int(latency_raw_frames),
            "worker_index": int(self.worker_index),
            "vector_index": int(self.vector_index),
            "env_id": int(self.env_ids[agent_idx]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "time_limit_truncated": time_limit_truncated,
            "timeout": bool(truncated or time_limit_truncated),
            "done_reason": done_reason,
            "submitted_observation_count": submitted,
            "dropped_observation_count": dropped,
        }
        if info.get("score") is not None:
            event["score"] = float(info["score"])
        if info.get("pipes_passed") is not None:
            event["pipes_passed"] = float(info["pipes_passed"])
        event.update(action_stats)
        append_episode_metrics_event(self.metrics_path, event)

    def _reset_full_game_counters(self, agent_idx: int) -> None:
        self.full_game_return[agent_idx] = 0.0
        self.full_game_raw_return[agent_idx] = 0.0
        self.full_game_length[agent_idx] = 0
        self.submitted_observation_count[agent_idx] = 0
        self.dropped_observation_count[agent_idx] = 0
        self.issued_action_counts[agent_idx, :] = 0
        self.issued_action_run_sums[agent_idx, :] = 0
        self.issued_action_run_counts[agent_idx, :] = 0
        self.issued_action_run_max[agent_idx, :] = 0
        self._last_issued_action_label_idx[agent_idx] = -1
        self._last_issued_action_run_len[agent_idx] = 0

    def _record_issued_actions(self, actions: np.ndarray, active: np.ndarray) -> None:
        if not self._action_value_to_label_idx:
            return
        label_indices = np.full(self.batch_size, -1, dtype=np.int64)
        for action, label_idx in self._action_value_to_label_idx.items():
            label_indices[actions == int(action)] = int(label_idx)
        valid = active & (label_indices >= 0)
        rows = np.flatnonzero(valid)
        if rows.size == 0:
            return
        self.issued_action_counts[rows, label_indices[rows]] += 1

        same_run = valid & (self._last_issued_action_label_idx == label_indices)
        self._last_issued_action_run_len[same_run] += 1

        new_run = valid & ~same_run
        self._finalize_runs(new_run & (self._last_issued_action_label_idx >= 0))
        self._last_issued_action_label_idx[new_run] = label_indices[new_run]
        self._last_issued_action_run_len[new_run] = 1

    def _finalize_runs(self, mask: np.ndarray) -> None:
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            return
        labels = self._last_issued_action_label_idx[rows]
        lengths = self._last_issued_action_run_len[rows]
        self.issued_action_run_sums[rows, labels] += lengths
        self.issued_action_run_counts[rows, labels] += 1
        self.issued_action_run_max[rows, labels] = np.maximum(self.issued_action_run_max[rows, labels], lengths)

    def _issued_action_stats(self, agent_idx: int) -> dict[str, float]:
        if self._last_issued_action_label_idx[agent_idx] >= 0 and self._last_issued_action_run_len[agent_idx] > 0:
            mask = np.zeros(self.batch_size, dtype=np.bool_)
            mask[agent_idx] = True
            self._finalize_runs(mask)
            self._last_issued_action_label_idx[agent_idx] = -1
            self._last_issued_action_run_len[agent_idx] = 0

        total = int(np.sum(self.issued_action_counts[agent_idx]))
        if total <= 0:
            return {}

        stats: dict[str, float] = {}
        entropy = 0.0
        for label_idx, label in enumerate(self._action_labels):
            count = int(self.issued_action_counts[agent_idx, label_idx])
            ratio = float(count) / float(total)
            stats[f"action/issued_{label}_ratio"] = ratio
            if ratio > 0.0:
                entropy -= ratio * math.log(ratio)
            run_count = int(self.issued_action_run_counts[agent_idx, label_idx])
            run_sum = int(self.issued_action_run_sums[agent_idx, label_idx])
            stats[f"action/issued_consecutive_{label}_mean"] = (
                float(run_sum) / float(run_count) if run_count > 0 else 0.0
            )
            stats[f"action/issued_consecutive_{label}_max"] = float(self.issued_action_run_max[agent_idx, label_idx])
        stats["action/issued_entropy"] = float(entropy)
        return stats

    def _indices(self, indices: Sequence[int] | np.ndarray | None) -> list[int]:
        return batch_indices(indices, batch_size=self.batch_size)

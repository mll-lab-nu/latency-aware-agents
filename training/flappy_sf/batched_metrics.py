from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from training.atari_episode_stats import TRUE_EPISODE_END_INFO_KEY, is_true_atari_episode_end
from training.sample_factory_common.episode_metrics import append_episode_metrics_event


class BatchedFlappyEpisodeMetrics:
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
    ) -> None:
        self.batch_size = int(batch_size)
        self.frames_per_decision = max(1, int(frames_per_decision))
        self.metrics_path = metrics_path or ""
        self.worker_index = int(worker_index)
        self.vector_index = int(vector_index)
        self.env_ids = np.asarray([int(item) for item in env_ids], dtype=np.int64)
        self.issued_action_labels = dict(issued_action_labels or {})
        self.full_game_return = np.zeros(self.batch_size, dtype=np.float64)
        self.full_game_raw_return = np.zeros(self.batch_size, dtype=np.float64)
        self.full_game_length = np.zeros(self.batch_size, dtype=np.int64)
        self.episode_idx = np.zeros(self.batch_size, dtype=np.int64)
        self._preserve_full_game_on_next_reset = np.zeros(self.batch_size, dtype=np.bool_)
        self.issued_action_counts: list[defaultdict[str, int]] = [defaultdict(int) for _ in range(self.batch_size)]
        self.issued_action_runs: list[defaultdict[str, list[int]]] = [defaultdict(list) for _ in range(self.batch_size)]
        self._last_issued_action_label: list[str | None] = [None for _ in range(self.batch_size)]
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
        infos: list[dict[str, Any]],
        active_mask: Sequence[bool] | np.ndarray,
        latency_raw_frames: Sequence[int] | np.ndarray,
    ) -> list[dict[str, Any]]:
        actions = np.asarray(issued_actions, dtype=np.int64).reshape(-1)
        reward_arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
        term_arr = np.asarray(terminated, dtype=np.bool_).reshape(-1)
        trunc_arr = np.asarray(truncated, dtype=np.bool_).reshape(-1)
        active = np.asarray(active_mask, dtype=np.bool_).reshape(-1)
        latency_arr = np.asarray(latency_raw_frames, dtype=np.int64).reshape(-1)
        out = [dict(info or {}) for info in infos]
        for agent_idx in np.flatnonzero(active):
            self._record_issued_action(int(agent_idx), int(actions[agent_idx]))
            reward_value = float(reward_arr[agent_idx])
            info = dict(out[agent_idx])
            raw_reward = float(info.get("raw_reward", reward_value))
            info["raw_reward"] = float(raw_reward)
            self.full_game_return[agent_idx] += raw_reward
            self.full_game_raw_return[agent_idx] += raw_reward
            self.full_game_length[agent_idx] += 1
            done = bool(term_arr[agent_idx] or trunc_arr[agent_idx])
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
            out[agent_idx] = info
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
        self.issued_action_counts[agent_idx].clear()
        self.issued_action_runs[agent_idx].clear()
        self._last_issued_action_label[agent_idx] = None
        self._last_issued_action_run_len[agent_idx] = 0

    def _record_issued_action(self, agent_idx: int, action: int) -> None:
        if not self.issued_action_labels:
            return
        label = self.issued_action_labels.get(int(action))
        if label is None:
            return
        label = str(label)
        self.issued_action_counts[agent_idx][label] += 1
        if label == self._last_issued_action_label[agent_idx]:
            self._last_issued_action_run_len[agent_idx] += 1
            return
        if self._last_issued_action_label[agent_idx] is not None:
            self.issued_action_runs[agent_idx][self._last_issued_action_label[agent_idx]].append(
                int(self._last_issued_action_run_len[agent_idx])
            )
        self._last_issued_action_label[agent_idx] = label
        self._last_issued_action_run_len[agent_idx] = 1

    def _issued_action_stats(self, agent_idx: int) -> dict[str, float]:
        if self._last_issued_action_label[agent_idx] is not None and self._last_issued_action_run_len[agent_idx] > 0:
            self.issued_action_runs[agent_idx][self._last_issued_action_label[agent_idx]].append(
                int(self._last_issued_action_run_len[agent_idx])
            )
            self._last_issued_action_label[agent_idx] = None
            self._last_issued_action_run_len[agent_idx] = 0

        total = sum(self.issued_action_counts[agent_idx].values())
        if total <= 0:
            return {}

        stats: dict[str, float] = {}
        entropy = 0.0
        labels = sorted({str(label) for label in self.issued_action_labels.values()})
        for label in labels:
            count = int(self.issued_action_counts[agent_idx].get(label, 0))
            ratio = float(count) / float(total)
            stats[f"action/issued_{label}_ratio"] = ratio
            if ratio > 0.0:
                entropy -= ratio * math.log(ratio)
            runs = self.issued_action_runs[agent_idx].get(label, [])
            stats[f"action/issued_consecutive_{label}_mean"] = float(np.mean(runs)) if runs else 0.0
            stats[f"action/issued_consecutive_{label}_max"] = float(max(runs)) if runs else 0.0
        stats["action/issued_entropy"] = float(entropy)
        return stats

    def _indices(self, indices: Sequence[int] | np.ndarray | None) -> list[int]:
        if indices is None:
            return list(range(self.batch_size))
        return [int(item) for item in np.asarray(indices).reshape(-1).tolist()]

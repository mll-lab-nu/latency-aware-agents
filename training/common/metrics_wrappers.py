"""Reward accounting and episode statistics for reference teacher training."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np

from latency_bench.envs.atari import TRUE_EPISODE_END_INFO_KEY, is_true_atari_episode_end
from training.common.episode_metrics import append_episode_metrics_event

_HURDLE_EPISODE_METRICS = (
    "first_hurdle_clean_pass",
    "hurdles_passed",
    "hurdle_contacts",
    "falls_or_nonfoot_ground_contact",
    "course_complete",
    "teacher_accepted",
    "residual_abs_mean",
    "residual_abs_max",
    "residual_saturation_fraction",
)


class RewardTraceWrapper(gym.Wrapper):
    """Preserve raw rewards separately from learner reward transforms."""

    def __init__(self, env: gym.Env, *, raw_reward_fn=None, learner_reward_fn=None):
        super().__init__(env)
        self.raw_reward_fn = raw_reward_fn
        self.learner_reward_fn = learner_reward_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        env_reward = float(reward)
        info = dict(info)
        raw_reward = info.get("raw_reward")
        if raw_reward is None:
            if self.raw_reward_fn is not None:
                raw_reward = self.raw_reward_fn(env_reward, info)
            else:
                raw_reward = env_reward
        raw_reward = float(raw_reward)
        learner_reward = (
            float(self.learner_reward_fn(raw_reward, info)) if self.learner_reward_fn is not None else env_reward
        )
        info["raw_reward"] = float(raw_reward)
        return obs, learner_reward, terminated, truncated, info


class EpisodeMetricsWrapper(gym.Wrapper):
    """Accurate per-episode metrics at the outer decision-step interface."""

    def __init__(
        self,
        env: gym.Env,
        *,
        frames_per_decision: int = 1,
        metrics_path: str | None = None,
        worker_index: int = 0,
        vector_index: int = 0,
        env_id: int = 0,
        issued_action_labels: dict[Any, str] | None = None,
    ):
        super().__init__(env)
        self.frames_per_decision = frames_per_decision
        self.metrics_path = metrics_path or ""
        self.worker_index = int(worker_index)
        self.vector_index = int(vector_index)
        self.env_id = int(env_id)
        self.issued_action_labels = dict(issued_action_labels or {})
        self.full_game_return = 0.0
        self.full_game_raw_return = 0.0
        self.full_game_length = 0
        self.full_game_raw_frames = 0
        self.submitted_observation_count = 0
        self.dropped_observation_count = 0
        self._has_chunk_trace = (
            isinstance(env.observation_space, gym.spaces.Dict) and "replay_length" in env.observation_space.spaces
        )
        self.issued_action_counts: dict[str, int] = defaultdict(int)
        self.issued_action_runs: dict[str, list[int]] = defaultdict(list)
        self._last_issued_action_label: str | None = None
        self._last_issued_action_run_len = 0
        self.episode_idx = 0
        self._preserve_full_game_on_next_reset = False

    def _reset_full_game_counters(self) -> None:
        self.full_game_return = 0.0
        self.full_game_raw_return = 0.0
        self.full_game_length = 0
        self.full_game_raw_frames = 0
        self.submitted_observation_count = 0
        self.dropped_observation_count = 0
        self.issued_action_counts.clear()
        self.issued_action_runs.clear()
        self._last_issued_action_label = None
        self._last_issued_action_run_len = 0

    def _action_key(self, action: Any) -> Any:
        if isinstance(action, np.ndarray):
            if action.size == 1:
                return self._action_key(action.reshape(-1)[0].item())
            return tuple(self._action_key(item) for item in action.reshape(-1).tolist())
        if isinstance(action, np.generic):
            return action.item()
        if isinstance(action, (list, tuple)) and len(action) == 1:
            return self._action_key(action[0])
        return action

    def _record_issued_action(self, action: Any) -> None:
        if not self.issued_action_labels:
            return
        label = self.issued_action_labels.get(self._action_key(action))
        if label is None:
            return
        label = str(label)
        self.issued_action_counts[label] += 1
        if label == self._last_issued_action_label:
            self._last_issued_action_run_len += 1
            return
        if self._last_issued_action_label is not None:
            self.issued_action_runs[self._last_issued_action_label].append(self._last_issued_action_run_len)
        self._last_issued_action_label = label
        self._last_issued_action_run_len = 1

    def _issued_action_stats(self) -> dict[str, float]:
        if self._last_issued_action_label is not None and self._last_issued_action_run_len > 0:
            self.issued_action_runs[self._last_issued_action_label].append(self._last_issued_action_run_len)
            self._last_issued_action_label = None
            self._last_issued_action_run_len = 0

        total = sum(self.issued_action_counts.values())
        if total <= 0:
            return {}

        stats: dict[str, float] = {}
        entropy = 0.0
        labels = sorted({str(label) for label in self.issued_action_labels.values()})
        for label in labels:
            count = int(self.issued_action_counts.get(label, 0))
            ratio = float(count) / float(total)
            stats[f"action/issued_{label}_ratio"] = ratio
            if ratio > 0.0:
                entropy -= ratio * math.log(ratio)
            runs = self.issued_action_runs.get(label, [])
            stats[f"action/issued_consecutive_{label}_mean"] = float(np.mean(runs)) if runs else 0.0
            stats[f"action/issued_consecutive_{label}_max"] = float(max(runs)) if runs else 0.0
        stats["action/issued_entropy"] = float(entropy)
        return stats

    def reset(self, **kwargs):
        if not self._preserve_full_game_on_next_reset:
            self._reset_full_game_counters()
        self._preserve_full_game_on_next_reset = False
        obs, info = self.env.reset(**kwargs)
        return obs, dict(info)

    def step(self, action):
        self._record_issued_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_value = float(reward)
        info = dict(info)
        if info["command_admitted"]:
            self.submitted_observation_count += 1
        else:
            self.dropped_observation_count += 1
        raw_reward = float(info.get("raw_reward", reward_value))
        self.full_game_return += raw_reward
        self.full_game_raw_return += raw_reward
        self.full_game_length += 1
        self.full_game_raw_frames += (
            obs["replay_length"] if self._has_chunk_trace
            else self.frames_per_decision
        )
        done = bool(terminated or truncated)
        true_episode_end = is_true_atari_episode_end(terminated, truncated, info)
        if done:
            info[TRUE_EPISODE_END_INFO_KEY] = bool(true_episode_end)
        if done and true_episode_end:
            extra_stats = dict(info.get("episode_extra_stats") or {})
            extra_stats["episode/return"] = float(self.full_game_return)
            extra_stats["episode/raw_return"] = float(self.full_game_raw_return)
            extra_stats["episode/length"] = int(self.full_game_length)
            extra_stats["episode/raw_frames"] = int(self.full_game_raw_frames)
            extra_stats["episode/index"] = int(self.episode_idx)
            extra_stats["episode/terminated"] = float(bool(terminated))
            extra_stats["episode/truncated"] = float(bool(truncated))
            timeout = bool(truncated or info.get("TimeLimit.truncated", False))
            extra_stats["episode/timeout"] = float(timeout)
            extra_stats["episode/return_per_step"] = (
                float(self.full_game_return) / float(self.full_game_length) if self.full_game_length > 0 else 0.0
            )
            if info.get("score") is not None:
                extra_stats["episode/score"] = float(info["score"])
            if info.get("pipes_passed") is not None:
                extra_stats["episode/pipes_passed"] = float(info["pipes_passed"])
            for key in _HURDLE_EPISODE_METRICS:
                if key in info:
                    extra_stats[f"episode/{key}"] = float(info[key])
            action_stats = self._issued_action_stats()
            extra_stats.update(action_stats)

            latency_raw_frames = info.get("latency_raw_frames")
            if latency_raw_frames is None:
                try:
                    latency_raw_frames = self.env.get_wrapper_attr("current_latency_raw_frames")
                except (AttributeError, LookupError):
                    latency_raw_frames = None
            if latency_raw_frames is not None:
                extra_stats["episode/latency_raw_frames"] = int(latency_raw_frames)
            total_observations = self.submitted_observation_count + self.dropped_observation_count
            extra_stats["episode/submitted_observation_count"] = int(self.submitted_observation_count)
            extra_stats["episode/dropped_observation_count"] = int(self.dropped_observation_count)
            extra_stats["episode/observation_drop_ratio"] = (
                float(self.dropped_observation_count) / float(total_observations)
            )

            info["episode_extra_stats"] = extra_stats
            info["episodic_raw_return"] = float(self.full_game_raw_return)
            info["episodic_return"] = float(self.full_game_return)
            info["episodic_length"] = int(self.full_game_length)
            done_reason = "terminated" if bool(terminated) else "truncated" if bool(truncated) else "unknown"
            time_limit_truncated = bool(info.get("TimeLimit.truncated", truncated))
            event = {
                "episode_idx": int(self.episode_idx),
                "return": float(self.full_game_return),
                "raw_return": float(self.full_game_raw_return),
                "length": int(self.full_game_length),
                "raw_frames": int(self.full_game_raw_frames),
                "latency_raw_frames": None if latency_raw_frames is None else int(latency_raw_frames),
                "worker_index": int(self.worker_index),
                "vector_index": int(self.vector_index),
                "env_id": int(self.env_id),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "time_limit_truncated": time_limit_truncated,
                "timeout": bool(truncated or time_limit_truncated),
                "done_reason": done_reason,
                "submitted_observation_count": int(self.submitted_observation_count),
                "dropped_observation_count": int(self.dropped_observation_count),
            }
            if info.get("score") is not None:
                event["score"] = float(info["score"])
            if info.get("pipes_passed") is not None:
                event["pipes_passed"] = float(info["pipes_passed"])
            for key in _HURDLE_EPISODE_METRICS:
                if key in info:
                    event[key] = float(info[key])
            event.update(action_stats)
            append_episode_metrics_event(
                self.metrics_path,
                event,
            )
            self.episode_idx += 1
            self._reset_full_game_counters()
            self._preserve_full_game_on_next_reset = False
        elif done:
            self._preserve_full_game_on_next_reset = True
        return obs, reward_value, terminated, truncated, info

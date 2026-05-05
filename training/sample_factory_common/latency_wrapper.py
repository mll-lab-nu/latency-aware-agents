from __future__ import annotations

import copy
import math
from bisect import insort_right
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, Tuple as GymTuple

from training.atari_episode_stats import TRUE_EPISODE_END_INFO_KEY, is_true_atari_episode_end
from latency_bench.core.latency_distribution import (
    LatencyDistributionSpec,
    derive_latency_seed,
    distribution_max_frame_latency,
    latency_distribution_label,
    make_latency_distribution_sampler,
    parse_latency_distribution,
)
from training.sample_factory_common.episode_metrics import append_episode_metrics_event


def noop_action_from_space(space: gym.Space):
    if isinstance(space, Discrete):
        return 0
    if isinstance(space, GymTuple):
        return tuple(noop_action_from_space(subspace) for subspace in space.spaces)
    if isinstance(space, Box):
        return np.zeros(space.shape, dtype=space.dtype)
    raise TypeError(f"Unsupported action space for no-op action: {space}")


def clone_action(action):
    if isinstance(action, np.ndarray):
        return action.copy()
    if isinstance(action, (list, tuple)):
        return type(action)(clone_action(item) for item in action)
    return copy.deepcopy(action)


class QueueLatencyWrapper(gym.Wrapper):
    """True input-delay wrapper with configurable training-time latency schedules."""

    def __init__(
        self,
        env: gym.Env,
        queue_latency_values: Sequence[int],
        *,
        latency_mode: str = "episode_fixed",
        latency_sampling_mode: str = "uniform_all",
        episodes_per_stage: int = 10,
        randomize_within_stage: bool = False,
        within_episode_change_interval: int = 1,
        within_episode_change_prob: float = 1.0,
        seed: int = 0,
        env_reset_seed_base: int | None = None,
        latency_info_keys: Sequence[str] = ("queue_latency",),
        add_latency_info: bool = True,
        latency_distribution_spec: LatencyDistributionSpec | dict[str, Any] | str | None = None,
        max_pending_actions: int | None = None,
    ):
        super().__init__(env)
        if latency_mode == "within_episode_change":
            latency_mode = "within_episode_random"
        if not queue_latency_values:
            raise ValueError("queue_latency_values cannot be empty")
        if latency_mode not in {"episode_fixed", "within_episode_random", "within_episode_curriculum"}:
            raise ValueError(f"Unsupported latency mode: {latency_mode}")
        if latency_sampling_mode not in {"uniform_all", "curriculum"}:
            raise ValueError(f"Unsupported latency sampling mode: {latency_sampling_mode}")

        self.queue_latency_values = [max(0, int(item)) for item in queue_latency_values]
        self.latency_distribution_spec = (
            None if latency_distribution_spec is None else parse_latency_distribution(latency_distribution_spec)
        )
        self.latency_distribution_label = (
            None if self.latency_distribution_spec is None else latency_distribution_label(self.latency_distribution_spec)
        )
        if self.latency_distribution_spec is not None and latency_mode == "within_episode_curriculum":
            raise ValueError("latency.train_distribution does not support within_episode_curriculum")
        self.latency_mode = str(latency_mode)
        self.latency_sampling_mode = str(latency_sampling_mode)
        self.episodes_per_stage = max(1, int(episodes_per_stage))
        self.randomize_within_stage = bool(randomize_within_stage)
        self.within_episode_change_interval = max(1, int(within_episode_change_interval))
        self.within_episode_change_prob = float(within_episode_change_prob)
        if not 0.0 <= self.within_episode_change_prob <= 1.0:
            raise ValueError("latency.within_episode_change_prob must be in [0, 1]")
        self.latency_info_keys = tuple(str(item) for item in latency_info_keys if str(item))
        self.add_latency_info = bool(add_latency_info)
        self.max_frame_latency = (
            distribution_max_frame_latency(self.latency_distribution_spec)
            if self.latency_distribution_spec is not None
            else max(self.queue_latency_values)
        )
        self.max_pending_actions = None if max_pending_actions is None else int(max_pending_actions)
        if self.max_pending_actions is not None and self.max_pending_actions < self.max_frame_latency + 1:
            raise ValueError(
                "latency.max_pending_actions must be at least train_distribution.max + 1 "
                f"({self.max_frame_latency + 1})"
            )

        self.noop_action = noop_action_from_space(env.action_space)
        self.base_latency_seed = int(seed)
        self.rng = np.random.RandomState(seed)
        self.latency_distribution_sampler = None
        self.env_reset_seed_base = None if env_reset_seed_base is None else int(env_reset_seed_base)
        self.current_latency = 0
        self.current_applied_action = clone_action(self.noop_action)
        self.episode_idx = 0
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions: list[tuple[int, int, Any, int]] = []
        self._schedule_seq = 0

    def _get_stage_index(self) -> int:
        stage = self.episode_idx // self.episodes_per_stage
        return min(stage, len(self.queue_latency_values) - 1)

    def _sample_latency(self) -> int:
        if self.latency_distribution_spec is not None:
            if self.latency_distribution_sampler is None:
                self._reset_latency_distribution_sampler()
            latency = int(self.latency_distribution_sampler.sample_frame())
            if latency > self.max_frame_latency:
                raise RuntimeError(
                    f"sampled latency {latency} exceeds configured max_frame_latency {self.max_frame_latency}"
                )
            return latency

        if self.latency_sampling_mode == "uniform_all":
            return int(self.rng.choice(self.queue_latency_values))

        stage = self._get_stage_index()
        if self.randomize_within_stage:
            return int(self.rng.choice(self.queue_latency_values[: stage + 1]))
        return int(self.queue_latency_values[stage])

    def _current_curriculum_latency(self) -> int:
        stage = self.decision_step_idx // self.within_episode_change_interval
        stage = min(stage, len(self.queue_latency_values) - 1)
        return int(self.queue_latency_values[stage])

    def _maybe_update_latency_for_this_decision(self) -> None:
        if self.latency_mode == "episode_fixed":
            return
        if self.latency_mode == "within_episode_random":
            if self.decision_step_idx == 0:
                return
            if self.decision_step_idx % self.within_episode_change_interval != 0:
                return
            if self.rng.rand() <= self.within_episode_change_prob:
                self.current_latency = self._sample_latency()
            return
        self.current_latency = self._current_curriculum_latency()

    def _schedule_action(self, action) -> int:
        latency_for_action = int(self.current_latency)
        apply_step = self.env_step_idx + latency_for_action
        self._schedule_seq += 1
        insort_right(
            self.pending_actions,
            (apply_step, self._schedule_seq, clone_action(action), latency_for_action),
        )
        if self.max_pending_actions is not None and len(self.pending_actions) > self.max_pending_actions:
            raise RuntimeError(
                "pending action queue exceeded latency.max_pending_actions="
                f"{self.max_pending_actions}"
            )
        return apply_step

    def _advance_pending_actions(self) -> None:
        while self.pending_actions and self.pending_actions[0][0] <= self.env_step_idx:
            _, _, matured_action, _ = self.pending_actions.pop(0)
            self.current_applied_action = clone_action(matured_action)

    def _latency_info(self, latency_value: int) -> dict[str, Any]:
        info = {
            "latency_mode": self.latency_mode,
            "episode_idx": int(self.episode_idx),
            "decision_step_idx": int(self.decision_step_idx),
            "env_step_idx": int(self.env_step_idx),
            "applied_action": clone_action(self.current_applied_action),
            "pending_queue_len": int(len(self.pending_actions)),
        }
        for key in self.latency_info_keys:
            info[key] = int(latency_value)
        if self.latency_distribution_spec is not None:
            info["latency_distribution"] = self.latency_distribution_spec.to_dict()
            info["latency_distribution_label"] = str(self.latency_distribution_label)
        return info

    def _reset_latency_distribution_sampler(self) -> None:
        if self.latency_distribution_spec is None:
            self.latency_distribution_sampler = None
            return
        seed = derive_latency_seed(self.base_latency_seed, episode_idx=int(self.episode_idx))
        self.latency_distribution_sampler = make_latency_distribution_sampler(
            self.latency_distribution_spec,
            seed=seed,
        )

    def reset(self, **kwargs):
        self.current_applied_action = clone_action(self.noop_action)
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions.clear()
        self._schedule_seq = 0
        self._reset_latency_distribution_sampler()
        if self.latency_mode == "within_episode_curriculum":
            self.current_latency = int(self.queue_latency_values[0])
        else:
            self.current_latency = self._sample_latency()
        reset_kwargs = dict(kwargs)
        inserted_seed = False
        if reset_kwargs.get("seed") is None and self.env_reset_seed_base is not None:
            reset_kwargs["seed"] = int(self.env_reset_seed_base + self.episode_idx)
            inserted_seed = True
        try:
            obs, info = self.env.reset(**reset_kwargs)
        except TypeError:
            if not inserted_seed:
                raise
            obs, info = self.env.reset(**kwargs)
        info = dict(info or {})
        if self.add_latency_info:
            info.update(self._latency_info(self.current_latency))
        self.episode_idx += 1
        return obs, info

    def step(self, action):
        self._maybe_update_latency_for_this_decision()
        scheduled_delay = int(self.current_latency)
        scheduled_apply_step = self._schedule_action(action)
        self._advance_pending_actions()
        obs, reward, terminated, truncated, info = self.env.step(clone_action(self.current_applied_action))
        info = dict(info or {})
        if self.add_latency_info:
            info.update(self._latency_info(scheduled_delay))
            info["queued_action"] = clone_action(action)
            info["scheduled_apply_step"] = int(scheduled_apply_step)
        self.env_step_idx += 1
        self.decision_step_idx += 1
        return obs, float(reward), terminated, truncated, info


class RewardTraceWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *, raw_reward_fn=None):
        super().__init__(env)
        self.raw_reward_fn = raw_reward_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        transformed_reward = float(reward)
        info = dict(info or {})
        raw_reward = info.get("raw_reward")
        if raw_reward is None:
            if self.raw_reward_fn is not None:
                raw_reward = self.raw_reward_fn(transformed_reward, info)
            else:
                raw_reward = transformed_reward
        info["raw_reward"] = float(raw_reward)
        info["transformed_reward"] = transformed_reward
        return obs, transformed_reward, terminated, truncated, info


class ClipRewardWithRawInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        raw_reward = float(reward)
        transformed_reward = float(np.sign(raw_reward))
        info = dict(info or {})
        info["raw_reward"] = raw_reward
        info["clipped_reward"] = transformed_reward
        info["transformed_reward"] = transformed_reward
        return obs, transformed_reward, terminated, truncated, info


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
        self.frames_per_decision = max(1, int(frames_per_decision))
        self.metrics_path = metrics_path or ""
        self.worker_index = int(worker_index)
        self.vector_index = int(vector_index)
        self.env_id = int(env_id)
        self.issued_action_labels = dict(issued_action_labels or {})
        self.full_game_return = 0.0
        self.full_game_raw_return = 0.0
        self.full_game_clipped_return = 0.0
        self.full_game_length = 0
        self.issued_action_counts: dict[str, int] = defaultdict(int)
        self.issued_action_runs: dict[str, list[int]] = defaultdict(list)
        self._last_issued_action_label: str | None = None
        self._last_issued_action_run_len = 0
        self.episode_idx = 0
        self._preserve_full_game_on_next_reset = False

    def _reset_full_game_counters(self) -> None:
        self.full_game_return = 0.0
        self.full_game_raw_return = 0.0
        self.full_game_clipped_return = 0.0
        self.full_game_length = 0
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
        return obs, dict(info or {})

    def step(self, action):
        self._record_issued_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_value = float(reward)
        info = dict(info or {})
        transformed_reward = float(info.get("transformed_reward", reward_value))
        raw_reward = float(info.get("raw_reward", transformed_reward))
        self.full_game_return += raw_reward
        self.full_game_raw_return += raw_reward
        self.full_game_clipped_return += transformed_reward
        self.full_game_length += 1
        done = bool(terminated or truncated)
        true_episode_end = is_true_atari_episode_end(terminated, truncated, info)
        if done:
            info[TRUE_EPISODE_END_INFO_KEY] = bool(true_episode_end)
        if done and true_episode_end:
            extra_stats = dict(info.get("episode_extra_stats") or {})
            extra_stats["episode/return"] = float(self.full_game_return)
            extra_stats["episode/raw_return"] = float(self.full_game_raw_return)
            extra_stats["episode/clipped_return"] = float(self.full_game_clipped_return)
            extra_stats["episode/length"] = int(self.full_game_length)
            extra_stats["episode/raw_frames"] = int(self.full_game_length * self.frames_per_decision)
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
            action_stats = self._issued_action_stats()
            extra_stats.update(action_stats)

            queue_latency = info.get("queue_latency")
            if queue_latency is None:
                try:
                    queue_latency = self.env.get_wrapper_attr("current_latency")
                except (AttributeError, LookupError):
                    queue_latency = None
            if queue_latency is not None:
                extra_stats["episode/queue_latency"] = int(queue_latency)

            info["episode_extra_stats"] = extra_stats
            info["episodic_raw_return"] = float(self.full_game_raw_return)
            info["episodic_clipped_return"] = float(self.full_game_clipped_return)
            info["episodic_return"] = float(self.full_game_return)
            info["episodic_length"] = int(self.full_game_length)
            done_reason = "terminated" if bool(terminated) else "truncated" if bool(truncated) else "unknown"
            time_limit_truncated = bool(info.get("TimeLimit.truncated", truncated))
            event = {
                "episode_idx": int(self.episode_idx),
                "return": float(self.full_game_return),
                "raw_return": float(self.full_game_raw_return),
                "clipped_return": float(self.full_game_clipped_return),
                "length": int(self.full_game_length),
                "raw_frames": int(self.full_game_length * self.frames_per_decision),
                "queue_latency": None if queue_latency is None else int(queue_latency),
                "worker_index": int(self.worker_index),
                "vector_index": int(self.vector_index),
                "env_id": int(self.env_id),
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

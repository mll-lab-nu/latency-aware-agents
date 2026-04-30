from __future__ import annotations

import copy
from bisect import insort_right
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, Tuple as GymTuple

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
        self.latency_mode = str(latency_mode)
        self.latency_sampling_mode = str(latency_sampling_mode)
        self.episodes_per_stage = max(1, int(episodes_per_stage))
        self.randomize_within_stage = bool(randomize_within_stage)
        self.within_episode_change_interval = max(1, int(within_episode_change_interval))
        self.within_episode_change_prob = float(within_episode_change_prob)
        self.latency_info_keys = tuple(str(item) for item in latency_info_keys if str(item))
        self.add_latency_info = bool(add_latency_info)

        self.noop_action = noop_action_from_space(env.action_space)
        self.rng = np.random.RandomState(seed)
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
        return info

    def reset(self, **kwargs):
        self.current_applied_action = clone_action(self.noop_action)
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self.pending_actions.clear()
        self._schedule_seq = 0
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
    ):
        super().__init__(env)
        self.frames_per_decision = max(1, int(frames_per_decision))
        self.metrics_path = metrics_path or ""
        self.worker_index = int(worker_index)
        self.vector_index = int(vector_index)
        self.env_id = int(env_id)
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_idx = 0

    def reset(self, **kwargs):
        self.episode_return = 0.0
        self.episode_length = 0
        obs, info = self.env.reset(**kwargs)
        return obs, dict(info or {})

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_value = float(reward)
        self.episode_return += reward_value
        self.episode_length += 1
        info = dict(info or {})
        done = bool(terminated or truncated)
        if done:
            extra_stats = dict(info.get("episode_extra_stats") or {})
            extra_stats["episode/return"] = float(self.episode_return)
            extra_stats["episode/length"] = int(self.episode_length)
            extra_stats["episode/raw_frames"] = int(self.episode_length * self.frames_per_decision)
            extra_stats["episode/index"] = int(self.episode_idx)

            queue_latency = getattr(self.env, "current_latency", None)
            if queue_latency is None and "queue_latency" in info:
                queue_latency = info.get("queue_latency")
            if queue_latency is not None:
                extra_stats["episode/queue_latency"] = int(queue_latency)

            info["episode_extra_stats"] = extra_stats
            info["episodic_return"] = float(self.episode_return)
            info["episodic_length"] = int(self.episode_length)
            append_episode_metrics_event(
                self.metrics_path,
                {
                    "episode_idx": int(self.episode_idx),
                    "return": float(self.episode_return),
                    "length": int(self.episode_length),
                    "raw_frames": int(self.episode_length * self.frames_per_decision),
                    "queue_latency": None if queue_latency is None else int(queue_latency),
                    "worker_index": int(self.worker_index),
                    "vector_index": int(self.vector_index),
                    "env_id": int(self.env_id),
                },
            )
            self.episode_idx += 1
        return obs, reward_value, terminated, truncated, info

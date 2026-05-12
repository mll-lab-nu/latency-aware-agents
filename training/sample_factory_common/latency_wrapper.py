from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, Tuple as GymTuple

from latency_bench.core.clock import EnvClock
from latency_bench.core.latency_distribution import (
    LatencyDistributionSpec,
    derive_latency_seed,
    distribution_max_frame_latency,
    latency_distribution_label,
    make_latency_distribution_sampler,
)
from latency_bench.core.types import Action, ActionEvent, ActionHoldPolicy, OrderingPolicy
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.persistence import HoldPolicy, OneFrameThenNoopPolicy
from training.atari_episode_stats import TRUE_EPISODE_END_INFO_KEY, is_true_atari_episode_end
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


class ActionLatencyWrapper(gym.Wrapper):
    """Raw-frame action latency wrapper shared by Sample Factory environments."""

    def __init__(
        self,
        env: gym.Env,
        fixed_latency_raw_frames: int = 0,
        *,
        env_fps: float,
        obs_fps: float,
        hold_policy: str | ActionHoldPolicy = ActionHoldPolicy.ONE_FRAME_THEN_NOOP,
        ordering_policy: str | OrderingPolicy = OrderingPolicy.LATEST_READY,
        seed: int = 0,
        env_reset_seed_base: int | None = None,
        add_latency_info: bool = True,
        latency_distribution_spec: LatencyDistributionSpec | None = None,
        max_pending_actions: int | None = None,
    ):
        super().__init__(env)
        self.fixed_latency_raw_frames = int(fixed_latency_raw_frames)
        self.latency_distribution_spec = latency_distribution_spec
        self.latency_distribution_label = (
            None if self.latency_distribution_spec is None else latency_distribution_label(self.latency_distribution_spec)
        )
        self.add_latency_info = bool(add_latency_info)
        self.max_frame_latency = (
            distribution_max_frame_latency(self.latency_distribution_spec)
            if self.latency_distribution_spec is not None
            else int(self.fixed_latency_raw_frames)
        )
        self.max_pending_actions = None if max_pending_actions is None else int(max_pending_actions)
        if self.max_pending_actions is not None and self.max_pending_actions < self.max_frame_latency + 1:
            raise ValueError(
                "latency.max_pending_actions must be at least train_distribution.max + 1 "
                f"({self.max_frame_latency + 1})"
            )

        self.noop_action = noop_action_from_space(env.action_space)
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
        self.action_scheduler = ActionScheduler(
            clock=self.clock,
            noop_action=Action(value=clone_action(self.noop_action), name="noop", is_noop=True),
            persistence_policy=self._make_persistence_policy(),
            ordering_policy=self.ordering_policy,
        )
        self.base_latency_seed = int(seed)
        self.latency_distribution_sampler = None
        self.env_reset_seed_base = None if env_reset_seed_base is None else int(env_reset_seed_base)
        self.current_latency_raw_frames = 0
        self.current_applied_action = clone_action(self.noop_action)
        self.episode_idx = 0
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self._next_action_id = 0

    def _make_persistence_policy(self):
        if self.hold_policy == ActionHoldPolicy.ONE_FRAME_THEN_NOOP:
            return OneFrameThenNoopPolicy()
        if self.hold_policy == ActionHoldPolicy.HOLD:
            return HoldPolicy(duration_raw_frames=self.clock.obs_stride_raw_frames)
        raise ValueError(f"Unsupported hold policy: {self.hold_policy}")

    def _next_latency_raw_frames(self) -> int:
        if self.latency_distribution_spec is not None:
            if self.latency_distribution_sampler is None:
                self._reset_latency_distribution_sampler()
            return int(self.latency_distribution_sampler.sample_frame())
        return int(self.fixed_latency_raw_frames)

    def _latency_info(self, latency_value: int) -> dict[str, Any]:
        latency_raw_frames = int(latency_value)
        info = {
            "episode_idx": int(self.episode_idx),
            "decision_step_idx": int(self.decision_step_idx),
            "env_step_idx": int(self.env_step_idx),
            "raw_frame": int(self.env_step_idx),
            "applied_action": clone_action(self.current_applied_action),
            "pending_action_count": int(len(self.action_scheduler.get_pending_events())),
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
        self.action_scheduler.reset()
        self.current_applied_action = clone_action(self.noop_action)
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self._next_action_id = 0
        self._reset_latency_distribution_sampler()
        self.current_latency_raw_frames = 0 if self.latency_distribution_spec is not None else int(self.fixed_latency_raw_frames)
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
            info.update(self._latency_info(self.current_latency_raw_frames))
        self.episode_idx += 1
        return obs, info

    def _enqueue_action_event(self, action, latency_raw_frames: int) -> int:
        obs_raw_frame = int(self.env_step_idx)
        ready_raw_frame = obs_raw_frame + int(latency_raw_frames)
        event = ActionEvent(
            action_id=int(self._next_action_id),
            obs_id=int(self.decision_step_idx),
            action=Action(value=clone_action(action), name=str(action), is_noop=False),
            obs_env_step=obs_raw_frame,
            obs_time_ms=self.clock.step_to_time_ms(obs_raw_frame),
            latency_ms=float(latency_raw_frames) * self.clock.frame_ms,
            ready_time_ms=self.clock.step_to_time_ms(ready_raw_frame),
            ready_step_float=float(ready_raw_frame),
            obs_raw_frame=obs_raw_frame,
            latency_raw_frames=int(latency_raw_frames),
            ready_raw_frame=ready_raw_frame,
        )
        self._next_action_id += 1
        self.action_scheduler.enqueue(event)
        if (
            self.max_pending_actions is not None
            and len(self.action_scheduler.get_pending_events()) > self.max_pending_actions
        ):
            raise RuntimeError(
                "pending action queue exceeded latency.max_pending_actions="
                f"{self.max_pending_actions}"
            )
        return ready_raw_frame

    def step(self, action):
        # 1. decide delay for this action
        scheduled_delay = int(self._next_latency_raw_frames())
        self.current_latency_raw_frames = scheduled_delay
        # 2. enqueue this action, instead of execute it directly
        scheduled_ready_raw_frame = self._enqueue_action_event(action, scheduled_delay)

        # 3. step raw env, raw frames
        obs = None
        info: dict[str, Any] = {}
        total_reward = 0.0
        total_raw_reward = 0.0
        saw_raw_reward = False
        terminated = False
        truncated = False
        raw_frames_advanced = 0
        for _ in range(self.clock.obs_stride_raw_frames):
            # for each raw frame step, ask scheduler which action should be taken
            active_action = self.action_scheduler.update(
                current_step=int(self.env_step_idx),
                current_time_ms=self.clock.step_to_time_ms(int(self.env_step_idx)),
            )
            self.current_applied_action = clone_action(active_action.value)
            obs, reward, terminated, truncated, info = self.env.step(clone_action(active_action.value))
            # accumulate rewards for raw frames
            total_reward += float(reward)
            step_info = dict(info or {})
            if "raw_reward" in step_info:
                total_raw_reward += float(step_info["raw_reward"])
                saw_raw_reward = True
            raw_frames_advanced += 1
            self.env_step_idx += 1
            if terminated or truncated:
                break

        if obs is None:
            raise RuntimeError("ActionLatencyWrapper did not advance any raw frames")

        info = dict(info or {})
        if saw_raw_reward:
            info["raw_reward"] = float(total_raw_reward)
        if self.add_latency_info:
            info.update(self._latency_info(scheduled_delay))
            info["queued_action"] = clone_action(action)
            info["ready_raw_frame"] = int(scheduled_ready_raw_frame)
            info["raw_frames_advanced"] = int(raw_frames_advanced)
            info["obs_stride_raw_frames"] = int(self.clock.obs_stride_raw_frames)
        self.decision_step_idx += 1
        return obs, float(total_reward), terminated, truncated, info


class RewardTraceWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *, raw_reward_fn=None, learner_reward_fn=None):
        super().__init__(env)
        self.raw_reward_fn = raw_reward_fn
        self.learner_reward_fn = learner_reward_fn

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        env_reward = float(reward)
        info = dict(info or {})
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
        self.frames_per_decision = max(1, int(frames_per_decision))
        self.metrics_path = metrics_path or ""
        self.worker_index = int(worker_index)
        self.vector_index = int(vector_index)
        self.env_id = int(env_id)
        self.issued_action_labels = dict(issued_action_labels or {})
        self.full_game_return = 0.0
        self.full_game_raw_return = 0.0
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
        raw_reward = float(info.get("raw_reward", reward_value))
        self.full_game_return += raw_reward
        self.full_game_raw_return += raw_reward
        self.full_game_length += 1
        done = bool(terminated or truncated)
        true_episode_end = is_true_atari_episode_end(terminated, truncated, info)
        if done:
            info[TRUE_EPISODE_END_INFO_KEY] = bool(true_episode_end)
        if done and true_episode_end:
            extra_stats = dict(info.get("episode_extra_stats") or {})
            extra_stats["episode/return"] = float(self.full_game_return)
            extra_stats["episode/raw_return"] = float(self.full_game_raw_return)
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

            latency_raw_frames = info.get("latency_raw_frames")
            if latency_raw_frames is None:
                try:
                    latency_raw_frames = self.env.get_wrapper_attr("current_latency_raw_frames")
                except (AttributeError, LookupError):
                    latency_raw_frames = None
            if latency_raw_frames is not None:
                extra_stats["episode/latency_raw_frames"] = int(latency_raw_frames)

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
                "raw_frames": int(self.full_game_length * self.frames_per_decision),
                "latency_raw_frames": None if latency_raw_frames is None else int(latency_raw_frames),
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

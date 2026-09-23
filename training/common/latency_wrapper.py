from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict as GymDict, Discrete, Tuple as GymTuple

from latency_bench.core.clock import EnvClock
from latency_bench.core.decision_action_history import DecisionActionHistory
from latency_bench.core.latency_distribution import (
    LatencyDistributionSpec,
    derive_seed,
    distribution_max_frame_latency,
    latency_distribution_label,
    make_latency_distribution_sampler,
)
from latency_bench.core.types import Action, ActionEvent, ActionHoldPolicy, LatencyRecord, OrderingPolicy
from latency_bench.envs.atari import TRUE_EPISODE_END_INFO_KEY, is_true_atari_episode_end
from latency_bench.executors._simulated_timeline import SimulatedResultTimeline, SimulatedWorkerCapacity
from latency_bench.latency.sample import LatencySample
from latency_bench.latency.samplers import HiddenRegimeSampler, IidInverseCdfSampler, build_episode_sampler_factory
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.factory import build_persistence_policy
from training.common.latency_config import (
    build_profile_sampler,
    build_latency_info,
    profile_sampler_max_latency_raw_frames,
)
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
        sample_distribution_per_episode: bool = False,
        profile_path: str | None = None,
        profile_worker_slot: int | None = None,
        profile_method: str = "iid",
        simulated_worker_capacity: int | None = None,
        max_pending_actions: int | None = None,
        hold_last_chunk_action: bool = False,
        release_results_at_observation_boundary: bool = False,
        action_history_decisions: int | None = None,
    ):
        super().__init__(env)
        self.decision_action_history = (
            DecisionActionHistory(env.action_space, num_envs=1, decisions=action_history_decisions)
            if action_history_decisions is not None else None
        )
        if self.decision_action_history is not None:
            spaces = (
                dict(env.observation_space.spaces)
                if isinstance(env.observation_space, GymDict)
                else {"obs": env.observation_space}
            )
            spaces["transport"] = Box(
                low=0, high=1, shape=(self.decision_action_history.observation_dim,), dtype=np.float32
            )
            self.observation_space = GymDict(spaces)
        self.base_latency_seed = int(seed)
        self.fixed_latency_raw_frames = int(fixed_latency_raw_frames)
        self.latency_distribution_spec = latency_distribution_spec
        self.sample_distribution_per_episode = bool(sample_distribution_per_episode)
        self.profile_path = profile_path
        self.profile_worker_slot = profile_worker_slot
        self.profile_method = profile_method
        self.latency_distribution_label = (
            None if self.latency_distribution_spec is None else latency_distribution_label(self.latency_distribution_spec)
        )
        self.add_latency_info = bool(add_latency_info)
        self.max_frame_latency = (
            distribution_max_frame_latency(self.latency_distribution_spec)
            if self.latency_distribution_spec is not None
            else int(self.fixed_latency_raw_frames)
        )
        self.noop_action = noop_action_from_space(env.action_space)
        self.clock = EnvClock(env_fps=float(env_fps), obs_fps=float(obs_fps))
        self.profile_sampler = None
        self.profile_sampler_factory = None
        if self.profile_path is not None:
            self.profile_sampler = build_profile_sampler(
                self.profile_path,
                method=self.profile_method,
                seed=self.base_latency_seed,
                env_fps=self.clock.env_fps,
                obs_fps=self.clock.obs_fps,
            )
            if self.profile_method == "iid":
                self.profile_sampler_factory = build_episode_sampler_factory(
                    {
                        "method": "iid",
                        "profile_path": self.profile_path,
                        "seed": self.base_latency_seed,
                    },
                    self.clock,
                )
            self.max_frame_latency = profile_sampler_max_latency_raw_frames(
                self.profile_sampler,
                worker_slot=self.profile_worker_slot,
                env_fps=self.clock.env_fps,
            )
        self.max_pending_actions = None if max_pending_actions is None else int(max_pending_actions)
        self.worker_capacity = (
            None
            if simulated_worker_capacity is None
            else SimulatedWorkerCapacity(simulated_worker_capacity, {})
        )
        if self.max_pending_actions is not None and self.max_pending_actions < self.max_frame_latency + 1:
            raise ValueError(
                "latency.max_pending_actions must be at least train_distribution.max + 1 "
                f"({self.max_frame_latency + 1})"
            )
        self.hold_policy = ActionHoldPolicy(hold_policy)
        self.ordering_policy = OrderingPolicy(ordering_policy)
        self.action_scheduler = ActionScheduler(
            clock=self.clock,
            noop_action=Action(value=clone_action(self.noop_action), name="noop", is_noop=True),
            persistence_policy=build_persistence_policy(self.hold_policy),
            ordering_policy=self.ordering_policy,
            hold_last_chunk_action=hold_last_chunk_action,
        )
        self.release_results_at_observation_boundary = release_results_at_observation_boundary
        self.result_timeline = SimulatedResultTimeline(ordering_policy=self.ordering_policy)
        self.latency_distribution_sampler = None
        self.env_reset_seed_base = None if env_reset_seed_base is None else int(env_reset_seed_base)
        self.current_latency_raw_frames = 0
        self.current_applied_action = clone_action(self.noop_action)
        self.episode_idx = -1
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self._next_action_id = 0

    def _next_latency_sample(self, *, worker_slot: int | None = None) -> LatencySample:
        if self.profile_sampler is not None:
            sample = self.profile_sampler.sample(
                {
                    "episode_id": int(self.episode_idx),
                    "worker_slot": self.profile_worker_slot if worker_slot is None else worker_slot,
                }
            )
            return sample
        if self.latency_distribution_spec is not None:
            if self.sample_distribution_per_episode:
                return LatencySample.from_latency_ms(
                    float(self.current_latency_raw_frames) * self.clock.frame_ms,
                    clock=self.clock,
                )
            if self.latency_distribution_sampler is None:
                self._reset_latency_distribution_sampler()
            return LatencySample.from_latency_ms(
                float(self.latency_distribution_sampler.sample_frame()) * self.clock.frame_ms,
                clock=self.clock,
            )
        return LatencySample.from_latency_ms(
            float(self.fixed_latency_raw_frames) * self.clock.frame_ms,
            clock=self.clock,
        )

    def _latency_info(self, latency_value: int) -> dict[str, Any]:
        return build_latency_info(
            episode_idx=self.episode_idx,
            decision_step_idx=self.decision_step_idx,
            env_step_idx=self.env_step_idx,
            applied_action=clone_action(self.current_applied_action),
            pending_action_count=(
                len(self.action_scheduler.get_pending_events())
                + self.result_timeline.pending_event_count
            ),
            latency_raw_frames=int(latency_value),
            clock=self.clock,
            hold_policy=self.hold_policy,
            ordering_policy=self.ordering_policy,
            latency_distribution_spec=self.latency_distribution_spec,
            latency_distribution_label=self.latency_distribution_label,
        )

    def _reset_latency_distribution_sampler(self) -> None:
        if self.latency_distribution_spec is None:
            self.latency_distribution_sampler = None
            return
        seed = derive_seed(self.base_latency_seed, episode_idx=int(self.episode_idx))
        self.latency_distribution_sampler = make_latency_distribution_sampler(
            self.latency_distribution_spec,
            seed=seed,
        )

    def reset(self, **kwargs):
        if self.decision_action_history is not None:
            self.decision_action_history.reset()
        self.episode_idx += 1
        self.action_scheduler.reset()
        self.result_timeline.reset()
        if self.worker_capacity is not None:
            self.worker_capacity.reset()
        self.current_applied_action = clone_action(self.noop_action)
        self.env_step_idx = 0
        self.decision_step_idx = 0
        self._next_action_id = 0
        reset_kwargs = dict(kwargs)
        inserted_seed = False
        if reset_kwargs.get("seed") is None and self.env_reset_seed_base is not None:
            reset_kwargs["seed"] = int(self.env_reset_seed_base + self.episode_idx)
            inserted_seed = True
        if self.profile_sampler_factory is not None:
            episode_seed = reset_kwargs["seed"] if reset_kwargs.get("seed") is not None else self.episode_idx
            self.profile_sampler = self.profile_sampler_factory(episode_seed)
        self._reset_latency_distribution_sampler()
        if self.latency_distribution_spec is None:
            self.current_latency_raw_frames = int(self.fixed_latency_raw_frames)
        elif self.sample_distribution_per_episode:
            self.current_latency_raw_frames = int(self.latency_distribution_sampler.sample_frame())
        else:
            self.current_latency_raw_frames = 0
        try:
            obs, info = self.env.reset(**reset_kwargs)
        except TypeError:
            if not inserted_seed:
                raise
            obs, info = self.env.reset(**kwargs)
        self._raw_observation = obs
        info = dict(info or {})
        if self.add_latency_info:
            info.update(self._latency_info(self.current_latency_raw_frames))
        return self._transport_obs(obs), info

    def _transport_obs(self, obs):
        if self.decision_action_history is not None:
            data = dict(obs) if isinstance(obs, dict) else {"obs": obs}
            data["transport"] = self.decision_action_history.observation()[0]
            return data
        return obs

    def _planned_chunk_window(self, event: ActionEvent) -> tuple[int, int]:
        """Replay-only handoff clock assuming the episode continues.

        Run the existing admission/result/scheduler rules on cloned timing
        state. Physics, actor inputs and the live sampler are never advanced.
        Immutable profile tables are shared; only sampler state is copied.
        """
        shadow = copy.copy(self)
        memo = {}
        if isinstance(self.profile_sampler, (HiddenRegimeSampler, IidInverseCdfSampler)):
            memo[id(self.profile_sampler.distributions)] = self.profile_sampler.distributions
        if isinstance(self.profile_sampler, HiddenRegimeSampler):
            for table in (self.profile_sampler.model, self.profile_sampler.rank_processes, self.profile_sampler.pre_worker_time):
                memo[id(table)] = table
        shadow.profile_sampler = copy.deepcopy(self.profile_sampler, memo)
        shadow.latency_distribution_sampler = copy.deepcopy(self.latency_distribution_sampler)
        shadow.worker_capacity = copy.deepcopy(self.worker_capacity)
        shadow.result_timeline = copy.deepcopy(self.result_timeline)
        shadow.action_scheduler = ActionScheduler(
            self.clock, self.action_scheduler.noop_action,
            copy.deepcopy(self.action_scheduler.persistence_policy), self.ordering_policy,
            hold_last_chunk_action=self.action_scheduler.hold_last_chunk_action,
        )
        for pending in copy.deepcopy(self.action_scheduler.get_pending_events()):
            shadow.action_scheduler.enqueue(pending)
        while True:
            shadow.env_step_idx += 1
            # Boundary arrivals are visible before the next admission, as in step().
            released, _ = shadow.result_timeline.release_arrived(shadow.env_step_idx)
            for pending in released:
                shadow.action_scheduler.enqueue(pending)
            if shadow.worker_capacity is not None:
                shadow.worker_capacity.release_ready(shadow.env_step_idx)
            if shadow.env_step_idx % self.clock.obs_stride_raw_frames == 0:
                shadow.decision_step_idx += 1
                if shadow.worker_capacity is None or shadow.worker_capacity.can_submit():
                    worker = None if shadow.worker_capacity is None else shadow.worker_capacity.assign_worker()
                    sample = shadow._next_latency_sample(worker_slot=worker)
                    shadow._enqueue_action_event(
                        self.noop_action, sample.action_ready_raw_frames,
                        latency_sample=sample, worker_id=worker,
                    )
                    if shadow.worker_capacity is not None:
                        shadow.worker_capacity.submit(worker, shadow.env_step_idx + sample.worker_service_raw_frames)
            released, _ = shadow.result_timeline.release_arrived(shadow.env_step_idx)
            for pending in released:
                shadow.action_scheduler.enqueue(pending)
            shadow.action_scheduler.update(shadow.env_step_idx, self.clock.step_to_time_ms(shadow.env_step_idx))
            if shadow.action_scheduler.applied_event_count:
                return event.apply_raw_frame - event.obs_env_step, shadow.env_step_idx - event.obs_env_step

    def _enqueue_action_event(
        self,
        action,
        latency_raw_frames: int,
        *,
        latency_sample: LatencySample,
        worker_id: int | None,
    ) -> int:
        obs_env_step = int(self.env_step_idx)
        ready_raw_frame = obs_env_step + int(latency_raw_frames)
        action_chunk = clone_action(action) if isinstance(action, np.ndarray) and action.ndim == 2 else None
        event_action = action_chunk[0] if action_chunk is not None else action
        event = ActionEvent(
            action_id=int(self._next_action_id),
            action=Action(value=clone_action(event_action), is_noop=False),
            obs_env_step=obs_env_step,
            ready_time_ms=self.clock.step_to_time_ms(ready_raw_frame),
            ready_step_float=float(ready_raw_frame),
            latency=LatencyRecord(
                obs_id=int(self.decision_step_idx),
                latency_ms=float(latency_raw_frames) * self.clock.frame_ms,
            ),
            latency_raw_frames=int(latency_raw_frames),
            ready_raw_frame=ready_raw_frame,
            action_chunk=action_chunk,
            metadata={
                "worker_id": worker_id,
                "worker_service_ms": latency_sample.worker_service_ms,
                "worker_service_raw_frames": latency_sample.worker_service_raw_frames,
            },
        )
        self._next_action_id += 1
        self.result_timeline.submit(
            obs_id=event.latency.obs_id,
            ready_raw_frame=ready_raw_frame,
            event=event,
        )
        if (
            self.max_pending_actions is not None
            and (
                len(self.action_scheduler.get_pending_events())
                + self.result_timeline.pending_event_count
            ) > self.max_pending_actions
        ):
            raise RuntimeError(
                "pending action queue exceeded latency.max_pending_actions="
                f"{self.max_pending_actions}"
            )
        return ready_raw_frame

    def step(self, action):
        # 1. admit the observation to the inference worker, then enqueue it.
        observation_dropped = False
        worker_id = None
        latency_sample = None
        if self.worker_capacity is not None:
            self.worker_capacity.release_ready(self.env_step_idx)
            if not self.worker_capacity.can_submit():
                observation_dropped = True
            else:
                worker_id = self.worker_capacity.assign_worker()
        if not observation_dropped:
            latency_sample = self._next_latency_sample(worker_slot=worker_id)
            scheduled_delay = latency_sample.action_ready_raw_frames
            self.current_latency_raw_frames = scheduled_delay
            scheduled_ready_raw_frame = self._enqueue_action_event(
                action,
                scheduled_delay,
                latency_sample=latency_sample,
                worker_id=worker_id,
            )
            if self.worker_capacity is not None:
                self.worker_capacity.submit(
                    worker_id,
                    self.env_step_idx + latency_sample.worker_service_raw_frames,
                )
        else:
            scheduled_delay = 0
            self.current_latency_raw_frames = 0
            scheduled_ready_raw_frame = None

        # 3. step raw env, raw frames
        obs = None
        info: dict[str, Any] = {}
        total_reward = 0.0
        total_raw_reward = 0.0
        saw_raw_reward = False
        terminated = False
        truncated = False
        raw_frames_advanced = 0
        environment_applied_action = None
        chunk_request = isinstance(action, np.ndarray) and action.ndim == 2
        if chunk_request:
            physical = self._raw_observation["obs"] if isinstance(self._raw_observation, dict) else self._raw_observation
            replay_obs = np.zeros((self.clock.obs_stride_raw_frames + 1, *physical.shape), dtype=np.float32)
            # Snapshot before stepping: native environments may reuse their observation buffer.
            replay_obs[0] = physical
            replay_rewards = np.zeros(self.clock.obs_stride_raw_frames, dtype=np.float32)
            replay_clock = np.full((self.clock.obs_stride_raw_frames, 7), -1, dtype=np.int64)
        for index in range(self.clock.obs_stride_raw_frames):
            applied_source_raw_frame = -1
            applied_source_obs_id = -1
            applied_chunk_index = -1
            applied_chunk_window = (-1, -1)
            released_events, _ = self.result_timeline.release_arrived(self.env_step_idx)
            for released_event in released_events:
                self.action_scheduler.enqueue(released_event)
            # for each raw frame step, ask scheduler which action should be taken
            applied_event_count = self.action_scheduler.applied_event_count
            active_action = self.action_scheduler.update(
                current_step=int(self.env_step_idx),
                current_time_ms=self.clock.step_to_time_ms(int(self.env_step_idx)),
            )
            if not active_action.is_noop:
                applied_event = self.action_scheduler.latest_applied_event
                applied_source_raw_frame = applied_event.obs_env_step
                applied_source_obs_id = applied_event.latency.obs_id
                if applied_event.action_chunk is not None:
                    applied_chunk_index = applied_event.metadata["chunk_index"]
                    if self.action_scheduler.applied_event_count != applied_event_count:
                        applied_event.metadata["planned_chunk_window"] = self._planned_chunk_window(applied_event)
                    applied_chunk_window = applied_event.metadata["planned_chunk_window"]
            self.current_applied_action = clone_action(active_action.value)
            if chunk_request:
                replay_clock[index] = (
                    self.episode_idx, self.env_step_idx, applied_source_raw_frame,
                    applied_chunk_index, index == 0 and not observation_dropped,
                    *applied_chunk_window,
                )
            obs, reward, terminated, truncated, info = self.env.step(clone_action(active_action.value))
            self._raw_observation = obs
            if chunk_request:
                replay_obs[index + 1] = obs["obs"] if isinstance(obs, dict) else obs
                replay_rewards[index] = reward
            # accumulate rewards for raw frames
            total_reward += float(reward)
            step_info = dict(info or {})
            if "applied_action" in step_info:
                environment_applied_action = clone_action(step_info["applied_action"])
            if "raw_reward" in step_info:
                total_raw_reward += float(step_info["raw_reward"])
                saw_raw_reward = True
            raw_frames_advanced += 1
            self.env_step_idx += 1
            if terminated or truncated:
                break

        if obs is None:
            raise RuntimeError("ActionLatencyWrapper did not advance any raw frames")

        if self.release_results_at_observation_boundary:
            released_events, _ = self.result_timeline.release_arrived(self.env_step_idx)
            for released_event in released_events:
                self.action_scheduler.enqueue(released_event)
            if self.worker_capacity is not None:
                self.worker_capacity.release_ready(self.env_step_idx)

        info = dict(info or {})
        if chunk_request:
            info["replay_trace"] = {
                "replay_obs": replay_obs,
                "replay_rewards": replay_rewards,
                "replay_clock": replay_clock,
                "replay_length": np.asarray(raw_frames_advanced, dtype=np.int64),
            }
        if saw_raw_reward:
            info["raw_reward"] = float(total_raw_reward)
        info["command_admitted"] = not observation_dropped
        info["observation_dropped"] = observation_dropped
        info["worker_id"] = -1 if worker_id is None else worker_id
        info["worker_service_ms"] = 0.0 if latency_sample is None else latency_sample.worker_service_ms
        info["worker_service_raw_frames"] = (
            0 if latency_sample is None else latency_sample.worker_service_raw_frames
        )
        info["ready_raw_frame"] = -1 if scheduled_ready_raw_frame is None else scheduled_ready_raw_frame
        if self.add_latency_info:
            info.update(self._latency_info(scheduled_delay))
            info["issued_action"] = clone_action(action)
            info["queued_action"] = clone_action(action)
            info["in_flight_count"] = (
                self.worker_capacity.in_flight_count if self.worker_capacity is not None else 0
            )
            info["idle_worker_count"] = (
                self.worker_capacity.idle_worker_count if self.worker_capacity is not None else None
            )
            info["raw_frames_advanced"] = int(raw_frames_advanced)
            info["obs_stride_raw_frames"] = int(self.clock.obs_stride_raw_frames)
            info["applied_source_raw_frame"] = applied_source_raw_frame
            info["applied_source_obs_id"] = applied_source_obs_id
            info["applied_chunk_index"] = applied_chunk_index
            info["applied_chunk_window"] = applied_chunk_window
            info["raw_done"] = terminated or truncated
        if environment_applied_action is not None:
            info["applied_action"] = environment_applied_action
        if self.decision_action_history is not None:
            self.decision_action_history.append(
                [0], [not observation_dropped], [action], [self.current_applied_action]
            )
        self.decision_step_idx += 1
        return self._transport_obs(obs), float(total_reward), terminated, truncated, info


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
        return obs, dict(info or {})

    def step(self, action):
        self._record_issued_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_value = float(reward)
        info = dict(info or {})
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

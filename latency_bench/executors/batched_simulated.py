from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ActionEvent, EpisodeMetrics, LatencyRecord, Observation, StepRecord
from latency_bench.envs.base import EnvAdapter
from latency_bench.executors._simulated_timeline import SimulatedLatencySample, sample_simulated_latency
from latency_bench.executors.env_step_backend import EnvStepBackend
from latency_bench.latency.sources import LatencySource
from latency_bench.logging.metrics import (
    active_action_step_info,
    apply_latency_raw_fields,
    apply_latency_record_raw_fields,
    compute_episode_metrics,
    compute_episode_metrics_from_aggregates,
    episode_raw_fact_metadata,
    latency_type_from_source,
    profile_metadata_from_source,
)
from latency_bench.logging.records import step_record_info
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.base import PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import DecisionScheduler


@dataclass
class _EpisodeBuffers:
    step_records: list[StepRecord] | None = None
    action_events: list[ActionEvent] | None = None
    latency_records: list[LatencyRecord] | None = None
    latency_values_ms: list[float] = field(default_factory=list)
    episode_return_env: float = 0.0
    survival_steps: int = 0
    game_score: float | None = None
    return_raw: float | None = None
    num_actions: int = 0
    num_dropped_actions: int = 0
    num_invalid_actions: int = 0

    def record_step(self, *, reward: float, info: dict) -> None:
        self.episode_return_env += float(reward)
        self.survival_steps += 1
        if "invalid_action" in info and info["invalid_action"]:
            self.num_invalid_actions += 1
        if "game_score" in info:
            self.game_score = float(info["game_score"])
        if "score" in info:
            self.game_score = float(info["score"])
        for key in ("return_raw", "raw_return", "episodic_raw_return", "episode/raw_return"):
            if key in info and info[key] is not None:
                self.return_raw = float(info[key])
        extra_stats = info["episode_extra_stats"] if "episode_extra_stats" in info else None
        if isinstance(extra_stats, dict):
            for key in ("return_raw", "raw_return", "episodic_raw_return", "episode/raw_return"):
                if key in extra_stats and extra_stats[key] is not None:
                    self.return_raw = float(extra_stats[key])


@dataclass
class _SlotState:
    slot_id: int
    env: EnvAdapter
    latency_source: LatencySource
    action_scheduler: ActionScheduler
    active: bool = False
    episode_id: int | None = None
    episode_seed: int | None = None
    env_step: int = 0
    buffers: _EpisodeBuffers = field(default_factory=_EpisodeBuffers)


@dataclass
class _PendingPolicyObservation:
    slot: _SlotState
    observation: Observation
    obs_id: int
    latency_sample: SimulatedLatencySample


class _BatchedEnvCollection:
    def __init__(self, envs: Sequence[EnvAdapter], close_fn: Callable[[], None]):
        self.envs = list(envs)
        self._close_fn = close_fn
        self.noop_action = self.envs[0].noop_action if self.envs else None
        self.env_fps = self.envs[0].env_fps if self.envs else 0.0

    def close(self) -> None:
        self._close_fn()


class BatchedSimulatedLatencyExecutor:
    """Run multiple simulated episodes concurrently with independent slot state.

    The main process owns policy inference, latency scheduling, episode accounting,
    and logging. Env stepping can be serial in-process or delegated to worker
    subprocesses through env_backend.
    """

    def __init__(
        self,
        *,
        envs: Sequence[EnvAdapter] | None = None,
        env_backend: EnvStepBackend | None = None,
        policy: PolicyRunner,
        decision_scheduler: DecisionScheduler,
        latency_sources: Sequence[LatencySource],
        action_schedulers: Sequence[ActionScheduler],
        clock: EnvClock,
        logger: TrajectoryLogger | None = None,
        episode_latency_source_factory: Callable[[int], LatencySource] | None = None,
    ):
        slot_count = env_backend.num_slots
        self.env_backend = env_backend
        self.envs = list(env_backend.slot_handles)
        self.env = _BatchedEnvCollection(self.envs, close_fn=self.env_backend.close)
        self.policy = policy
        self.decision_scheduler = decision_scheduler
        self.clock = clock
        self.logger = logger
        self._collect_step_records = bool(logger is not None and logger.save_step_records)
        self._collect_action_records = bool(logger is not None and logger.save_action_records)
        self._collect_latency_records = bool(logger is not None and logger.save_latency_records)
        self.episode_latency_source_factory = episode_latency_source_factory
        self.slots = [
            _SlotState(
                slot_id=slot_id,
                env=self.envs[slot_id],
                latency_source=latency_sources[slot_id],
                action_scheduler=action_schedulers[slot_id],
                buffers=self._new_episode_buffers(),
            )
            for slot_id in range(slot_count)
        ]
        self._next_obs_id = 0
        self._next_action_id = 0
        self.started_episodes = 0
        self.completed_episodes = 0
        self.active_slots = 0
        self.slot_episode_id: list[int | None] = [None for _slot in self.slots]
        self._completed_metrics: dict[int, EpisodeMetrics] = {}
        self._completed_buffers: dict[int, _EpisodeBuffers] = {}
        self._episode_log_order: list[int] = []
        self._next_episode_log_index = 0

    @property
    def num_slots(self) -> int:
        return len(self.slots)

    def close(self) -> None:
        self.env_backend.close()

    def run_episodes(
        self,
        *,
        episode_ids: Sequence[int],
        seeds: Sequence[int | None],
        eval_max_steps: int = 10000,
        on_episode_complete: Callable[[EpisodeMetrics], None] | None = None,
    ) -> list[EpisodeMetrics]:
        if eval_max_steps < 0:
            raise ValueError("eval_max_steps must be non-negative")
        episode_ids = [int(episode_id) for episode_id in episode_ids]
        if len(seeds) != len(episode_ids):
            raise ValueError("seeds length must match episode_ids length")

        self._reset_run_state(episode_ids)
        if not episode_ids:
            return []

        next_episode_index = 0
        initial_slots = min(self.num_slots, len(episode_ids))
        for slot in self.slots[:initial_slots]:
            self._start_slot(
                slot,
                episode_id=episode_ids[next_episode_index],
                seed=seeds[next_episode_index],
            )
            next_episode_index += 1

        while self.completed_episodes < len(episode_ids):
            active_slots = self._active_slots()
            if eval_max_steps == 0:
                for slot in active_slots:
                    self._complete_slot(slot, on_episode_complete=on_episode_complete)
                    if next_episode_index < len(episode_ids):
                        self._start_slot(
                            slot,
                            episode_id=episode_ids[next_episode_index],
                            seed=seeds[next_episode_index],
                        )
                        next_episode_index += 1
                continue

            observations = []
            observation_slots: list[_SlotState] = []
            for slot in active_slots:
                current_time_ms = self.clock.step_to_time_ms(slot.env_step)
                if self.decision_scheduler.should_observe(slot.env_step, current_time_ms):
                    observation_slots.append(slot)

            if observation_slots:
                observations_by_slot = self.env_backend.observe_slots([slot.slot_id for slot in observation_slots])
                pending_observations = [
                    self._sample_policy_observation(
                        slot,
                        self._policy_observation(slot, observations_by_slot[slot.slot_id]),
                    )
                    for slot in observation_slots
                ]
                observations = [pending.observation for pending in pending_observations]

            if observations:
                policy_outputs = self.policy.predict_batch(observations)
                if len(policy_outputs) != len(observations):
                    raise RuntimeError("policy.predict_batch returned the wrong number of outputs")
                for pending, policy_output in zip(pending_observations, policy_outputs):
                    self._enqueue_policy_output(
                        pending.slot,
                        pending.observation,
                        policy_output,
                        obs_id=pending.obs_id,
                        latency_sample=pending.latency_sample,
                    )

            actions_by_slot = {}
            for slot in active_slots:
                current_time_ms = self.clock.step_to_time_ms(slot.env_step)
                active_action = slot.action_scheduler.update(slot.env_step, current_time_ms)
                actions_by_slot[slot.slot_id] = active_action

            step_responses = self.env_backend.step_slots(actions_by_slot)
            for slot in active_slots:
                current_time_ms = self.clock.step_to_time_ms(slot.env_step)
                active_action = actions_by_slot[slot.slot_id]
                result = step_responses[slot.slot_id].result
                soft_reset = bool(result.info.get("soft_reset")) if isinstance(result.info, dict) else False
                episode_done = bool(result.done or result.truncated) and not soft_reset
                if slot.buffers.step_records is not None:
                    latency_type = latency_type_from_source(slot.latency_source)
                    step_info = step_record_info(result.info)
                    step_info.update(
                        active_action_step_info(
                            applied_events=slot.action_scheduler.get_applied_events(),
                            env_step=slot.env_step,
                            scheduled_time_ms=current_time_ms,
                            frame_ms=self.clock.frame_ms,
                            latency_type=latency_type,
                            profile_metadata=profile_metadata_from_source(slot.latency_source),
                        )
                    )
                    record = StepRecord(
                        episode_id=int(slot.episode_id),
                        env_step=slot.env_step,
                        sim_time_ms=current_time_ms,
                        active_action=active_action,
                        reward=float(result.reward),
                        done=episode_done,
                        info=step_info,
                    )
                    slot.buffers.step_records.append(record)
                    slot.buffers.record_step(reward=float(result.reward), info=step_info)
                else:
                    slot.buffers.record_step(reward=float(result.reward), info=result.info)
                if soft_reset:
                    slot.buffers.num_dropped_actions += slot.action_scheduler.dropped_events_count
                    slot.action_scheduler.reset()
                    self.policy.reset_state(slot_id=slot.slot_id)

                slot.env_step += 1
                if episode_done or slot.env_step >= eval_max_steps:
                    self._complete_slot(slot, on_episode_complete=on_episode_complete)
                    if next_episode_index < len(episode_ids):
                        self._start_slot(
                            slot,
                            episode_id=episode_ids[next_episode_index],
                            seed=seeds[next_episode_index],
                        )
                        next_episode_index += 1

        return self._ordered_metrics(episode_ids)

    def _policy_observation(self, slot: _SlotState, observation: Observation) -> Observation:
        metadata = dict(observation.metadata)
        metadata["slot_id"] = slot.slot_id
        metadata["episode_id"] = int(slot.episode_id)
        return Observation(
            data=observation.data,
            env_step=observation.env_step,
            sim_time_ms=observation.sim_time_ms,
            metadata=metadata,
        )

    def _sample_policy_observation(
        self,
        slot: _SlotState,
        observation: Observation,
    ) -> _PendingPolicyObservation:
        obs_id = self._next_obs_id
        self._next_obs_id += 1
        raw_frame = int(slot.env_step)
        current_time_ms = self.clock.step_to_time_ms(raw_frame)
        latency_context = {
            "observation": observation,
            "obs_id": obs_id,
            "env_step": raw_frame,
            "raw_frame": raw_frame,
            "sim_time_ms": current_time_ms,
            "episode_id": slot.episode_id,
            "slot_id": slot.slot_id,
        }
        latency_sample = sample_simulated_latency(
            latency_source=slot.latency_source,
            clock=self.clock,
            context=latency_context,
        )
        metadata = dict(observation.metadata)
        metadata.update(
            {
                "obs_id": obs_id,
                "latency_ms": latency_sample.latency_ms,
                "latency_raw_frames": latency_sample.latency_raw_frames,
            }
        )
        return _PendingPolicyObservation(
            slot=slot,
            observation=Observation(
                data=observation.data,
                env_step=observation.env_step,
                sim_time_ms=observation.sim_time_ms,
                metadata=metadata,
            ),
            obs_id=obs_id,
            latency_sample=latency_sample,
        )

    def _reset_run_state(self, episode_ids: Sequence[int]) -> None:
        self.started_episodes = 0
        self.completed_episodes = 0
        self.active_slots = 0
        self.slot_episode_id = [None for _slot in self.slots]
        self._completed_metrics.clear()
        self._completed_buffers.clear()
        self._episode_log_order = [int(episode_id) for episode_id in episode_ids]
        self._next_episode_log_index = 0
        for slot in self.slots:
            slot.active = False
            slot.episode_id = None
            slot.episode_seed = None
            slot.env_step = 0
            slot.buffers = self._new_episode_buffers()
            slot.action_scheduler.reset()

    def _active_slots(self) -> list[_SlotState]:
        return [slot for slot in self.slots if slot.active]

    def _start_slot(self, slot: _SlotState, *, episode_id: int, seed: int | None) -> None:
        if self.episode_latency_source_factory is not None:
            slot.latency_source = self.episode_latency_source_factory(episode_id)
        slot.active = True
        slot.episode_id = int(episode_id)
        slot.episode_seed = None if seed is None else int(seed)
        slot.env_step = 0
        slot.buffers = self._new_episode_buffers()
        slot.action_scheduler.reset()
        self.policy.reset_state(slot_id=slot.slot_id)
        self.env_backend.reset_slot(slot.slot_id, episode_id=episode_id, seed=seed)
        self.slot_episode_id[slot.slot_id] = int(episode_id)
        self.started_episodes += 1
        self.active_slots = sum(1 for item in self.slots if item.active)

    def _complete_slot(
        self,
        slot: _SlotState,
        *,
        on_episode_complete: Callable[[EpisodeMetrics], None] | None = None,
    ) -> None:
        if not slot.active or slot.episode_id is None:
            return
        episode_id = int(slot.episode_id)
        slot.buffers.num_dropped_actions += slot.action_scheduler.dropped_events_count
        metrics = self._compute_episode_metrics(
            episode_id=episode_id,
            buffers=slot.buffers,
            metadata=episode_raw_fact_metadata(
                mode="simulated",
                episode_seed=slot.episode_seed,
                env_fps=self.clock.env_fps,
                obs_fps=self.clock.obs_fps,
                frame_ms=self.clock.frame_ms,
                latency_type=latency_type_from_source(slot.latency_source),
                latency_source=slot.latency_source,
            ),
        )
        self._completed_metrics[episode_id] = metrics
        self._completed_buffers[episode_id] = slot.buffers
        self.completed_episodes += 1
        slot.active = False
        slot.episode_id = None
        slot.episode_seed = None
        slot.env_step = 0
        slot.buffers = self._new_episode_buffers()
        slot.action_scheduler.reset()
        self.slot_episode_id[slot.slot_id] = None
        self.active_slots = sum(1 for item in self.slots if item.active)
        self._flush_completed_in_episode_order()
        if on_episode_complete is not None:
            on_episode_complete(metrics)

    def _enqueue_policy_output(
        self,
        slot: _SlotState,
        observation,
        policy_output,
        *,
        obs_id: int,
        latency_sample: SimulatedLatencySample,
    ) -> None:
        raw_frame = int(slot.env_step)
        latency_ms = latency_sample.latency_ms
        ready_raw_frame = raw_frame + latency_sample.latency_raw_frames
        ready_time_ms = self.clock.step_to_time_ms(ready_raw_frame)
        latency_type = latency_type_from_source(slot.latency_source)
        profile_metadata = profile_metadata_from_source(slot.latency_source)
        latency_record = LatencyRecord(
            obs_id=obs_id,
            latency_ms=latency_ms,
            metadata={
                "episode_id": int(slot.episode_id),
                "slot_id": int(slot.slot_id),
            },
        )
        apply_latency_record_raw_fields(
            record=latency_record,
            latency_type=latency_type,
            profile_metadata=profile_metadata,
        )
        metadata = {
            **policy_output.metadata,
            "episode_id": int(slot.episode_id),
            "slot_id": int(slot.slot_id),
            "frame_ms": self.clock.frame_ms,
        }
        event = ActionEvent(
            action_id=self._next_action_id,
            action=policy_output.action,
            obs_env_step=raw_frame,
            ready_time_ms=ready_time_ms,
            ready_step_float=float(ready_raw_frame),
            latency=latency_record,
            latency_raw_frames=latency_sample.latency_raw_frames,
            ready_raw_frame=ready_raw_frame,
            metadata=metadata,
        )
        apply_latency_raw_fields(
            event=event,
            latency_type=latency_type,
            frame_ms=self.clock.frame_ms,
            profile_metadata=profile_metadata,
        )
        self._next_action_id += 1
        slot.action_scheduler.enqueue(event)
        slot.buffers.num_actions += 1
        if slot.buffers.action_events is not None:
            slot.buffers.action_events.append(event)
        slot.buffers.latency_values_ms.append(latency_ms)
        if slot.buffers.latency_records is not None:
            slot.buffers.latency_records.append(latency_record)

    def _flush_completed_in_episode_order(self) -> None:
        if self.logger is None:
            return
        while self._next_episode_log_index < len(self._episode_log_order):
            episode_id = self._episode_log_order[self._next_episode_log_index]
            if episode_id not in self._completed_metrics:
                break
            buffers = self._completed_buffers[episode_id]
            metrics = self._completed_metrics[episode_id]
            if buffers.step_records is not None:
                for record in buffers.step_records:
                    self.logger.log_step(record)
            if buffers.action_events is not None:
                for event in buffers.action_events:
                    self.logger.log_action_event(event)
            if buffers.latency_records is not None:
                for latency_record in buffers.latency_records:
                    self.logger.log_latency(latency_record)
            self.logger.log_episode_metrics(metrics)
            self._next_episode_log_index += 1

    def _ordered_metrics(self, episode_ids: Sequence[int]) -> list[EpisodeMetrics]:
        return [self._completed_metrics[int(episode_id)] for episode_id in episode_ids]

    def _new_episode_buffers(self) -> _EpisodeBuffers:
        return _EpisodeBuffers(
            step_records=[] if self._collect_step_records else None,
            action_events=[] if self._collect_action_records else None,
            latency_records=[] if self._collect_latency_records else None,
        )

    def _compute_episode_metrics(
        self,
        *,
        episode_id: int,
        buffers: _EpisodeBuffers,
        metadata: dict,
    ) -> EpisodeMetrics:
        if buffers.step_records is not None and buffers.action_events is not None:
            return compute_episode_metrics(
                episode_id=episode_id,
                step_records=buffers.step_records,
                action_events=buffers.action_events,
                latency_values_ms=buffers.latency_values_ms,
                metadata=metadata,
                frame_ms=self.clock.frame_ms,
            )
        return compute_episode_metrics_from_aggregates(
            episode_id=episode_id,
            episode_return_env=buffers.episode_return_env,
            survival_steps=buffers.survival_steps,
            return_raw=buffers.return_raw,
            game_score=buffers.game_score,
            latency_values_ms=buffers.latency_values_ms,
            num_actions=buffers.num_actions,
            num_dropped_actions=buffers.num_dropped_actions,
            num_invalid_actions=buffers.num_invalid_actions,
            metadata=metadata,
        )

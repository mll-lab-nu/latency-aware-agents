from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from latency_bench.core.clock import EnvClock
from latency_bench.core.decision_action_history import DecisionActionHistory
from latency_bench.core.timing import StageProfiler, profiler_scope
from latency_bench.core.types import ActionEvent, EpisodeMetrics, LatencyRecord, Observation, StepRecord
from latency_bench.envs.atari import TRUE_EPISODE_END_INFO_KEY
from latency_bench.envs.base import EnvAdapter
from latency_bench.executors._simulated_timeline import (
    SimulatedResultTimeline,
    SimulatedWorkerCapacity,
    build_simulated_action_event,
)
from latency_bench.executors.base import BatchedExecutor
from latency_bench.executors.env_step_backend import EnvStepBackend, env_action_space
from latency_bench.latency.sample import LatencySample
from latency_bench.latency.samplers import LatencySampler
from latency_bench.logging.metrics import (
    EpisodeMetricAccumulator,
    active_action_step_info,
    episode_raw_fact_metadata,
    latency_type_from_source,
    profile_metadata_from_source,
)
from latency_bench.logging.records import build_step_record
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.action_prefix import with_action_prefix
from latency_bench.policy.base import PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import DecisionScheduler
from latency_bench.utils.io import write_json
from latency_bench.utils.stats import series_stats


@dataclass
class _EpisodeBuffers(EpisodeMetricAccumulator):
    step_records: list[StepRecord] | None = None
    action_events: list[ActionEvent] | None = None
    latency_records: list[LatencyRecord] | None = None
    latency_values_ms: list[float] = field(default_factory=list)
    num_actions: int = 0
    num_dropped_actions: int = 0
    submitted_observation_frames: int = 0
    dropped_observation_count: int = 0
    soft_reset_count: int = 0
    final_lives: int | None = None
    final_is_true_episode_end: bool | None = None

    def record_step(self, *, reward: float, info: dict) -> None:
        self.update(reward, info)
        if "soft_reset" in info and info["soft_reset"]:
            self.soft_reset_count += 1
        if "lives" in info:
            self.final_lives = info["lives"]
        if TRUE_EPISODE_END_INFO_KEY in info:
            self.final_is_true_episode_end = info[TRUE_EPISODE_END_INFO_KEY]


@dataclass
class _SlotState:
    slot_id: int
    env: EnvAdapter
    latency_source: LatencySampler
    action_scheduler: ActionScheduler
    result_timeline: SimulatedResultTimeline
    active: bool = False
    episode_id: int | None = None
    episode_seed: int | None = None
    env_step: int = 0
    recent_drop_count: int = 0
    decision_action_history: DecisionActionHistory | None = None
    decision_admitted: bool = False
    decision_issued_action: object = None
    buffers: _EpisodeBuffers = field(default_factory=_EpisodeBuffers)
    worker_capacity: SimulatedWorkerCapacity = field(
        default_factory=lambda: SimulatedWorkerCapacity(capacity=None, busy_until_by_worker={})
    )


@dataclass
class _PendingPolicyObservation:
    slot: _SlotState
    observation: Observation
    obs_id: int
    latency_sample: LatencySample
    worker_slot: int


class BatchedSimulatedLatencyExecutor(BatchedExecutor):
    """Run multiple simulated episodes concurrently with independent slot state.

    The main process owns policy inference, latency scheduling, episode accounting,
    and logging. Env stepping can be serial in-process or delegated to worker
    subprocesses through env_backend.
    """

    def __init__(
        self,
        *,
        env_backend: EnvStepBackend,
        policy: PolicyRunner,
        decision_scheduler: DecisionScheduler,
        latency_sources: Sequence[LatencySampler],
        action_schedulers: Sequence[ActionScheduler],
        clock: EnvClock,
        logger: TrajectoryLogger | None = None,
        episode_latency_source_factory: Callable[[int], LatencySampler] | None = None,
        simulated_worker_capacity: int | None = None,
        profile_pipeline: bool = False,
        inference_pool=None,
        action_prefix=None,
        action_history_decisions: int | None = None,
    ):
        slot_count = env_backend.num_slots
        self.env_backend = env_backend
        self.envs = list(env_backend.slot_handles)
        self.policy = policy
        self.decision_scheduler = decision_scheduler
        self.clock = clock
        self.logger = logger
        self.profile_pipeline = bool(profile_pipeline)
        self.inference_pool = inference_pool
        self.action_prefix = action_prefix
        self._pipeline_profile_rows: list[dict[str, float]] = []
        self.simulated_worker_capacity = simulated_worker_capacity
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
                result_timeline=SimulatedResultTimeline(
                    ordering_policy=action_schedulers[slot_id].ordering_policy
                ),
                decision_action_history=(
                    DecisionActionHistory(
                        env_action_space(self.envs[slot_id]), num_envs=1, decisions=action_history_decisions
                    ) if action_history_decisions is not None else None
                ),
                buffers=self._new_episode_buffers(),
                worker_capacity=SimulatedWorkerCapacity(
                    capacity=simulated_worker_capacity,
                    busy_until_by_worker={},
                ),
            )
            for slot_id in range(slot_count)
        ]
        self._next_obs_id = 0
        self._next_action_id = 0
        self.started_episodes = 0
        self.completed_episodes = 0
        self._completed_metrics: dict[int, EpisodeMetrics] = {}
        self._completed_buffers: dict[int, _EpisodeBuffers] = {}
        self._episode_log_order: list[int] = []
        self._next_episode_log_index = 0

    @property
    def num_slots(self) -> int:
        return len(self.slots)

    def close(self) -> None:
        if self.inference_pool is not None:
            self.inference_pool.close()
        self.env_backend.close()

    def run_episodes(
        self,
        *,
        episode_ids: Sequence[int],
        seeds: Sequence[int | None],
        eval_max_steps: int = 10000,
        on_episode_complete: Callable[[EpisodeMetrics], None] | None = None,
        episode_groups: Sequence[Sequence[int]] | None = None,
        latency_sources_by_episode: Mapping[int, LatencySampler] | None = None,
    ) -> list[EpisodeMetrics]:
        """Run episodes, optionally keeping each ordered session on one slot.

        Explicit sources let session callers share a Temporal sampler across
        their episodes without tying its random stream to physical slot IDs.
        """
        if eval_max_steps < 0:
            raise ValueError("eval_max_steps must be non-negative")
        episode_ids = [int(episode_id) for episode_id in episode_ids]
        if len(seeds) != len(episode_ids):
            raise ValueError("seeds length must match episode_ids length")

        self._reset_run_state(episode_ids)
        if not episode_ids:
            return []

        groups = deque(
            deque(group) for group in (
                episode_groups if episode_groups is not None else [[eid] for eid in episode_ids]
            )
        )
        slot_episodes = [deque() for _ in self.slots]
        seeds_by_episode = dict(zip(episode_ids, seeds))

        def start_next(slot):
            pending = slot_episodes[slot.slot_id]
            if not pending and groups:
                pending.extend(groups.popleft())
            if pending:
                episode_id = pending.popleft()
                self._start_slot(
                    slot, episode_id=episode_id, seed=seeds_by_episode[episode_id],
                    latency_source=(latency_sources_by_episode[episode_id]
                                    if latency_sources_by_episode is not None else None),
                )

        for slot in self.slots:
            start_next(slot)

        while self.completed_episodes < len(episode_ids):
            active_slots = self._active_slots()
            if eval_max_steps == 0:
                for slot in active_slots:
                    self._complete_slot(slot, on_episode_complete=on_episode_complete,
                                        preserve_episode_order=episode_groups is None)
                    start_next(slot)
                continue

            observations = []
            observation_slots: list[_SlotState] = []
            step_capacity_info: dict[int, dict[str, int | bool | None]] = {}
            for slot in active_slots:
                current_time_ms = self.clock.step_to_time_ms(slot.env_step)
                slot.worker_capacity.release_ready(slot.env_step)
                self._deliver_arrived_results(slot, raw_frame=slot.env_step)
                observation_submitted = False
                observation_dropped = False
                if self.decision_scheduler.should_observe(slot.env_step, current_time_ms):
                    prefix_request_pending = (
                        self.action_prefix is not None
                        and self.action_prefix["mode"] != "none"
                        and slot.result_timeline.pending_observation_count > 0
                    )
                    if slot.worker_capacity.can_submit() and not prefix_request_pending:
                        observation_slots.append(slot)
                        observation_submitted = True
                    else:
                        slot.buffers.dropped_observation_count += 1
                        observation_dropped = True
                        slot.recent_drop_count += 1
                if self.simulated_worker_capacity is not None:
                    step_capacity_info[slot.slot_id] = {
                        "observation_submitted": observation_submitted,
                        "observation_dropped": observation_dropped,
                    }
                if slot.decision_action_history is not None and slot.env_step % self.clock.obs_stride_raw_frames == 0:
                    slot.decision_admitted = observation_submitted
                    slot.decision_issued_action = slot.action_scheduler.noop_action.value

            observe_ms = 0.0
            if observation_slots:
                observe_start = time.perf_counter()
                observations_by_slot = self.env_backend.observe_slots([slot.slot_id for slot in observation_slots])
                observe_ms = (time.perf_counter() - observe_start) * 1000.0
                pending_observations = [
                    self._sample_policy_observation(
                        slot,
                        self._policy_observation(
                            slot,
                            observations_by_slot[slot.slot_id],
                            transport=(
                                slot.decision_action_history.observation()[0]
                                if slot.decision_action_history is not None else None
                            ),
                        ),
                    )
                    for slot in observation_slots
                ]
                observations = [pending.observation for pending in pending_observations]

            profile_row = None
            if observations:
                profiler = StageProfiler(enabled=self.profile_pipeline)
                with profiler_scope(profiler):
                    policy_outputs = (
                        self.inference_pool.predict_batch(observations)
                        if self.inference_pool is not None
                        else self.policy.predict_batch(observations)
                    )
                if len(policy_outputs) != len(observations):
                    raise RuntimeError("policy.predict_batch returned the wrong number of outputs")
                if self.profile_pipeline:
                    profile_row = {
                        "active_slots": float(len(active_slots)),
                        "batch_size": float(len(observations)),
                        "observe_slots_ms": observe_ms,
                        **{key: float(value) for key, value in profiler.timings.items()},
                    }
                for pending, policy_output in zip(pending_observations, policy_outputs):
                    if pending.slot.decision_action_history is not None:
                        pending.slot.decision_issued_action = policy_output.action.value
                    self._enqueue_policy_output(
                        pending.slot,
                        pending.observation,
                        policy_output,
                        obs_id=pending.obs_id,
                        latency_sample=pending.latency_sample,
                        worker_slot=pending.worker_slot,
                    )

            actions_by_slot = {}
            for slot in active_slots:
                current_time_ms = self.clock.step_to_time_ms(slot.env_step)
                self._deliver_arrived_results(slot, raw_frame=slot.env_step)
                active_action = slot.action_scheduler.update(slot.env_step, current_time_ms)
                actions_by_slot[slot.slot_id] = active_action

            env_step_start = time.perf_counter()
            step_responses = self.env_backend.step_slots(actions_by_slot)
            if profile_row is not None:
                profile_row["env_step_ms"] = (time.perf_counter() - env_step_start) * 1000.0
                self._pipeline_profile_rows.append(profile_row)
            for slot in active_slots:
                current_time_ms = self.clock.step_to_time_ms(slot.env_step)
                active_action = actions_by_slot[slot.slot_id]
                if (
                    slot.decision_action_history is not None
                    and (slot.env_step + 1) % self.clock.obs_stride_raw_frames == 0
                ):
                    slot.decision_action_history.append(
                        [0], [slot.decision_admitted],
                        [slot.decision_issued_action], [active_action.value],
                    )
                result = step_responses[slot.slot_id].result
                soft_reset = bool(result.info.get("soft_reset")) if isinstance(result.info, dict) else False
                episode_done = bool(result.done or result.truncated) and not soft_reset
                if slot.buffers.step_records is not None:
                    record = build_step_record(
                        episode_id=int(slot.episode_id),
                        env_step=slot.env_step,
                        scheduled_time_ms=current_time_ms,
                        active_action=active_action,
                        reward=result.reward,
                        done=episode_done,
                        info=result.info,
                        active_event=slot.action_scheduler.latest_applied_event,
                        frame_ms=self.clock.frame_ms,
                        latency_type=latency_type_from_source(slot.latency_source),
                    )
                    slot.buffers.step_records.append(record)
                    slot.buffers.record_step(reward=float(result.reward), info=record.info)
                else:
                    info = result.info
                    event = slot.action_scheduler.latest_applied_event
                    if event is not None and event.action_chunk is not None:
                        info = info | active_action_step_info(
                            active_event=event, scheduled_time_ms=current_time_ms,
                            frame_ms=self.clock.frame_ms,
                            latency_type=latency_type_from_source(slot.latency_source),
                        )
                    slot.buffers.record_step(reward=float(result.reward), info=info)
                if self.simulated_worker_capacity is not None and slot.buffers.step_records is not None:
                    slot.buffers.step_records[-1].info.update(
                        {
                            **step_capacity_info[slot.slot_id],
                            "in_flight_count": slot.worker_capacity.in_flight_count,
                            "idle_worker_count": slot.worker_capacity.idle_worker_count,
                        }
                    )
                if soft_reset:
                    slot.buffers.num_dropped_actions += slot.action_scheduler.dropped_events_count
                    slot.action_scheduler.reset()
                    slot.result_timeline.reset()
                    self._reset_policy_state(slot.slot_id)
                    slot.worker_capacity.reset()
                    if slot.decision_action_history is not None:
                        slot.decision_action_history.reset()
                    slot.recent_drop_count = 0

                slot.env_step += 1
                if episode_done or slot.env_step >= eval_max_steps:
                    self._complete_slot(slot, on_episode_complete=on_episode_complete,
                                        preserve_episode_order=episode_groups is None)
                    start_next(slot)

        self._write_pipeline_profile_summary()
        return self._ordered_metrics(episode_ids)

    def _policy_observation(
        self,
        slot: _SlotState,
        observation: Observation,
        transport: np.ndarray | None = None,
    ) -> Observation:
        observation = with_action_prefix(observation, slot.action_scheduler, self.action_prefix)
        metadata = dict(observation.metadata)
        metadata["slot_id"] = slot.slot_id
        metadata["episode_id"] = int(slot.episode_id)
        metadata["action_noise_seed"] = slot.episode_seed
        data = observation.data
        if transport is not None:
            data = {**data, "transport": transport} if isinstance(data, Mapping) else {"obs": data, "transport": transport}
        return Observation(
            data=data,
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
        worker_slot = slot.worker_capacity.assign_worker()
        latency_context = {
            "observation": observation,
            "obs_id": obs_id,
            "env_step": raw_frame,
            "raw_frame": raw_frame,
            "sim_time_ms": current_time_ms,
            "episode_id": slot.episode_id,
            "slot_id": slot.slot_id,
            "worker_slot": worker_slot,
            "recent_drop_count": slot.recent_drop_count,
            "in_flight_count": slot.worker_capacity.in_flight_count,
            "idle_worker_count": slot.worker_capacity.idle_worker_count,
        }
        latency_sample = slot.latency_source.sample(latency_context)
        metadata = dict(observation.metadata)
        metadata["obs_id"] = obs_id
        policy_observation = Observation(
            data=observation.data,
            env_step=observation.env_step,
            sim_time_ms=observation.sim_time_ms,
            metadata=metadata,
        )
        slot.worker_capacity.submit(
            worker_slot, raw_frame + latency_sample.worker_service_raw_frames
        )
        return _PendingPolicyObservation(
            slot=slot,
            obs_id=obs_id,
            latency_sample=latency_sample,
            worker_slot=worker_slot,
            observation=policy_observation,
        )

    def _reset_run_state(self, episode_ids: Sequence[int]) -> None:
        self.started_episodes = 0
        self.completed_episodes = 0
        self._pipeline_profile_rows.clear()
        self._completed_metrics.clear()
        self._completed_buffers.clear()
        self._episode_log_order = [int(episode_id) for episode_id in episode_ids]
        self._next_episode_log_index = 0
        for slot in self.slots:
            slot.active = False
            slot.episode_id = None
            slot.episode_seed = None
            slot.env_step = 0
            slot.recent_drop_count = 0
            slot.buffers = self._new_episode_buffers()
            slot.action_scheduler.reset()
            slot.result_timeline.reset()
            slot.worker_capacity.reset()
            if slot.decision_action_history is not None:
                slot.decision_action_history.reset()

    def _active_slots(self) -> list[_SlotState]:
        return [slot for slot in self.slots if slot.active]

    def _deliver_arrived_results(self, slot: _SlotState, *, raw_frame: int | None) -> None:
        released, dropped = slot.result_timeline.release_arrived(raw_frame)
        slot.buffers.num_dropped_actions += len(dropped)
        for event in released:
            slot.action_scheduler.enqueue(event)

    def _start_slot(self, slot: _SlotState, *, episode_id: int, seed: int | None,
                    latency_source: LatencySampler | None = None) -> None:
        if latency_source is not None:
            slot.latency_source = latency_source
        elif self.episode_latency_source_factory is not None:
            slot.latency_source = self.episode_latency_source_factory(episode_id)
        slot.active = True
        slot.episode_id = int(episode_id)
        slot.episode_seed = None if seed is None else int(seed)
        slot.env_step = 0
        slot.recent_drop_count = 0
        slot.buffers = self._new_episode_buffers()
        slot.action_scheduler.reset()
        slot.result_timeline.reset()
        slot.worker_capacity.reset()
        if slot.decision_action_history is not None:
            slot.decision_action_history.reset()
        self._reset_policy_state(slot.slot_id)
        self.env_backend.reset_slot(slot.slot_id, episode_id=episode_id, seed=seed)
        self.started_episodes += 1

    def _reset_policy_state(self, slot_id: int) -> None:
        if self.inference_pool is not None:
            self.inference_pool.reset_state(slot_id)
        else:
            self.policy.reset_state(slot_id=slot_id)

    def _complete_slot(
        self,
        slot: _SlotState,
        *,
        on_episode_complete: Callable[[EpisodeMetrics], None] | None = None,
        preserve_episode_order: bool = True,
    ) -> None:
        if not slot.active or slot.episode_id is None:
            return
        episode_id = int(slot.episode_id)
        self._deliver_arrived_results(slot, raw_frame=None)
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
            )
            | slot.action_scheduler.chunk_metrics()
            | (
                {
                    "submitted_observation_frames": slot.buffers.submitted_observation_frames,
                    "dropped_observation_count": slot.buffers.dropped_observation_count,
                    "simulated_worker_capacity": self.simulated_worker_capacity,
                    "inference_worker_count": self.simulated_worker_capacity,
                    "in_flight_count": slot.worker_capacity.in_flight_count,
                    "idle_worker_count": slot.worker_capacity.idle_worker_count,
                }
                if self.simulated_worker_capacity is not None
                else {}
            ),
        )
        self._completed_metrics[episode_id] = metrics
        self._completed_buffers[episode_id] = slot.buffers
        self.completed_episodes += 1
        slot.active = False
        slot.episode_id = None
        slot.episode_seed = None
        slot.env_step = 0
        slot.recent_drop_count = 0
        slot.buffers = self._new_episode_buffers()
        slot.action_scheduler.reset()
        slot.result_timeline.reset()
        slot.worker_capacity.reset()
        if slot.decision_action_history is not None:
            slot.decision_action_history.reset()
        if preserve_episode_order:
            self._flush_completed_in_episode_order()
        else:
            self._flush_episode(episode_id)
        if on_episode_complete is not None:
            on_episode_complete(metrics)

    def _enqueue_policy_output(
        self,
        slot: _SlotState,
        observation,
        policy_output,
        *,
        obs_id: int,
        latency_sample: LatencySample,
        worker_slot: int,
    ) -> None:
        raw_frame = int(slot.env_step)
        latency_ms = latency_sample.latency_ms
        ready_raw_frame = raw_frame + latency_sample.action_ready_raw_frames
        ready_time_ms = self.clock.step_to_time_ms(ready_raw_frame)
        latency_type = latency_type_from_source(slot.latency_source)
        profile_metadata = profile_metadata_from_source(slot.latency_source)
        slot_metadata = {
            "episode_id": int(slot.episode_id),
            "slot_id": int(slot.slot_id),
            "worker_id": int(worker_slot),
        }
        latency_record, event = build_simulated_action_event(
            action_id=self._next_action_id,
            obs_id=obs_id,
            policy_output=policy_output,
            raw_frame=raw_frame,
            ready_raw_frame=ready_raw_frame,
            ready_time_ms=ready_time_ms,
            latency_sample=latency_sample,
            frame_ms=self.clock.frame_ms,
            latency_type=latency_type,
            profile_metadata=profile_metadata,
            latency_record_metadata=slot_metadata,
            extra_event_metadata=slot_metadata,
        )
        self._next_action_id += 1
        slot.result_timeline.submit(obs_id=obs_id, ready_raw_frame=ready_raw_frame, event=event)
        slot.buffers.submitted_observation_frames += 1
        slot.recent_drop_count = 0
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
            self._flush_episode(episode_id)
            self._next_episode_log_index += 1

    def _flush_episode(self, episode_id: int) -> None:
        if self.logger is None:
            return
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
        buffers.step_records = None
        buffers.action_events = None
        buffers.latency_records = None
        buffers.latency_values_ms.clear()

    def _ordered_metrics(self, episode_ids: Sequence[int]) -> list[EpisodeMetrics]:
        return [self._completed_metrics[int(episode_id)] for episode_id in episode_ids]

    def _new_episode_buffers(self) -> _EpisodeBuffers:
        return _EpisodeBuffers(
            step_records=[] if self._collect_step_records else None,
            action_events=[] if self._collect_action_records else None,
            latency_records=[] if self._collect_latency_records else None,
        )

    def _write_pipeline_profile_summary(self) -> None:
        if not self.profile_pipeline or self.logger is None or not self._pipeline_profile_rows:
            return
        keys = sorted({key for row in self._pipeline_profile_rows for key in row})
        summary = {
            "num_profiled_batches": len(self._pipeline_profile_rows),
            **{
                key: series_stats([float(row[key]) for row in self._pipeline_profile_rows if key in row])
                for key in keys
            },
        }
        write_json(Path(self.logger.output_dir) / "simulated_pipeline_summary.json", summary)

    def _compute_episode_metrics(
        self,
        *,
        episode_id: int,
        buffers: _EpisodeBuffers,
        metadata: dict,
    ) -> EpisodeMetrics:
        if buffers.final_lives is not None:
            metadata["final_lives"] = buffers.final_lives
        if buffers.final_is_true_episode_end is not None:
            metadata["final_is_true_episode_end"] = buffers.final_is_true_episode_end
            metadata["soft_reset_count"] = buffers.soft_reset_count
        return buffers.compute(
            episode_id=episode_id,
            latency_values_ms=buffers.latency_values_ms,
            num_actions=buffers.num_actions,
            num_dropped_actions=buffers.num_dropped_actions,
            metadata=metadata,
        )

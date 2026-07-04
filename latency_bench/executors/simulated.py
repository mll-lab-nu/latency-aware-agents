from __future__ import annotations

from collections.abc import Callable

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ActionEvent, EpisodeMetrics, LatencyRecord, Observation, StepRecord
from latency_bench.envs.base import EnvAdapter
from latency_bench.executors._simulated_timeline import SimulatedWorkerCapacity
from latency_bench.executors.base import EpisodeExecutor
from latency_bench.latency.samplers import LatencySampler
from latency_bench.logging.metrics import (
    active_action_step_info,
    apply_latency_raw_fields,
    apply_latency_record_raw_fields,
    compute_episode_metrics,
    episode_raw_fact_metadata,
    latency_type_from_source,
    profile_metadata_from_source,
)
from latency_bench.logging.records import step_record_info
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.base import PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import DecisionScheduler


class SimulatedLatencyExecutor(EpisodeExecutor):
    def __init__(
        self,
        env: EnvAdapter,
        policy: PolicyRunner,
        decision_scheduler: DecisionScheduler,
        latency_source: LatencySampler,
        action_scheduler: ActionScheduler,
        clock: EnvClock,
        logger: TrajectoryLogger | None = None,
        video_recorder=None,
        video_episode_metadata: dict[int, dict] | None = None,
        episode_latency_source_factory: Callable[[int], LatencySampler] | None = None,
        simulated_worker_capacity: int | None = None,
    ):
        self.env = env
        self.policy = policy
        self.decision_scheduler = decision_scheduler
        self.latency_source = latency_source
        self.action_scheduler = action_scheduler
        self.clock = clock
        self.logger = logger
        self.video_recorder = video_recorder
        self.video_episode_metadata = video_episode_metadata
        self.episode_latency_source_factory = episode_latency_source_factory
        self.simulated_worker_capacity = simulated_worker_capacity
        self._next_obs_id = 0
        self._next_action_id = 0

    def run_episode(
        self,
        episode_id: int,
        seed: int | None = None,
        eval_max_steps: int = 10000,
    ) -> EpisodeMetrics:
        if self.episode_latency_source_factory is not None:
            self.latency_source = self.episode_latency_source_factory(episode_id)
        self.action_scheduler.reset()
        self.policy.reset_state(slot_id=0)
        self.env.reset(seed=seed)
        if self.video_recorder is not None:
            self.video_recorder.start_episode(
                episode_id=episode_id,
                metadata=self.video_episode_metadata[episode_id],
            )
            self.video_recorder.record_frame(self.env.render_game_frame())

        step_records: list[StepRecord] = []
        action_events: list[ActionEvent] = []
        latency_values_ms: list[float] = []
        worker_capacity = SimulatedWorkerCapacity(
            capacity=self.simulated_worker_capacity,
            busy_until_raw_frames=[],
        )
        submitted_observation_frames = 0
        dropped_observation_count = 0
        latency_type = latency_type_from_source(self.latency_source)
        episode_metadata = episode_raw_fact_metadata(
            mode="simulated",
            episode_seed=seed,
            env_fps=self.clock.env_fps,
            obs_fps=self.clock.obs_fps,
            frame_ms=self.clock.frame_ms,
            latency_type=latency_type,
            latency_source=self.latency_source,
        )

        raw_frame = 0
        recent_drop_count = 0
        while raw_frame < eval_max_steps:
            current_time_ms = self.clock.step_to_time_ms(raw_frame)
            worker_capacity.release_ready(raw_frame)
            observation_submitted = False
            observation_dropped = False
            if self.decision_scheduler.should_observe(raw_frame, current_time_ms):
                observation = self.env.observe()
                obs_id = self._next_obs_id
                self._next_obs_id += 1
                if worker_capacity.can_submit():
                    latency_context = {
                        "observation": observation,
                        "obs_id": obs_id,
                        "env_step": raw_frame,
                        "raw_frame": raw_frame,
                        "sim_time_ms": current_time_ms,
                        "episode_id": episode_id,
                        "slot_id": 0,
                        "recent_drop_count": recent_drop_count,
                        "in_flight_count": worker_capacity.in_flight_count,
                        "idle_worker_count": worker_capacity.idle_worker_count,
                    }
                    latency_sample = self.latency_source.sample(latency_context)
                    latency_ms = latency_sample.latency_ms
                    policy_metadata = dict(observation.metadata)
                    policy_metadata.update(
                        {
                            "obs_id": obs_id,
                            "episode_id": episode_id,
                            "slot_id": 0,
                        }
                    )
                    policy_observation = Observation(
                        data=observation.data,
                        env_step=observation.env_step,
                        sim_time_ms=observation.sim_time_ms,
                        metadata=policy_metadata,
                    )
                    policy_output = self.policy.predict(policy_observation)
                    ready_raw_frame = raw_frame + latency_sample.action_ready_raw_frames
                    ready_time_ms = self.clock.step_to_time_ms(ready_raw_frame)
                    worker_capacity.submit(raw_frame + latency_sample.worker_service_raw_frames)
                    latency_record = LatencyRecord(
                        obs_id=obs_id,
                        latency_ms=latency_ms,
                    )
                    profile_metadata = profile_metadata_from_source(self.latency_source)
                    apply_latency_record_raw_fields(
                        record=latency_record,
                        latency_type=latency_type,
                        profile_metadata=profile_metadata,
                    )
                    event_metadata = dict(policy_output.metadata)
                    event_metadata["frame_ms"] = self.clock.frame_ms
                    event_metadata["worker_service_ms"] = latency_sample.worker_service_ms
                    event_metadata["worker_service_raw_frames"] = latency_sample.worker_service_raw_frames
                    event = ActionEvent(
                        action_id=self._next_action_id,
                        action=policy_output.action,
                        obs_env_step=raw_frame,
                        ready_time_ms=ready_time_ms,
                        ready_step_float=float(ready_raw_frame),
                        latency=latency_record,
                        latency_raw_frames=latency_sample.action_ready_raw_frames,
                        ready_raw_frame=ready_raw_frame,
                        metadata=event_metadata,
                    )
                    apply_latency_raw_fields(
                        event=event,
                        latency_type=latency_type,
                        frame_ms=self.clock.frame_ms,
                        profile_metadata=profile_metadata,
                    )
                    self._next_action_id += 1
                    self.action_scheduler.enqueue(event)
                    action_events.append(event)
                    latency_values_ms.append(latency_ms)
                    submitted_observation_frames += 1
                    observation_submitted = True
                    recent_drop_count = 0
                    if self.logger is not None:
                        self.logger.log_action_event(event)
                        self.logger.log_latency(latency_record)
                else:
                    dropped_observation_count += 1
                    observation_dropped = True
                    recent_drop_count += 1

            active_action = self.action_scheduler.update(raw_frame, current_time_ms)
            result = self.env.step(active_action)
            soft_reset = bool(result.info.get("soft_reset")) if isinstance(result.info, dict) else False
            episode_done = result.done or result.truncated and not soft_reset
            if self.video_recorder is not None and not episode_done:
                self.video_recorder.record_frame(self.env.render_game_frame())
            step_info = step_record_info(result.info)
            if self.simulated_worker_capacity is not None:
                step_info.update(
                    {
                        "observation_submitted": observation_submitted,
                        "observation_dropped": observation_dropped,
                        "in_flight_count": worker_capacity.in_flight_count,
                        "idle_worker_count": worker_capacity.idle_worker_count,
                    }
                )
            step_info.update(
                active_action_step_info(
                    active_event=self.action_scheduler.latest_applied_event,
                    env_step=raw_frame,
                    scheduled_time_ms=current_time_ms,
                    frame_ms=self.clock.frame_ms,
                    latency_type=latency_type,
                    profile_metadata=profile_metadata_from_source(self.latency_source),
                )
            )
            record = StepRecord(
                episode_id=episode_id,
                env_step=raw_frame,
                sim_time_ms=current_time_ms,
                active_action=active_action,
                reward=float(result.reward),
                done=episode_done,
                info=step_info,
            )
            step_records.append(record)
            if self.logger is not None:
                self.logger.log_step(record)
            if soft_reset:
                self.action_scheduler.reset()
                self.policy.reset_state(slot_id=0)
                worker_capacity.reset()
                recent_drop_count = 0

            raw_frame += 1
            if episode_done:
                break

        if self.simulated_worker_capacity is not None:
            episode_metadata.update(
                {
                    "submitted_observation_frames": submitted_observation_frames,
                    "dropped_observation_count": dropped_observation_count,
                    "simulated_worker_capacity": self.simulated_worker_capacity,
                    "inference_worker_count": self.simulated_worker_capacity,
                    "in_flight_count": worker_capacity.in_flight_count,
                    "idle_worker_count": worker_capacity.idle_worker_count,
                }
            )
        metrics = compute_episode_metrics(
            episode_id=episode_id,
            step_records=step_records,
            action_events=action_events,
            latency_values_ms=latency_values_ms,
            metadata=episode_metadata,
            frame_ms=self.clock.frame_ms,
        )
        if self.logger is not None:
            self.logger.log_episode_metrics(metrics)
        if self.video_recorder is not None:
            self.video_recorder.close_episode({"episode_return": metrics.episode_return})
        return metrics

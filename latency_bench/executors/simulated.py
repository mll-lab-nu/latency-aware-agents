from __future__ import annotations

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ActionEvent, EpisodeMetrics, LatencyRecord, StepRecord
from latency_bench.envs.base import EnvAdapter
from latency_bench.executors._simulated_timeline import sample_simulated_latency
from latency_bench.latency.sources import LatencySource
from latency_bench.logging.metrics import (
    active_action_step_info,
    apply_latency_raw_fields,
    apply_latency_record_raw_fields,
    compute_episode_metrics,
    episode_raw_fact_metadata,
    latency_type_from_source,
    profile_metadata_from_source,
)
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.base import PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import DecisionScheduler


class SimulatedLatencyExecutor:
    def __init__(
        self,
        env: EnvAdapter,
        policy: PolicyRunner,
        decision_scheduler: DecisionScheduler,
        latency_source: LatencySource,
        action_scheduler: ActionScheduler,
        clock: EnvClock,
        logger: TrajectoryLogger | None = None,
        observation_builder=None,
        history=None,
    ):
        self.env = env
        self.policy = policy
        self.decision_scheduler = decision_scheduler
        self.latency_source = latency_source
        self.action_scheduler = action_scheduler
        self.clock = clock
        self.logger = logger
        self._next_obs_id = 0
        self._next_action_id = 0

    def run_episode(
        self,
        episode_id: int,
        seed: int | None = None,
        eval_max_steps: int = 10000,
    ) -> EpisodeMetrics:
        if eval_max_steps < 0:
            raise ValueError("eval_max_steps must be non-negative")

        self.action_scheduler.reset()
        self.policy.reset_state()
        self.env.reset(seed=seed)

        step_records: list[StepRecord] = []
        action_events: list[ActionEvent] = []
        latency_values_ms: list[float] = []
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
        while raw_frame < eval_max_steps:
            current_time_ms = self.clock.step_to_time_ms(raw_frame)
            if self.decision_scheduler.should_observe(raw_frame, current_time_ms):
                observation = self.env.observe()
                policy_output = self.policy.predict(observation)
                obs_id = self._next_obs_id
                self._next_obs_id += 1

                latency_context = {
                    "observation": observation,
                    "obs_id": obs_id,
                    "env_step": raw_frame,
                    "raw_frame": raw_frame,
                    "sim_time_ms": current_time_ms,
                }
                latency_sample = sample_simulated_latency(
                    latency_source=self.latency_source,
                    clock=self.clock,
                    context=latency_context,
                )
                latency_ms = latency_sample.latency_ms
                ready_raw_frame = raw_frame + latency_sample.latency_raw_frames
                ready_time_ms = self.clock.step_to_time_ms(ready_raw_frame)
                event = ActionEvent(
                    action_id=self._next_action_id,
                    obs_id=obs_id,
                    action=policy_output.action,
                    obs_env_step=raw_frame,
                    obs_time_ms=current_time_ms,
                    latency_ms=latency_ms,
                    ready_time_ms=ready_time_ms,
                    ready_step_float=float(ready_raw_frame),
                    obs_raw_frame=raw_frame,
                    latency_raw_frames=latency_sample.latency_raw_frames,
                    ready_raw_frame=ready_raw_frame,
                    metadata=policy_output.metadata,
                )
                profile_metadata = profile_metadata_from_source(self.latency_source)
                apply_latency_raw_fields(
                    event=event,
                    latency_type=latency_type,
                    frame_ms=self.clock.frame_ms,
                    latency_ms_sampled=latency_ms,
                    latency_ms_measured=None,
                    profile_metadata=profile_metadata,
                )
                self._next_action_id += 1
                self.action_scheduler.enqueue(event)
                action_events.append(event)
                latency_values_ms.append(latency_ms)
                if self.logger is not None:
                    self.logger.log_action_event(event)
                    latency_record = LatencyRecord(
                        obs_id=obs_id,
                        latency_ms=latency_ms,
                        source=self.latency_source.__class__.__name__,
                    )
                    apply_latency_record_raw_fields(
                        record=latency_record,
                        latency_type=latency_type,
                        latency_ms_sampled=latency_ms,
                        latency_ms_measured=None,
                        profile_metadata=profile_metadata,
                    )
                    self.logger.log_latency(latency_record)

            active_action = self.action_scheduler.update(raw_frame, current_time_ms)
            result = self.env.step(active_action)
            soft_reset = bool(result.info.get("soft_reset")) if isinstance(result.info, dict) else False
            episode_done = bool(result.done or result.truncated) and not soft_reset
            step_info = dict(result.info)
            step_info.update(
                active_action_step_info(
                    applied_events=self.action_scheduler.get_applied_events(),
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
                self.policy.reset_state()

            raw_frame += 1
            if episode_done:
                break

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
        return metrics

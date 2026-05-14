from __future__ import annotations

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ActionEvent, EpisodeMetrics, LatencyRecord, StepRecord
from latency_bench.envs.base import EnvAdapter
from latency_bench.latency.sources import LatencySource
from latency_bench.logging.metrics import (
    active_action_step_info,
    apply_latency_raw_fields,
    apply_latency_record_raw_fields,
    compute_episode_metrics,
    episode_raw_fact_metadata,
)
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.base import PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import DecisionScheduler


class PausedLatencyExecutor:
    def __init__(
        self,
        env: EnvAdapter,
        policy: PolicyRunner,
        decision_scheduler: DecisionScheduler,
        latency_source: LatencySource,
        action_scheduler: ActionScheduler,
        clock: EnvClock,
        logger: TrajectoryLogger | None = None,
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
        self.env.reset(seed=seed)

        step_records: list[StepRecord] = []
        action_events: list[ActionEvent] = []
        latency_values_ms: list[float] = []
        latency_type = "zero"
        episode_metadata = episode_raw_fact_metadata(
            mode="paused",
            episode_seed=seed,
            env_fps=self.clock.env_fps,
            obs_fps=self.clock.obs_fps,
            frame_ms=self.clock.frame_ms,
            latency_type=latency_type,
        )

        env_step = 0
        while env_step < eval_max_steps:
            current_time_ms = self.clock.step_to_time_ms(env_step)
            if self.decision_scheduler.should_observe(env_step, current_time_ms):
                observation = self.env.observe()
                policy_output = self.policy.predict(observation)
                obs_id = self._next_obs_id
                self._next_obs_id += 1

                latency_record = LatencyRecord(
                    obs_id=obs_id,
                    latency_ms=0.0,
                )
                apply_latency_record_raw_fields(
                    record=latency_record,
                    latency_type=latency_type,
                    profile_metadata={},
                )
                ready_time_ms = current_time_ms
                event = ActionEvent(
                    action_id=self._next_action_id,
                    action=policy_output.action,
                    obs_env_step=observation.env_step,
                    ready_time_ms=ready_time_ms,
                    ready_step_float=self.clock.time_ms_to_step_float(ready_time_ms),
                    latency=latency_record,
                    metadata=policy_output.metadata,
                )
                apply_latency_raw_fields(
                    event=event,
                    latency_type=latency_type,
                    frame_ms=self.clock.frame_ms,
                    profile_metadata={},
                )
                self._next_action_id += 1
                self.action_scheduler.enqueue(event)
                action_events.append(event)
                latency_values_ms.append(latency_record.latency_ms)
                if self.logger is not None:
                    self.logger.log_action_event(event)
                    self.logger.log_latency(latency_record)

            active_action = self.action_scheduler.update(env_step, current_time_ms)
            result = self.env.step(active_action)
            step_info = dict(result.info)
            step_info.update(
                active_action_step_info(
                    applied_events=self.action_scheduler.get_applied_events(),
                    env_step=env_step,
                    scheduled_time_ms=current_time_ms,
                    frame_ms=self.clock.frame_ms,
                    latency_type=latency_type,
                    profile_metadata={},
                )
            )
            record = StepRecord(
                episode_id=episode_id,
                env_step=env_step,
                sim_time_ms=current_time_ms,
                active_action=active_action,
                reward=float(result.reward),
                done=bool(result.done or result.truncated),
                info=step_info,
            )
            step_records.append(record)
            if self.logger is not None:
                self.logger.log_step(record)
            env_step += 1
            if result.done or result.truncated:
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

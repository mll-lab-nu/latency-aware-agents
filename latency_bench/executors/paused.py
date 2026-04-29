from __future__ import annotations

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ActionEvent, EpisodeMetrics, LatencyRecord, StepRecord
from latency_bench.envs.base import EnvAdapter
from latency_bench.latency.sources import LatencySource
from latency_bench.logging.metrics import compute_episode_metrics
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.observation.builder import ObservationBuilder
from latency_bench.observation.history import HistoryBuffer
from latency_bench.policy.base import PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import DecisionScheduler


class PausedLatencyExecutor:
    def __init__(
        self,
        env: EnvAdapter,
        policy: PolicyRunner,
        observation_builder: ObservationBuilder,
        history: HistoryBuffer,
        decision_scheduler: DecisionScheduler,
        latency_source: LatencySource,
        action_scheduler: ActionScheduler,
        clock: EnvClock,
        logger: TrajectoryLogger | None = None,
    ):
        self.env = env
        self.policy = policy
        self.observation_builder = observation_builder
        self.history = history
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
        max_env_steps: int = 10000,
    ) -> EpisodeMetrics:
        if max_env_steps < 0:
            raise ValueError("max_env_steps must be non-negative")

        self.history.clear()
        self.action_scheduler.reset()
        initial_observation = self.env.reset(seed=seed)
        self.history.append(initial_observation)

        step_records: list[StepRecord] = []
        action_events: list[ActionEvent] = []
        latency_values_ms: list[float] = []

        env_step = 0
        while env_step < max_env_steps:
            current_time_ms = self.clock.step_to_time_ms(env_step)
            if self.decision_scheduler.should_observe(env_step, current_time_ms):
                observation = self.observation_builder.build(self.env, self.history)
                policy_output = self.policy.predict(observation)
                obs_id = self._next_obs_id
                self._next_obs_id += 1

                configured_latency_ms = float(self.latency_source.next_latency_ms({
                    "observation": observation,
                    "obs_id": obs_id,
                    "env_step": env_step,
                    "sim_time_ms": current_time_ms,
                }))
                effective_latency_ms = 0.0
                ready_time_ms = current_time_ms
                event = ActionEvent(
                    action_id=self._next_action_id,
                    obs_id=obs_id,
                    action=policy_output.action,
                    obs_env_step=observation.env_step,
                    obs_time_ms=observation.sim_time_ms,
                    latency_ms=effective_latency_ms,
                    ready_time_ms=ready_time_ms,
                    ready_step_float=self.clock.time_ms_to_step_float(ready_time_ms),
                    metadata={
                        **policy_output.metadata,
                        "paused_configured_latency_ms": configured_latency_ms,
                    },
                )
                self._next_action_id += 1
                self.action_scheduler.enqueue(event)
                action_events.append(event)
                latency_values_ms.append(configured_latency_ms)
                if self.logger is not None:
                    self.logger.log_action_event(event)
                    self.logger.log_latency(LatencyRecord(
                        obs_id=obs_id,
                        latency_ms=configured_latency_ms,
                        source=self.latency_source.__class__.__name__,
                    ))

            active_action = self.action_scheduler.update(env_step, current_time_ms)
            result = self.env.step(active_action)
            record = StepRecord(
                episode_id=episode_id,
                env_step=env_step,
                sim_time_ms=current_time_ms,
                active_action=active_action,
                reward=float(result.reward),
                done=bool(result.done or result.truncated),
                info=result.info,
            )
            step_records.append(record)
            if self.logger is not None:
                self.logger.log_step(record)
            if result.observation is not None:
                self.history.append(result.observation)

            env_step += 1
            if result.done or result.truncated:
                break

        metrics = compute_episode_metrics(
            episode_id=episode_id,
            step_records=step_records,
            action_events=action_events,
            latency_values_ms=latency_values_ms,
        )
        if self.logger is not None:
            self.logger.log_episode_metrics(metrics)
        return metrics

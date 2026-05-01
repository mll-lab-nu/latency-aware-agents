from __future__ import annotations

import json
import time
from pathlib import Path

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, Observation, PolicyOutput, StaleActionMode
from latency_bench.envs.base import DummyEnvAdapter
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.latency.sources import FixedLatencySource, TraceReplayLatencySource, ZeroLatencySource
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.observation.builder import ObservationBuilder
from latency_bench.observation.history import HistoryBuffer
from latency_bench.policy.base import ConstantPolicyRunner, PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import EveryFrameDecisionScheduler
from latency_bench.scheduler.persistence import HoldPreviousActionPolicy


class CountingPolicyRunner(PolicyRunner):
    def __init__(self):
        self.count = 0

    def predict(self, observation: Observation) -> PolicyOutput:
        action = Action(value=f"a{self.count}", name=f"a{self.count}")
        self.count += 1
        return PolicyOutput(action=action)


def _executor(
    latency_source,
    policy: PolicyRunner | None = None,
    max_steps: int = 5,
    logger: TrajectoryLogger | None = None,
    stale_action_mode: StaleActionMode = StaleActionMode.LATEST_READY,
) -> tuple[SimulatedLatencyExecutor, DummyEnvAdapter, EnvClock, Action]:
    noop = Action("noop", is_noop=True)
    env = DummyEnvAdapter(fps_env=30, max_steps=max_steps, noop_action=noop)
    clock = EnvClock(30)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop,
        persistence_policy=HoldPreviousActionPolicy(),
        stale_action_mode=stale_action_mode,
    )
    executor = SimulatedLatencyExecutor(
        env=env,
        policy=policy or ConstantPolicyRunner(Action("jump")),
        observation_builder=ObservationBuilder(),
        history=HistoryBuffer(),
        decision_scheduler=EveryFrameDecisionScheduler(),
        latency_source=latency_source,
        action_scheduler=action_scheduler,
        clock=clock,
        logger=logger,
    )
    return executor, env, clock, noop


def test_zero_latency_action_can_affect_same_frame():
    executor, env, _, _ = _executor(ZeroLatencySource(), max_steps=3)

    metrics = executor.run_episode(episode_id=0, max_env_steps=3)

    assert env.applied_actions[0].value == "jump"
    assert metrics.survival_steps == 3
    assert metrics.episode_return == 3.0


def test_fixed_two_frame_latency_applies_at_t_plus_two():
    clock = EnvClock(30)
    executor, env, _, noop = _executor(FixedLatencySource(clock.frame_ms * 2), max_steps=4)

    executor.run_episode(episode_id=0, max_env_steps=4)

    assert [action.value for action in env.applied_actions[:3]] == [noop.value, noop.value, "jump"]


def test_trace_replay_latency_applies_event_at_trace_ready_step(tmp_path: Path):
    clock = EnvClock(30)
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        '{"obs_id": 0, "latency_ms": 0.0}\n'
        f'{{"obs_id": 1, "latency_ms": {clock.frame_ms * 2}}}\n'
        f'{{"obs_id": 2, "latency_ms": {clock.frame_ms}}}\n',
        encoding="utf-8",
    )
    executor, _, _, _ = _executor(
        TraceReplayLatencySource(trace_path, loop_trace=True),
        policy=CountingPolicyRunner(),
        max_steps=4,
        stale_action_mode=StaleActionMode.FIFO,
    )

    executor.run_episode(episode_id=0, max_env_steps=4)
    applied = executor.action_scheduler.get_applied_events()

    assert applied[0].action.value == "a0"
    assert applied[0].applied_env_step == 0
    assert applied[1].action.value == "a1"
    assert applied[1].applied_env_step == 3


def test_simulated_executor_does_not_sleep(monkeypatch):
    def fail_sleep(seconds):
        raise AssertionError("SimulatedLatencyExecutor must not sleep")

    monkeypatch.setattr(time, "sleep", fail_sleep)
    executor, env, _, _ = _executor(ZeroLatencySource(), max_steps=2)

    executor.run_episode(episode_id=0, max_env_steps=2)

    assert len(env.applied_actions) == 2


def test_simulated_executor_logs_records_and_metrics(tmp_path: Path):
    logger = TrajectoryLogger(tmp_path)
    executor, _, _, _ = _executor(ZeroLatencySource(), max_steps=3, logger=logger)

    metrics = executor.run_episode(episode_id=7, max_env_steps=3)
    logger.flush()

    assert metrics.episode_id == 7
    assert (tmp_path / "steps.jsonl").exists()
    assert (tmp_path / "actions.jsonl").exists()
    assert (tmp_path / "latencies.jsonl").exists()
    assert (tmp_path / "episode_metrics.jsonl").exists()
    step_lines = (tmp_path / "steps.jsonl").read_text(encoding="utf-8").strip().splitlines()
    metrics_line = json.loads((tmp_path / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert len(step_lines) == 3
    assert metrics_line["episode_id"] == 7
    assert metrics_line["survival_steps"] == 3


def test_each_episode_resets_scheduler_to_noop_for_fixed_latency():
    clock = EnvClock(30)
    executor, env, _, noop = _executor(FixedLatencySource(clock.frame_ms * 2), max_steps=3)

    executor.run_episode(episode_id=0, max_env_steps=3)
    executor.run_episode(episode_id=1, max_env_steps=3)

    assert env.applied_actions[0].value == noop.value

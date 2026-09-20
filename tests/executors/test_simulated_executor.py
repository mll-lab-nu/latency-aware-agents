from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, Observation, OrderingPolicy, PolicyOutput, StepResult
from latency_bench.envs.base import DummyEnvAdapter, EnvAdapter
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.latency.sample import LatencySample
from latency_bench.latency.samplers import FixedSampler, LatencySampler, ZeroSampler
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.base import ConstantPolicyRunner, PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import DecisionScheduler, ObservationCadenceDecisionScheduler
from latency_bench.scheduler.persistence import ActionPersistencePolicy, HoldPolicy, OneFrameThenNoopPolicy
from latency_bench.utils.io import load_jsonl_records


def _zero_sampler() -> ZeroSampler:
    return ZeroSampler(clock=EnvClock(env_fps=30, obs_fps=30))


class CountingPolicyRunner(PolicyRunner):
    def __init__(self):
        self.count = 0

    def predict(self, observation: Observation) -> PolicyOutput:
        action = Action(value=f"a{self.count}", name=f"a{self.count}")
        self.count += 1
        return PolicyOutput(action=action)


class RawFrameLatencySource(LatencySampler):
    latency_type = "fixed"

    def __init__(self, raw_frames: list[int], *, env_fps: float = 30.0):
        self.raw_frames = list(raw_frames)
        self.frame_ms = 1000.0 / env_fps
        self.calls: list[dict] = []

    def sample(self, context=None) -> LatencySample:
        self.calls.append(dict(context or {}))
        if not self.raw_frames:
            raise AssertionError("raw-frame latency source exhausted")
        raw = self.raw_frames.pop(0)
        return LatencySample(
            latency_ms=raw * self.frame_ms,
            latency_raw_frames=raw,
            action_ready_raw_frames=raw,
            worker_service_raw_frames=raw,
            worker_service_ms=raw * self.frame_ms,
        )


class WorkerServiceStructuredLatencySource(LatencySampler):
    latency_type = "profile_sample"

    def __init__(self):
        self.contexts: list[dict] = []

    def sample(self, context=None) -> LatencySample:
        self.contexts.append(dict(context or {}))
        return LatencySample(
            latency_ms=10.0,
            latency_raw_frames=1,
            action_ready_raw_frames=1,
            worker_service_raw_frames=3,
            worker_service_ms=30.0,
        )


class SelectedRawFrameDecisionScheduler(DecisionScheduler):
    def __init__(self, raw_frames: set[int]):
        self.raw_frames = set(raw_frames)

    def should_observe(self, env_step: int, sim_time_ms: float) -> bool:
        return int(env_step) in self.raw_frames


class ResetTrackingPolicyRunner(PolicyRunner):
    def __init__(self, action: Action):
        self.action = action
        self.reset_calls: list[int | None] = []

    def predict(self, observation: Observation) -> PolicyOutput:
        return PolicyOutput(action=self.action)

    def reset_state(self, slot_id: int | None = None) -> None:
        self.reset_calls.append(slot_id)


class SoftResetEnvAdapter(EnvAdapter):
    def __init__(self, *, noop_action: Action):
        self.env_fps = 30.0
        self.noop_action = noop_action
        self.env_step = 0
        self.total_steps = 0
        self.applied_actions: list[Action] = []

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.env_fps

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        self.total_steps = 0
        self.applied_actions.clear()
        return self._observation(seed=seed)

    def step(self, action: Action) -> StepResult:
        self.applied_actions.append(action)
        self.total_steps += 1
        if self.total_steps == 2:
            self.env_step = 0
            return StepResult(
                observation=self._observation(phase="soft_reset"),
                reward=1.0,
                done=False,
                truncated=False,
                info={"soft_reset": True},
            )

        self.env_step += 1
        return StepResult(
            observation=self._observation(),
            reward=1.0,
            done=self.total_steps >= 4,
            truncated=False,
            info={"total_steps": self.total_steps},
        )

    def render_frame(self):
        return {"env_step": self.env_step, "total_steps": self.total_steps}

    def get_state_info(self):
        return {
            "env_step": self.env_step,
            "sim_time_ms": self.env_step * self.frame_ms,
            "total_steps": self.total_steps,
        }

    def _observation(self, **metadata) -> Observation:
        return Observation(
            data={"env_step": self.env_step, "total_steps": self.total_steps},
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata=metadata,
        )


class ObserveOnlyPolicyInputEnv(EnvAdapter):
    def __init__(self, *, noop_action: Action, max_steps: int = 1):
        self.env_fps = 30.0
        self.noop_action = noop_action
        self.max_steps = int(max_steps)
        self.env_step = 0
        self.observe_calls = 0

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.env_fps

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        return Observation(
            data={"source": "reset"},
            env_step=0,
            sim_time_ms=0.0,
            metadata={"seed": seed},
        )

    def observe(self) -> Observation:
        self.observe_calls += 1
        return Observation(
            data={"source": "observe", "observe_calls": self.observe_calls},
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata={"source": "observe"},
        )

    def step(self, action: Action) -> StepResult:
        self.env_step += 1
        return StepResult(
            observation=Observation(
                data={"source": "step"},
                env_step=self.env_step,
                sim_time_ms=self.env_step * self.frame_ms,
                metadata={"source": "step"},
            ),
            reward=1.0,
            done=self.env_step >= self.max_steps,
            truncated=False,
            info={"env_step": self.env_step},
        )


class RawRgbStepInfoEnv(EnvAdapter):
    def __init__(self, *, noop_action: Action):
        self.env_fps = 30.0
        self.noop_action = noop_action
        self.env_step = 0

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.env_fps

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        return self._observation()

    def observe(self) -> Observation:
        return self._observation()

    def step(self, action: Action) -> StepResult:
        self.env_step += 1
        return StepResult(
            observation=self._observation(),
            reward=1.0,
            done=True,
            truncated=False,
            info={
                "env_raw_rgb_frame_stack": np.full((1, 2, 3, 3), 7, dtype=np.uint8),
                "score": 5,
            },
        )

    def render_frame(self):
        return {"env_step": self.env_step}

    def get_state_info(self):
        return {"env_step": self.env_step, "sim_time_ms": self.env_step * self.frame_ms}

    def _observation(self) -> Observation:
        return Observation(
            data={"env_step": self.env_step},
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata={
                "env_raw_rgb_frame_stack": np.full((1, 2, 3, 3), 9, dtype=np.uint8),
            },
        )


class CapturingPolicyRunner(PolicyRunner):
    def __init__(self, action: Action):
        self.action = action
        self.observations: list[Observation] = []

    def predict(self, observation: Observation) -> PolicyOutput:
        self.observations.append(observation)
        return PolicyOutput(action=self.action)


def _executor(
    latency_source,
    policy: PolicyRunner | None = None,
    max_steps: int = 5,
    logger: TrajectoryLogger | None = None,
    ordering_policy: OrderingPolicy = OrderingPolicy.LATEST_READY,
    env_fps: float = 30,
    obs_fps: float | None = None,
    decision_scheduler: DecisionScheduler | None = None,
    persistence_policy: ActionPersistencePolicy | None = None,
    simulated_worker_capacity: int | None = None,
) -> tuple[SimulatedLatencyExecutor, DummyEnvAdapter, EnvClock, Action]:
    noop = Action("noop", is_noop=True)
    env = DummyEnvAdapter(env_fps=env_fps, max_steps=max_steps, noop_action=noop)
    clock = EnvClock(env_fps=env_fps, obs_fps=obs_fps if obs_fps is not None else env_fps)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop,
        persistence_policy=persistence_policy or HoldPolicy(),
        ordering_policy=ordering_policy,
    )
    executor = SimulatedLatencyExecutor(
        env=env,
        policy=policy or ConstantPolicyRunner(Action("jump")),
        decision_scheduler=decision_scheduler or ObservationCadenceDecisionScheduler(clock),
        latency_source=latency_source,
        action_scheduler=action_scheduler,
        clock=clock,
        logger=logger,
        simulated_worker_capacity=simulated_worker_capacity,
    )
    return executor, env, clock, noop


def test_policy_input_comes_from_env_observe_not_step_history():
    noop = Action("noop", is_noop=True)
    env = ObserveOnlyPolicyInputEnv(noop_action=noop)
    policy = CapturingPolicyRunner(Action("jump"))
    clock = EnvClock(env_fps=30, obs_fps=30)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop,
        persistence_policy=OneFrameThenNoopPolicy(),
        ordering_policy=OrderingPolicy.LATEST_READY,
    )
    executor = SimulatedLatencyExecutor(
        env=env,
        policy=policy,
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_source=_zero_sampler(),
        action_scheduler=action_scheduler,
        clock=clock,
    )

    executor.run_episode(episode_id=0, eval_max_steps=1)

    assert env.observe_calls == 1
    assert policy.observations[0].data == {"source": "observe", "observe_calls": 1}


def test_observe_cadence_skips_raw_step_observations():
    noop = Action("noop", is_noop=True)
    env = ObserveOnlyPolicyInputEnv(noop_action=noop, max_steps=4)
    policy = CapturingPolicyRunner(Action("jump"))
    clock = EnvClock(env_fps=30, obs_fps=10)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop,
        persistence_policy=OneFrameThenNoopPolicy(),
        ordering_policy=OrderingPolicy.LATEST_READY,
    )
    executor = SimulatedLatencyExecutor(
        env=env,
        policy=policy,
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_source=_zero_sampler(),
        action_scheduler=action_scheduler,
        clock=clock,
    )

    executor.run_episode(episode_id=0, eval_max_steps=4)

    assert [observation.env_step for observation in policy.observations] == [0, 3]
    assert [observation.data["observe_calls"] for observation in policy.observations] == [1, 2]


def test_zero_latency_action_can_affect_same_frame():
    executor, env, _, _ = _executor(_zero_sampler(), max_steps=3)

    metrics = executor.run_episode(episode_id=0, eval_max_steps=3)

    assert env.applied_actions[0].value == "jump"
    applied = executor.action_scheduler.get_applied_events()
    assert applied[0].obs_env_step == 0
    assert applied[0].latency_raw_frames == 0
    assert applied[0].ready_raw_frame == 0
    assert applied[0].apply_raw_frame == 0
    assert metrics.survival_steps == 3
    assert metrics.episode_return == 3.0


def test_raw_frame_latency_source_and_observation_stride_share_raw_timeline():
    latency_source = RawFrameLatencySource([3, 3], env_fps=60)
    executor, _, _, _ = _executor(
        latency_source,
        policy=CountingPolicyRunner(),
        max_steps=8,
        env_fps=60,
        obs_fps=10,
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
    )

    executor.run_episode(episode_id=0, eval_max_steps=8)

    events = executor.action_scheduler.get_applied_events() + executor.action_scheduler.get_pending_events()
    assert [event.obs_env_step for event in events] == [0, 6]
    assert events[0].latency_raw_frames == 3
    assert events[0].ready_raw_frame == 3
    assert events[0].apply_raw_frame == 3
    assert events[1].latency_raw_frames == 3
    assert events[1].ready_raw_frame == 9
    assert events[1].apply_raw_frame is None
    assert [call["raw_frame"] for call in latency_source.calls] == [0, 6]


@pytest.mark.parametrize("delay", [1, 3, 5])
@pytest.mark.parametrize("prefix_mode", ["none", "profile_mean"])
def test_policy_observation_does_not_leak_sampled_latency(delay, prefix_mode):
    # Realtime execution cannot know an action's latency before inference runs,
    # so simulated replay must not reveal it either: leaking the sampled value
    # changed the StarVLA prompt per step and skewed sim-vs-real returns.
    policy = CapturingPolicyRunner(Action([1.0]))
    executor, _, _, _ = _executor(
        RawFrameLatencySource([delay]),
        policy=policy,
        max_steps=6,
        decision_scheduler=SelectedRawFrameDecisionScheduler({0}),
    )

    executor.action_scheduler.noop_action = Action([7.0])
    executor.action_prefix = {"mode": prefix_mode, "frames": 3, "action_horizon": 8, "action_dim": 1}
    executor.run_episode(episode_id=0, eval_max_steps=6)

    assert policy.observations[0].metadata["obs_id"] == 0
    assert "latency_ms" not in policy.observations[0].metadata
    assert "latency_raw_frames" not in policy.observations[0].metadata
    metadata = policy.observations[0].metadata
    if prefix_mode == "none":
        assert "action_prefix" not in metadata
        assert "action_prefix_mask" not in metadata
    else:
        np.testing.assert_array_equal(metadata["action_prefix"][:3, 0], [7, 7, 7])
        np.testing.assert_array_equal(metadata["action_prefix_mask"], np.arange(8) < 3)
    event = executor.action_scheduler.get_applied_events()[0]
    assert event.latency.obs_id == policy.observations[0].metadata["obs_id"]
    assert event.latency_raw_frames == delay


def test_single_simulated_executor_adds_slot_id_to_policy_observation_and_latency_context():
    latency_source = RawFrameLatencySource([3])
    policy = CapturingPolicyRunner(Action("jump"))
    executor, _, _, _ = _executor(
        latency_source,
        policy=policy,
        max_steps=4,
        decision_scheduler=SelectedRawFrameDecisionScheduler({0}),
    )

    executor.run_episode(episode_id=7, eval_max_steps=4)

    assert policy.observations[0].metadata["slot_id"] == 0
    assert policy.observations[0].metadata["episode_id"] == 7
    assert latency_source.calls[0]["slot_id"] == 0


def test_simulated_worker_capacity_drops_observations_when_worker_is_busy():
    latency_source = RawFrameLatencySource([3, 3])
    policy = CountingPolicyRunner()
    executor, _, _, _ = _executor(
        latency_source,
        policy=policy,
        max_steps=5,
        decision_scheduler=SelectedRawFrameDecisionScheduler({0, 1, 2, 3}),
        simulated_worker_capacity=1,
    )

    metrics = executor.run_episode(episode_id=0, eval_max_steps=5)

    events = executor.action_scheduler.get_applied_events() + executor.action_scheduler.get_pending_events()
    assert [event.obs_env_step for event in events] == [0, 3]
    assert [call["raw_frame"] for call in latency_source.calls] == [0, 3]
    assert policy.count == 2
    assert metrics.metadata["submitted_observation_frames"] == 2
    assert metrics.metadata["dropped_observation_count"] == 2
    assert metrics.metadata["simulated_worker_capacity"] == 1
    assert metrics.metadata["inference_worker_count"] == 1


def test_simulated_worker_capacity_uses_worker_service_frames_and_passes_drop_state():
    latency_source = WorkerServiceStructuredLatencySource()
    executor, _, _, _ = _executor(
        latency_source,
        policy=CountingPolicyRunner(),
        max_steps=5,
        env_fps=100,
        decision_scheduler=SelectedRawFrameDecisionScheduler({0, 1, 3}),
        simulated_worker_capacity=1,
    )

    metrics = executor.run_episode(episode_id=0, eval_max_steps=5)

    events = executor.action_scheduler.get_applied_events() + executor.action_scheduler.get_pending_events()
    assert [event.obs_env_step for event in events] == [0, 3]
    assert [event.ready_raw_frame for event in events] == [1, 4]
    assert [event.metadata["worker_service_raw_frames"] for event in events] == [3, 3]
    assert [context["raw_frame"] for context in latency_source.contexts] == [0, 3]
    assert [context["recent_drop_count"] for context in latency_source.contexts] == [0, 1]
    assert metrics.metadata["submitted_observation_frames"] == 2
    assert metrics.metadata["dropped_observation_count"] == 1


def test_simulated_executor_omits_raw_rgb_frame_stack_from_logged_step_info(tmp_path: Path):
    noop = Action("noop", is_noop=True)
    env = RawRgbStepInfoEnv(noop_action=noop)
    policy = CapturingPolicyRunner(Action("jump"))
    clock = EnvClock(env_fps=30, obs_fps=30)
    logger = TrajectoryLogger(
        tmp_path,
        save_step_records=True,
        save_action_records=True,
        save_latency_records=True,
    )
    executor = SimulatedLatencyExecutor(
        env=env,
        policy=policy,
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_source=_zero_sampler(),
        action_scheduler=ActionScheduler(
            clock=clock,
            noop_action=noop,
            persistence_policy=OneFrameThenNoopPolicy(),
            ordering_policy=OrderingPolicy.LATEST_READY,
        ),
        clock=clock,
        logger=logger,
    )

    executor.run_episode(episode_id=0, eval_max_steps=1)
    logger.flush()

    assert "env_raw_rgb_frame_stack" in policy.observations[0].metadata
    assert "env_raw_rgb_frame_stack" not in load_jsonl_records(tmp_path / "steps.jsonl")[0]["info"]
    row = json.loads((tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["info"]["score"] == 5
    assert "env_raw_rgb_frame_stack" not in row["info"]


def test_fixed_ms_latency_is_ceil_converted_to_raw_frames():
    executor, _, _, _ = _executor(
        FixedSampler(50.0, clock=EnvClock(env_fps=60, obs_fps=60)), max_steps=4, env_fps=60
    )

    executor.run_episode(episode_id=0, eval_max_steps=4)

    event = executor.action_scheduler.get_applied_events()[0]
    assert event.latency_raw_frames == 3
    assert event.ready_raw_frame == 3
    assert event.ready_time_ms == 50.0


def test_one_frame_then_noop_applies_raw_frame_then_returns_to_noop():
    executor, env, _, noop = _executor(
        RawFrameLatencySource([3]),
        max_steps=5,
        decision_scheduler=SelectedRawFrameDecisionScheduler({0}),
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
        persistence_policy=OneFrameThenNoopPolicy(),
    )

    executor.run_episode(episode_id=0, eval_max_steps=6)

    assert [action.value for action in env.applied_actions] == [
        noop.value,
        noop.value,
        noop.value,
        "jump",
        noop.value,
    ]


def test_hold_policy_keeps_action_after_raw_frame_arrival():
    executor, env, _, noop = _executor(
        RawFrameLatencySource([3]),
        max_steps=10,
        env_fps=30,
        obs_fps=10,
        decision_scheduler=SelectedRawFrameDecisionScheduler({0}),
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
        persistence_policy=HoldPolicy(),
    )

    executor.run_episode(episode_id=0, eval_max_steps=10)

    assert [action.value for action in env.applied_actions] == [noop.value] * 3 + ["jump"] * 7


def test_fifo_raw_frame_blocking_waits_for_head_of_line_ready():
    executor, env, _, noop = _executor(
        RawFrameLatencySource([10, 6]),
        policy=CountingPolicyRunner(),
        max_steps=12,
        decision_scheduler=SelectedRawFrameDecisionScheduler({0, 1}),
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
        persistence_policy=HoldPolicy(),
    )

    executor.run_episode(episode_id=0, eval_max_steps=12)

    assert env.applied_actions[7].value == noop.value
    assert env.applied_actions[10].value == "a0"
    assert env.applied_actions[11].value == "a1"
    applied = executor.action_scheduler.get_applied_events()
    assert [event.action.value for event in applied] == ["a0", "a1"]
    assert [event.ready_raw_frame for event in applied] == [10, 7]
    assert [event.apply_raw_frame for event in applied] == [10, 11]


def test_late_stale_result_is_ignored_not_applied():
    # obs0's 5-frame result is still in flight when obs1's 2-frame result gets
    # released at frame 3; that release permanently voids obs0, so when its
    # result matures alone at frame 5 it is dropped instead of rolling the
    # active action back -- the same rule the realtime coordinator enforces.
    executor, env, _, noop = _executor(
        RawFrameLatencySource([5, 2]),
        policy=CountingPolicyRunner(),
        max_steps=8,
        decision_scheduler=SelectedRawFrameDecisionScheduler({0, 1}),
        persistence_policy=HoldPolicy(),
    )

    metrics = executor.run_episode(episode_id=0, eval_max_steps=8)

    assert [action.value for action in env.applied_actions] == [noop.value] * 3 + ["a1"] * 5
    assert [event.action.value for event in executor.action_scheduler.get_applied_events()] == ["a1"]
    assert metrics.num_dropped_actions == 1


def test_fixed_two_frame_latency_applies_at_t_plus_two():
    clock = EnvClock(env_fps=30, obs_fps=30)
    executor, env, _, noop = _executor(FixedSampler(clock.frame_ms * 2, clock=clock), max_steps=4)

    executor.run_episode(episode_id=0, eval_max_steps=4)

    assert [action.value for action in env.applied_actions[:3]] == [noop.value, noop.value, "jump"]


def test_fixed_one_frame_latency_at_35_fps_does_not_drop_boundary_action():
    clock = EnvClock(env_fps=35, obs_fps=35)
    executor, env, _, noop = _executor(
        FixedSampler(clock.frame_ms, clock=clock),
        policy=CountingPolicyRunner(),
        max_steps=9,
        env_fps=35,
    )

    metrics = executor.run_episode(episode_id=0, eval_max_steps=9)

    assert [action.value for action in env.applied_actions] == [
        noop.value,
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
    ]
    assert metrics.num_dropped_actions == 0


def test_simulated_executor_does_not_sleep(monkeypatch):
    def fail_sleep(seconds):
        raise AssertionError("SimulatedLatencyExecutor must not sleep")

    monkeypatch.setattr(time, "sleep", fail_sleep)
    executor, env, _, _ = _executor(_zero_sampler(), max_steps=2)

    executor.run_episode(episode_id=0, eval_max_steps=2)

    assert len(env.applied_actions) == 2


def test_simulated_executor_logs_records_and_metrics(tmp_path: Path):
    logger = TrajectoryLogger(
        tmp_path,
        save_step_records=True,
        save_action_records=True,
        save_latency_records=True,
    )
    executor, _, _, _ = _executor(_zero_sampler(), max_steps=3, logger=logger)

    metrics = executor.run_episode(episode_id=7, eval_max_steps=3)
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
    clock = EnvClock(env_fps=30, obs_fps=30)
    executor, env, _, noop = _executor(FixedSampler(clock.frame_ms * 2, clock=clock), max_steps=3)

    executor.run_episode(episode_id=0, eval_max_steps=3)
    executor.run_episode(episode_id=1, eval_max_steps=3)

    assert env.applied_actions[0].value == noop.value


def test_soft_reset_continues_same_episode_and_clears_latency_state():
    clock = EnvClock(env_fps=30, obs_fps=30)
    noop = Action("noop", name="noop", is_noop=True)
    env = SoftResetEnvAdapter(noop_action=noop)
    policy = ResetTrackingPolicyRunner(Action("jump", name="jump"))
    executor = SimulatedLatencyExecutor(
        env=env,
        policy=policy,
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_source=FixedSampler(clock.frame_ms * 2, clock=clock),
        action_scheduler=ActionScheduler(
            clock=clock,
            noop_action=noop,
            persistence_policy=HoldPolicy(),
            ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
        ),
        clock=clock,
    )

    metrics = executor.run_episode(episode_id=0, seed=5, eval_max_steps=10)

    assert metrics.survival_steps == 4
    assert metrics.episode_return == 4.0
    assert [action.value for action in env.applied_actions] == ["noop", "noop", "noop", "noop"]
    assert policy.reset_calls == [0, 0]

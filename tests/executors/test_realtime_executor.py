from __future__ import annotations

import gc
import json
import weakref
from types import SimpleNamespace
from dataclasses import dataclass

import numpy as np
import pytest
from gymnasium.spaces import Discrete

import latency_bench.executors.realtime.executor as realtime_executor
import latency_bench.executors.realtime.inference_worker as realtime_inference_worker
import latency_bench.executors.realtime.pool as realtime_pool
from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionEvent, LatencyRecord, Observation, OrderingPolicy, PolicyOutput, StepRecord
from latency_bench.envs.adapters.dummy import DummyEnvAdapter
from latency_bench.executors.realtime.env_worker import (
    _drain_env_action_messages,
    _drain_env_actions_until_wall_time,
)
from latency_bench.executors.realtime.executor import RealtimeLatencyExecutor, RealtimeRuntime
from latency_bench.executors.realtime.inference_worker import (
    _inference_worker,
    _worker_runtime_metadata,
)
from latency_bench.executors.realtime.messages import (
    RealtimeActionDispatch,
    RealtimeObservationTask,
    RealtimePolicyCompletion,
    RealtimePolicyResult,
    _action_event_ack_payload,
    _action_event_dispatch_payload,
    _action_event_from_dispatch_payload,
)
from latency_bench.executors.realtime.pool import ProcessInferencePool
from latency_bench.executors.realtime.shared_observation import ObservationSlotPool, ObservationAttachmentCache
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy.coding_agent import InvalidAgentAction
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler
from latency_bench.scheduler.persistence import HoldPolicy


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = float(now)
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(float(seconds))
        self.now += float(seconds)


@dataclass
class CompletionScript:
    action: Action
    ready_time_ms: float
    worker_id: int
    device: str


class ScriptedInferencePool:
    def __init__(self, scripts: list[CompletionScript], *, num_workers: int):
        self.scripts = list(scripts)
        self.num_workers = int(num_workers)
        self.submitted: list[RealtimeObservationTask] = []
        self.closed = False
        self._busy_worker_ids: set[int] = set()
        self._pending: list[tuple[RealtimeObservationTask, CompletionScript]] = []
        self.episode_start_s = 0.0

    @property
    def in_flight_count(self) -> int:
        return len(self._busy_worker_ids)

    @property
    def idle_worker_count(self) -> int:
        return self.num_workers - len(self._busy_worker_ids)

    @property
    def worker_count(self) -> int:
        return self.num_workers

    def drain_until_idle(self) -> dict[str, float | int]:
        self.episode_start_s = self.time_fn()
        self.submitted.clear()
        self._busy_worker_ids.clear()
        self._pending.clear()
        return {
            "pre_episode_in_flight_count": 0,
            "pre_episode_drained_completion_count": 0,
            "pre_episode_drain_wait_ms": 0.0,
            "pre_episode_in_flight_after_drain": 0,
        }

    def reset_state(self, slot_id: int) -> None:
        del slot_id

    def prepare_episode(self, episode_id: int) -> None:
        del episode_id

    def submit(self, task: RealtimeObservationTask) -> bool:
        idle_worker_ids = [
            worker_id
            for worker_id in range(self.num_workers)
            if worker_id not in self._busy_worker_ids
        ]
        if not idle_worker_ids:
            return False
        if len(self.submitted) >= len(self.scripts):
            return False
        script = self.scripts[len(self.submitted)]
        self.submitted.append(task)
        self._busy_worker_ids.add(script.worker_id)
        self._pending.append((task, script))
        return True

    def drain_completed(self) -> list[RealtimePolicyCompletion]:
        ready: list[RealtimePolicyCompletion] = []
        still_pending: list[tuple[RealtimeObservationTask, CompletionScript]] = []
        for task, script in self._pending:
            if script.ready_time_ms <= (self.time_fn() - self.episode_start_s) * 1000.0:
                self._busy_worker_ids.remove(script.worker_id)
                ready.append(
                    RealtimePolicyCompletion(
                        task=task,
                        policy_output=PolicyOutput(action=script.action),
                        completion_time_ms=script.ready_time_ms,
                        latency_record=LatencyRecord(
                            obs_id=task.obs_id,
                            latency_ms=script.ready_time_ms - task.submit_time_ms,
                            device=script.device,
                            metadata={"worker_id": script.worker_id},
                        ),
                        observation_to_action_latency_ms=(
                            script.ready_time_ms - task.submit_time_ms
                        ),
                        worker_service_time_ms=(
                            script.ready_time_ms - task.submit_time_ms
                        ),
                        worker_id=script.worker_id,
                        device=script.device,

                    )
                )
            else:
                still_pending.append((task, script))
        self._pending = still_pending
        return ready

    def close(self) -> None:
        self.closed = True


def test_process_inference_pool_uses_spawn_start_method(monkeypatch):
    requested_methods: list[str] = []

    class FakeContext:
        pass

    def fake_get_context(start_method: str) -> FakeContext:
        requested_methods.append(start_method)
        return FakeContext()

    monkeypatch.setattr(realtime_pool.mp, "get_context", fake_get_context)
    monkeypatch.setattr(ProcessInferencePool, "_start_workers", lambda self: None)

    pool = ProcessInferencePool(config={}, inference_devices=["cuda:0"])

    assert requested_methods == ["spawn"]
    assert pool.start_method == "spawn"


def test_process_inference_pool_uses_configured_worker_python(monkeypatch):
    monkeypatch.setattr(ProcessInferencePool, "_start_workers", lambda self: None)
    monkeypatch.setattr(
        realtime_pool.subprocess,
        "check_output",
        lambda *args, **kwargs: '["/worker/site-packages"]\n',
    )

    pool = ProcessInferencePool(
        config={
            "policy": {
                "type": "starvla",
                "worker_python_executable": "/venv/starvla/bin/python",
            }
        },
        inference_devices=["cuda:1"],
    )

    assert pool.worker_python_executable == "/venv/starvla/bin/python"
    assert pool.worker_sys_path[-1] == "/worker/site-packages"


def test_process_inference_pool_does_not_accept_start_method(monkeypatch):
    monkeypatch.setattr(ProcessInferencePool, "_start_workers", lambda self: None)

    with pytest.raises(TypeError):
        ProcessInferencePool(config={}, inference_devices=["cuda:0"], start_method="fork")


def test_worker_runtime_metadata_records_spawn_context(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    metadata = _worker_runtime_metadata(worker_id=2, start_method="spawn")

    assert metadata["worker_id"] == 2
    assert metadata["worker_start_method"] == "spawn"
    assert metadata["worker_torch_version"]
    assert type(metadata["worker_torch_version"]) is str
    assert "worker_torch_cuda_version" in metadata
    assert isinstance(metadata["worker_cuda_device_count"], int)
    assert "worker_pid" not in metadata
    assert "worker_python_executable" not in metadata
    assert "worker_cuda_visible_devices" not in metadata
    assert "worker_cuda_flash_sdp_enabled" in metadata
    assert "worker_torch_float32_matmul_precision" in metadata


def test_inference_worker_seeds_before_policy_build_with_worker_offset(monkeypatch):
    import latency_bench.core.actions as actions_mod
    import latency_bench.policy.factory as factory_mod

    class DummyActionResolver:
        def __init__(self, env_config):
            self.env_config = env_config

    events = []

    class CloseWorkerConnection:
        def send(self, _message):
            return None

        def recv(self):
            return "close", {}

    monkeypatch.setattr(actions_mod, "ActionResolver", DummyActionResolver)
    monkeypatch.setattr(
        realtime_inference_worker,
        "seed_everything",
        lambda seed: events.append(("seed", seed)),
    )
    monkeypatch.setattr(
        realtime_inference_worker,
        "_worker_runtime_metadata",
        lambda worker_id, start_method: {},
    )
    monkeypatch.setattr(
        factory_mod,
        "build_policy",
        lambda config, action_resolver: events.append(("build",)) or DummyWorkerPolicy(),
    )
    monkeypatch.setattr(
        realtime_inference_worker,
        "RuntimeLatencyMeasurer",
        SuccessfulWorkerMeasurer,
    )

    config = {"experiment": {"seed": 123}, "env": {}, "policy": {}, "executor": {"sync_cuda": True}}
    for worker_id in (0, 1, 0):
        _inference_worker(
            CloseWorkerConnection(),
            config,
            worker_id=worker_id,
            device="cpu",
            start_method="spawn",
        )

    assert events == [
        ("seed", 123),
        ("build",),
        ("seed", 124),
        ("build",),
        ("seed", 123),
        ("build",),
    ]


def test_inference_worker_selects_cuda_device_before_policy_build(monkeypatch):
    import torch
    import latency_bench.core.actions as actions_mod
    import latency_bench.policy.factory as factory_mod

    events = []

    class Connection:
        def send(self, _message):
            return None

        def recv(self):
            return "close", {}

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: events.append(("device", device)))
    monkeypatch.setattr(actions_mod, "ActionResolver", lambda _config: object())
    monkeypatch.setattr(
        realtime_inference_worker,
        "seed_everything",
        lambda _seed: events.append(("seed",)),
    )
    monkeypatch.setattr(
        realtime_inference_worker,
        "_worker_runtime_metadata",
        lambda _worker_id, _start_method: events.append(("metadata",)) or {},
    )
    monkeypatch.setattr(
        factory_mod,
        "build_policy",
        lambda _config, _resolver: events.append(("build",)) or DummyWorkerPolicy(),
    )
    monkeypatch.setattr(
        realtime_inference_worker,
        "RuntimeLatencyMeasurer",
        SuccessfulWorkerMeasurer,
    )

    _inference_worker(
        Connection(),
        {"experiment": {"seed": 0}, "env": {}, "policy": {}, "executor": {"sync_cuda": True}},
        worker_id=1,
        device="cuda:1",
        start_method="spawn",
    )

    assert events == [("device", "cuda:1"), ("seed",), ("metadata",), ("build",)]


class ClosedActionEventPipe:
    def __init__(self):
        self.closed = False

    def send(self, message):
        command, _payload = message
        if command in {"action_dispatch", "close"}:
            raise BrokenPipeError

    def poll(self, timeout: float | None = None) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


class DummyChildPipe:
    def close(self) -> None:
        return None


class AliveUntilTerminatedProcess:
    def __init__(self, *args, **kwargs):
        self.terminated = False

    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return not self.terminated

    def join(self, timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        self.terminated = True


class ScriptedEnvProcess:
    def __init__(self, connection):
        self.connection = connection
        self.process = AliveUntilTerminatedProcess()

    def start_episode(self, **kwargs):
        return 0.0

    def finish_episode(self):
        self.connection.send(("finish_episode", {}))

    def close(self):
        self.connection.close()
        self.process.terminate()


class QueuedEnvMessagePipe:
    def __init__(self, messages: list[tuple[str, dict]]):
        self.messages = list(messages)
        self.sent: list[tuple[str, dict]] = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def poll(self, timeout: float | None = None) -> bool:
        return bool(self.messages)

    def recv(self):
        return self.messages.pop(0)

    def close(self) -> None:
        self.closed = True


class TimedEnvMessagePipe:
    def __init__(self, messages: list[tuple[str, dict]], clock: FakeClock):
        self.messages = list(messages)
        self.clock = clock
        self.poll_timeouts: list[float | None] = []

    def poll(self, timeout: float | None = None) -> bool:
        self.poll_timeouts.append(timeout)
        if self.messages:
            if timeout is not None:
                self.clock.now += 0.010
            return True
        if timeout is not None:
            self.clock.now += timeout
        return False

    def recv(self):
        return self.messages.pop(0)




class DrainBeforeSubmitPool:
    def __init__(self):
        self.drained = False
        self.submitted: list[RealtimeObservationTask] = []

    @property
    def in_flight_count(self) -> int:
        return 0 if self.drained else 1

    @property
    def idle_worker_count(self) -> int:
        return 1 if self.drained else 0

    @property
    def worker_count(self) -> int:
        return 1

    def drain_until_idle(self) -> dict[str, float | int]:
        return {
            "pre_episode_in_flight_count": 0,
            "pre_episode_drained_completion_count": 0,
            "pre_episode_drain_wait_ms": 0.0,
            "pre_episode_in_flight_after_drain": 0,
        }

    def reset_state(self, slot_id: int) -> None:
        del slot_id

    def prepare_episode(self, episode_id: int) -> None:
        del episode_id

    def submit(self, task: RealtimeObservationTask) -> bool:
        if not self.drained:
            return False
        self.submitted.append(task)
        return True


class LateCompletionAfterFinalizePool:
    def __init__(self):
        self.submitted: list[RealtimeObservationTask] = []
        self.drain_calls = 0
        self.emitted = False

    @property
    def in_flight_count(self) -> int:
        return 1 if self.submitted and not self.emitted else 0

    @property
    def idle_worker_count(self) -> int:
        return 1 - self.in_flight_count

    @property
    def worker_count(self) -> int:
        return 1

    def drain_until_idle(self) -> dict[str, float | int]:
        return {
            "pre_episode_in_flight_count": 0,
            "pre_episode_drained_completion_count": 0,
            "pre_episode_drain_wait_ms": 0.0,
            "pre_episode_in_flight_after_drain": 0,
        }

    def reset_state(self, slot_id: int) -> None:
        del slot_id

    def prepare_episode(self, episode_id: int) -> None:
        del episode_id

    def submit(self, task: RealtimeObservationTask) -> bool:
        self.submitted.append(task)
        return True

    def drain_completed(self) -> list[RealtimePolicyCompletion]:
        self.drain_calls += 1
        if self.drain_calls < 3:
            return []
        self.emitted = True
        task = self.submitted[0]
        return [
            RealtimePolicyCompletion(
                task=task,
                policy_output=PolicyOutput(action=Action("late")),
                completion_time_ms=self.time_fn() * 1000.0,
                latency_record=LatencyRecord(obs_id=task.obs_id, latency_ms=1.0),
                observation_to_action_latency_ms=1.0,
                worker_service_time_ms=1.0,
                worker_id=0,
                device="cpu",

            )
        ]


class CloseOrderConnection:
    def __init__(self, events: list[tuple]):
        self.events = events

    def send(self, message):
        command, _payload = message
        self.events.append(("send", command))

    def close(self):
        self.events.append(("close",))


class ExitsOnFirstJoinProcess:
    def __init__(self, events: list[tuple]):
        self.events = events
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.events.append(("join", timeout))
        self.alive = False

    def terminate(self) -> None:
        self.events.append(("terminate",))
        self.alive = False

    def kill(self) -> None:
        self.events.append(("kill",))
        self.alive = False


class ExitsAfterTerminateProcess:
    def __init__(self, events: list[tuple]):
        self.events = events
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.events.append(("join", timeout))

    def terminate(self) -> None:
        self.events.append(("terminate",))
        self.alive = False

    def kill(self) -> None:
        self.events.append(("kill",))
        self.alive = False


class BrokenResultConnection:
    def __init__(self, task: RealtimeObservationTask):
        self.commands = [("predict", {"task": task})]
        self.sent: list[str] = []

    def send(self, message):
        command, _payload = message
        self.sent.append(command)
        if command == "result":
            raise BrokenPipeError
        if command == "error":
            raise AssertionError("worker tried to report an error over a closed pipe")

    def recv(self):
        return self.commands.pop(0)


class RecordingWorkerConnection:
    def __init__(self, task: RealtimeObservationTask):
        self.commands = [("predict", {"task": task})]
        self.sent: list[tuple[str, dict]] = []

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        return self.commands.pop(0)


class ReadyResultConnection:
    def __init__(self, result):
        self.result = result
        self.drained = False

    def poll(self):
        return not self.drained

    def recv(self):
        self.drained = True
        return "result", {"result": self.result}


class DummyWorkerPolicy(PolicyRunner):
    def predict(self, observation):
        return PolicyOutput(action=Action("unused"))


class SuccessfulWorkerMeasurer:
    def __init__(self, *, device: str, sync_cuda: bool):
        self.device = device
        self.sync_cuda = sync_cuda

    def run_and_measure(self, predict, observation, *, obs_id: int):
        return (
            PolicyOutput(action=Action("jump")),
            LatencyRecord(obs_id=obs_id, latency_ms=1.0),
        )


class FailingWorkerMeasurer:
    def __init__(self, *, device: str, sync_cuda: bool):
        self.device = device
        self.sync_cuda = sync_cuda

    def run_and_measure(self, predict, observation, *, obs_id: int):
        raise RuntimeError("policy boom")


class CallingWorkerMeasurer:
    def __init__(self, *, device: str, sync_cuda: bool):
        self.device = device
        self.sync_cuda = sync_cuda

    def run_and_measure(self, predict, observation, *, obs_id: int):
        return predict(observation), LatencyRecord(obs_id=obs_id, latency_ms=1.0)


def _patch_worker_dependencies(monkeypatch, measurer_cls):
    import latency_bench.core.actions as actions_mod
    import latency_bench.policy.factory as factory_mod

    class DummyActionResolver:
        def __init__(self, env_config):
            self.env_config = env_config

    monkeypatch.setattr(actions_mod, "ActionResolver", DummyActionResolver)
    monkeypatch.setattr(factory_mod, "build_policy", lambda config, action_resolver: DummyWorkerPolicy())
    monkeypatch.setattr(realtime_inference_worker, "RuntimeLatencyMeasurer", measurer_cls)
    monkeypatch.setattr(realtime_inference_worker, "_worker_runtime_metadata", lambda worker_id, start_method: {})


def _executor(
    *,
    pool: ScriptedInferencePool,
    ordering_policy: OrderingPolicy = OrderingPolicy.LATEST_READY,
    env_fps: float = 30.0,
    obs_fps: float = 30.0,
    max_steps: int = 5,
    logger: TrajectoryLogger | None = None,
    fake_clock: FakeClock | None = None,
    video_recorder=None,
    config: dict | None = None,
    action_space=None,
) -> RealtimeLatencyExecutor:
    fake_clock = fake_clock or FakeClock()
    noop = Action("noop" if action_space is None else 0, name="noop", is_noop=True)
    env = DummyEnvAdapter(env_fps=env_fps, max_steps=max_steps, noop_action=noop)
    if action_space is not None:
        env.action_space = action_space
    clock = EnvClock(env_fps=env_fps, obs_fps=obs_fps)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop,
        persistence_policy=HoldPolicy(),
        ordering_policy=ordering_policy,
    )
    runtime = RealtimeRuntime(
        env=env,
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        action_scheduler=action_scheduler,
        clock=clock,
    )
    pool.time_fn = fake_clock.time
    return RealtimeLatencyExecutor(
        runtime=runtime,
        logger=logger,
        time_fn=fake_clock.time,
        sleep_fn=fake_clock.sleep,
        inference_pool=pool,
        video_recorder=video_recorder,
        config=config,
    )


def test_realtime_history_joins_late_results_to_source_windows_and_resets():
    pool = ScriptedInferencePool([
        CompletionScript(Action(2), ready_time_ms=150, worker_id=0, device="cuda:0"),
        CompletionScript(Action(1), ready_time_ms=300, worker_id=0, device="cuda:0"),
    ], num_workers=1)
    executor = _executor(pool=pool, env_fps=60, obs_fps=15, max_steps=13,
                         config={"env": {"action_history_decisions": 8}}, action_space=Discrete(4))
    for episode in range(2):
        executor.run_episode(episode_id=episode, max_env_steps=13)
        first, second = pool.submitted
        np.testing.assert_array_equal(first.observation.data["transport"], np.zeros(56))
        assert second.submit_env_step == 12
        expected = np.zeros((8, 7))
        expected[-3] = [1, 0, 1, 0, 0, 0, 0]
        expected[-1] = [0, 0, 0, 0, 0, 1, 0]
        np.testing.assert_array_equal(second.observation.data["transport"], expected.reshape(-1))


def test_env_process_fails_when_action_pipe_is_closed(monkeypatch):
    parent_conn = ClosedActionEventPipe()
    pool = ScriptedInferencePool([], num_workers=1)
    executor = _executor(pool=pool)
    event = ActionEvent(
        action_id=0,
        action=Action("late"),
        obs_env_step=0,
        ready_time_ms=1.0,
        ready_step_float=0.03,
        latency=LatencyRecord(obs_id=0, latency_ms=1.0),
    )
    drain_calls = 0

    def fake_drain_and_release_results(**kwargs):
        nonlocal drain_calls
        drain_calls += 1
        if drain_calls == 1:
            kwargs["action_events"].append(event)
            kwargs["latency_values_ms"].append(1.0)
            return [event]
        return []

    executor.env = ScriptedEnvProcess(parent_conn)
    monkeypatch.setattr(executor, "_drain_and_release_results", fake_drain_and_release_results)

    with pytest.raises(BrokenPipeError):
        executor._run_episode_with_env_process(pool=pool, episode_id=0, seed=0, max_env_steps=1)
    assert parent_conn.closed


def test_env_process_drains_ready_results_before_submitting_observation(monkeypatch):
    task = RealtimeObservationTask(
        obs_id=0,
        observation=Observation(data=None, env_step=0, sim_time_ms=0.0),
        submit_env_step=0,
        submit_time_ms=0.0,
        observation_start_time_ms=0.0,
        shared_obs_id=0,
    )
    parent_conn = QueuedEnvMessagePipe(
        [
            ("observation", {"task": task}),
            (
                "step",
                {
                    "record": StepRecord(
                        episode_id=0,
                        env_step=0,
                        sim_time_ms=0.0,
                        active_action=Action("noop", name="noop", is_noop=True),
                        reward=0.0,
                        done=True,
                        info={"wall_clock_lag_ms": 0.0, "wall_elapsed_ms": 0.0},
                    )
                },
            ),
            ("episode_done", {}),
        ]
    )
    pool = DrainBeforeSubmitPool()
    executor = _executor(pool=pool, max_steps=1)

    def fake_drain_and_release_results(**kwargs):
        kwargs["pool"].drained = True
        return []

    executor.env = ScriptedEnvProcess(parent_conn)
    monkeypatch.setattr(executor, "_drain_and_release_results", fake_drain_and_release_results)

    metrics = executor._run_episode_with_env_process(
        pool=pool,
        episode_id=0,
        seed=0,
        max_env_steps=1,
    )

    assert len(pool.submitted) == 1
    assert metrics.metadata["submitted_observation_frames"] == 1
    assert metrics.metadata["dropped_observation_count"] == 0


def test_env_process_does_not_enqueue_released_actions_on_parent_scheduler(monkeypatch):
    task = RealtimeObservationTask(
        obs_id=0,
        observation=Observation(data=None, env_step=0, sim_time_ms=0.0),
        submit_env_step=0,
        submit_time_ms=0.0,
        observation_start_time_ms=0.0,
        shared_obs_id=0,
    )
    parent_conn = QueuedEnvMessagePipe(
        [
            ("observation", {"task": task}),
            (
                "step",
                {
                    "record": StepRecord(
                        episode_id=0,
                        env_step=0,
                        sim_time_ms=0.0,
                        active_action=Action("noop", name="noop", is_noop=True),
                        reward=0.0,
                        done=True,
                        info={"wall_clock_lag_ms": 0.0, "wall_elapsed_ms": 0.0},
                    )
                },
            ),
            ("episode_done", {}),
        ]
    )
    pool = ScriptedInferencePool(
        [CompletionScript(Action("jump"), ready_time_ms=0.0, worker_id=0, device="cpu")],
        num_workers=1,
    )
    executor = _executor(pool=pool, max_steps=1)

    executor.env = ScriptedEnvProcess(parent_conn)

    metrics = executor._run_episode_with_env_process(
        pool=pool,
        episode_id=0,
        seed=0,
        max_env_steps=1,
    )

    assert metrics.num_actions == 1
    assert executor.action_scheduler.get_pending_events() == []
    assert executor.action_scheduler.get_applied_events() == []
    assert ("release_observation", {"obs_id": 0}) in parent_conn.sent


def test_env_process_records_tail_attempt_and_releases_observation_before_episode_boundary(
    monkeypatch,
    tmp_path,
):
    task = RealtimeObservationTask(
        obs_id=0,
        observation=Observation(data=None, env_step=0, sim_time_ms=0.0),
        submit_env_step=0,
        submit_time_ms=0.0,
        observation_start_time_ms=0.0,
        shared_obs_id=42,
    )
    parent_conn = QueuedEnvMessagePipe(
        [
            ("observation", {"task": task}),
            (
                "step",
                {
                    "record": StepRecord(
                        episode_id=0,
                        env_step=0,
                        sim_time_ms=0.0,
                        active_action=Action("noop", name="noop", is_noop=True),
                        reward=0.0,
                        done=True,
                        info={"wall_clock_lag_ms": 0.0, "wall_elapsed_ms": 0.0},
                    )
                },
            ),
            ("episode_done", {}),
        ]
    )
    pool = LateCompletionAfterFinalizePool()
    logger = TrajectoryLogger(
        tmp_path,
        save_step_records=True,
        save_action_records=True,
        save_latency_records=True,
    )
    executor = _executor(pool=pool, max_steps=1, logger=logger)

    executor.env = ScriptedEnvProcess(parent_conn)

    metrics = executor._run_episode_with_env_process(
        pool=pool,
        episode_id=0,
        seed=0,
        max_env_steps=1,
    )

    assert metrics.num_actions == 0
    assert parent_conn.sent.index(("release_observation", {"obs_id": 42})) < parent_conn.sent.index(("finish_episode", {}))
    attempts = [
        json.loads(line)
        for line in (tmp_path / "observation_attempts.jsonl").read_text().splitlines()
    ]
    assert attempts == [
        {
            "episode_id": 0,
            "obs_env_step": 0,
            "obs_id": 0,
            "observation_to_action_latency_ms": 1.0,
            "submitted": True,
            "worker_id": 0,
            "worker_service_time_ms": 1.0,
        }
    ]


def test_inference_pool_closes_connections_after_workers_exit():
    events: list[tuple] = []
    pool = ProcessInferencePool.__new__(ProcessInferencePool)
    pool.closed = False
    pool._connections = [CloseOrderConnection(events)]
    pool._processes = [ExitsOnFirstJoinProcess(events)]

    pool.close()

    assert events == [
        ("send", "close"),
        ("join", 0.2),
        ("close",),
    ]


def test_inference_pool_terminates_workers_still_alive_after_close():
    events: list[tuple] = []
    pool = ProcessInferencePool.__new__(ProcessInferencePool)
    pool.closed = False
    pool._connections = [CloseOrderConnection(events)]
    pool._processes = [ExitsAfterTerminateProcess(events)]

    pool.close()

    assert events == [
        ("send", "close"),
        ("join", 0.2),
        ("terminate",),
        ("join", 1.0),
        ("close",),
    ]


def test_inference_worker_exits_when_result_pipe_is_closed(monkeypatch):
    _patch_worker_dependencies(monkeypatch, SuccessfulWorkerMeasurer)
    task = RealtimeObservationTask(
        obs_id=7,
        observation=Observation(data=object(), env_step=0, sim_time_ms=0.0),
        submit_env_step=0,
        submit_time_ms=0.0,
        observation_start_time_ms=0.0,
    )
    connection = BrokenResultConnection(task)

    _inference_worker(
        connection,
        {"experiment": {"seed": 0}, "env": {}, "policy": {}, "executor": {"sync_cuda": True}},
        worker_id=0,
        device="cpu",
        start_method="spawn",
    )

    assert connection.sent == ["ready", "result"]


def test_inference_worker_returns_slim_result_without_observation_task(monkeypatch):
    _patch_worker_dependencies(monkeypatch, SuccessfulWorkerMeasurer)
    task = RealtimeObservationTask(
        obs_id=7,
        observation=Observation(data={"pixels": "large"}, env_step=3, sim_time_ms=100.0),
        submit_env_step=3,
        submit_time_ms=100.0,
        observation_start_time_ms=100.0,
    )
    connection = RecordingWorkerConnection(task)
    connection.commands.append(("close", {}))

    _inference_worker(
        connection,
        {"experiment": {"seed": 0}, "env": {}, "policy": {}, "executor": {"sync_cuda": True}},
        worker_id=0,
        device="cpu",
        start_method="spawn",
    )

    command, payload = connection.sent[-1]
    assert command == "result"
    assert isinstance(payload["result"], RealtimePolicyResult)
    assert not hasattr(payload["result"], "task")
    assert not hasattr(payload["result"], "obs_id")
    assert not hasattr(payload["result"], "worker_id")
    assert not hasattr(payload["result"], "device")


def test_inference_worker_batch_predicts_raw_observations_without_localizing(monkeypatch):
    import latency_bench.core.actions as actions_mod
    import latency_bench.policy.factory as factory_mod

    class DummyActionResolver:
        def __init__(self, env_config):
            self.env_config = env_config

    observations = [
        Observation(data=np.array([index], dtype=np.float32), env_step=index, sim_time_ms=0.0)
        for index in range(2)
    ]
    captured = []

    class BatchPolicy(DummyWorkerPolicy):
        def predict_batch(self, raw_observations):
            captured.extend(raw_observations)
            return [PolicyOutput(action=Action(index)) for index in range(len(raw_observations))]

    class ExplodingLocalizer:
        def localize(self, observation):
            raise AssertionError("batch RPC must pass raw observations")

    class Connection:
        def __init__(self):
            self.commands = [
                ("predict_batch", {"observations": observations}),
                ("close", {}),
            ]
            self.sent = []

        def send(self, message):
            self.sent.append(message)

        def recv(self):
            return self.commands.pop(0)

    monkeypatch.setattr(actions_mod, "ActionResolver", DummyActionResolver)
    monkeypatch.setattr(factory_mod, "build_policy", lambda config, action_resolver: BatchPolicy())
    monkeypatch.setattr(realtime_inference_worker, "ObservationLocalizer", ExplodingLocalizer)
    monkeypatch.setattr(realtime_inference_worker, "_worker_runtime_metadata", lambda worker_id, start_method: {})

    connection = Connection()
    _inference_worker(
        connection,
        {"experiment": {"seed": 0}, "env": {}, "policy": {}, "executor": {"sync_cuda": True}},
        worker_id=0,
        device="cpu",
        start_method="spawn",
    )

    assert [command for command, _payload in connection.sent] == ["ready", "batch_result"]
    assert [id(item) for item in captured] == [id(item) for item in observations]
    assert [output.action.value for output in connection.sent[-1][1]["policy_outputs"]] == [0, 1]


def test_inference_worker_predicts_from_worker_local_observation_copy(monkeypatch):
    import latency_bench.core.actions as actions_mod
    import latency_bench.policy.factory as factory_mod

    class DummyActionResolver:
        def __init__(self, env_config):
            self.env_config = env_config

    class MutatingWorkerPolicy(DummyWorkerPolicy):
        def predict(self, observation):
            assert observation.metadata["obs_id"] == 7
            observation.data[...] = 99
            observation.metadata["env_raw_rgb_frame_stack"][...] = 77
            return PolicyOutput(action=Action("jump"))

    monkeypatch.setattr(actions_mod, "ActionResolver", DummyActionResolver)
    monkeypatch.setattr(factory_mod, "build_policy", lambda config, action_resolver: MutatingWorkerPolicy())
    monkeypatch.setattr(realtime_inference_worker, "RuntimeLatencyMeasurer", CallingWorkerMeasurer)
    monkeypatch.setattr(realtime_inference_worker, "_worker_runtime_metadata", lambda worker_id, start_method: {})

    observation = Observation(
        data=np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        env_step=3,
        sim_time_ms=100.0,
        metadata={
            "env_raw_rgb_frame_stack": np.arange(24, dtype=np.uint8).reshape(1, 2, 4, 3),
        },
    )
    owner = ObservationSlotPool.from_observation(observation, capacity=1)
    try:
        shared_observation = owner.write(obs_id=7, observation=observation)
        task = RealtimeObservationTask(
            obs_id=7,
            observation=shared_observation,
            submit_env_step=3,
            submit_time_ms=100.0,
            observation_start_time_ms=100.0,
        )
        connection = RecordingWorkerConnection(task)
        connection.commands.append(("close", {}))

        _inference_worker(
            connection,
            {"experiment": {"seed": 0}, "env": {}, "policy": {}, "executor": {"sync_cuda": True}},
            worker_id=0,
            device="cpu",
            start_method="spawn",
        )

        cache = ObservationAttachmentCache(blocks={})
        try:
            attached = cache.attach(shared_observation)
            np.testing.assert_array_equal(attached.data, observation.data)
            np.testing.assert_array_equal(
                attached.metadata["env_raw_rgb_frame_stack"],
                observation.metadata["env_raw_rgb_frame_stack"],
            )
        finally:
            cache.close()
    finally:
        owner.close()




def test_env_action_drain_happens_before_scheduler_update() -> None:
    clock = EnvClock(env_fps=10.0, obs_fps=10.0)
    scheduler = ActionScheduler(
        clock=clock,
        noop_action=Action("noop", name="noop", is_noop=True),
        persistence_policy=HoldPolicy(),
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
    )
    event = ActionEvent(
        action_id=0,
        action=Action("jump"),
        obs_env_step=0,
        ready_time_ms=0.0,
        ready_step_float=0.0,
        latency=LatencyRecord(obs_id=0, latency_ms=0.0),
        metadata={},
    )
    event.latency_raw_frames = 0
    event.ready_raw_frame = 0
    pipe = QueuedEnvMessagePipe([("action_dispatch", _action_event_dispatch_payload(event))])

    closed = _drain_env_action_messages(
        connection=pipe,
        action_scheduler=scheduler,
        release_observation=lambda obs_id: None,
    )
    active_action = scheduler.update(0, 0.0)
    applied = scheduler.get_applied_events()[0]

    assert closed is False
    assert active_action == Action("jump")


def test_env_action_drain_releases_observations_with_explicit_callback() -> None:
    clock = EnvClock(env_fps=10.0, obs_fps=10.0)
    scheduler = ActionScheduler(
        clock=clock,
        noop_action=Action("noop", name="noop", is_noop=True),
        persistence_policy=HoldPolicy(),
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
    )
    released: list[int] = []
    pipe = QueuedEnvMessagePipe([("release_observation", {"obs_id": 9})])

    closed = _drain_env_action_messages(
        connection=pipe,
        action_scheduler=scheduler,
        release_observation=released.append,
    )

    assert closed is False
    assert released == [9]


def test_env_wait_drains_actions_before_next_frame_boundary() -> None:
    fake_clock = FakeClock(now=1.0)
    clock = EnvClock(env_fps=10.0, obs_fps=10.0)
    scheduler = ActionScheduler(
        clock=clock,
        noop_action=Action("noop", name="noop", is_noop=True),
        persistence_policy=HoldPolicy(),
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
    )
    event = ActionEvent(
        action_id=0,
        action=Action("jump"),
        obs_env_step=0,
        ready_time_ms=0.0,
        ready_step_float=0.0,
        latency=LatencyRecord(obs_id=0, latency_ms=0.0),
        metadata={},
    )
    event.ready_raw_frame = 0
    pipe = TimedEnvMessagePipe(
        [("action_dispatch", _action_event_dispatch_payload(event))],
        fake_clock,
    )

    closed = _drain_env_actions_until_wall_time(
        connection=pipe,
        action_scheduler=scheduler,
        release_observation=lambda obs_id: None,
        target_wall_s=1.100,
        time_fn=fake_clock.time,
    )
    active_action = scheduler.update(0, 0.0)

    assert closed is False
    assert fake_clock.now == 1.100
    assert pipe.poll_timeouts[0] == pytest.approx(0.100)
    assert pipe.poll_timeouts[-1] == pytest.approx(0.090)
    assert active_action == Action("jump")


def test_process_inference_pool_reconstructs_completion_from_busy_task(monkeypatch):
    monkeypatch.setattr(realtime_pool, "monotonic_ms", lambda: 20.0)
    task = RealtimeObservationTask(
        obs_id=11,
        observation=Observation(data="obs", env_step=4, sim_time_ms=120.0),
        submit_env_step=4,
        submit_time_ms=120.0,
        observation_start_time_ms=5.0,

    )
    result = RealtimePolicyResult(
        policy_output=PolicyOutput(action=Action("jump")),
        completion_time_ms=135.0,
        latency_record=LatencyRecord(obs_id=11, latency_ms=15.0),

    )
    pool = ProcessInferencePool.__new__(ProcessInferencePool)
    pool.closed = False
    pool._connections = [ReadyResultConnection(result)]
    pool.inference_devices = ["cpu"]
    pool._busy_worker_tasks = {0: (task, 10.0)}

    completions = pool._drain_all_ready()

    assert len(completions) == 1
    assert completions[0].task is task
    assert completions[0].policy_output.action.value == "jump"
    assert completions[0].worker_id == 0
    assert completions[0].device == "cpu"
    assert completions[0].observation_to_action_latency_ms == 15.0
    assert completions[0].worker_service_time_ms == 10.0


def test_action_dispatch_and_ack_payloads_are_scheduler_slim():
    event = ActionEvent(
        action_id=5,
        action=Action("jump"),
        obs_env_step=3,
        ready_time_ms=90.0,
        ready_step_float=4.0,
        latency=LatencyRecord(obs_id=8, latency_ms=25.0),
        latency_raw_frames=1,
        ready_raw_frame=4,
        apply_raw_frame=6,
        applied=True,
        metadata={"frame_ms": 30.0, "policy_blob": "large"},
    )
    dispatched = RealtimeActionDispatch(
        action_id=5,
        action=Action("jump"),
        obs_env_step=3,
        ready_time_ms=90.0,
        ready_step_float=4.0,
        apply_raw_frame=6,
        applied=True,
    )

    dispatch = _action_event_dispatch_payload(event)
    ack = _action_event_ack_payload(dispatched)

    assert dispatch == {
        "action_id": 5,
        "action": Action("jump"),
        "obs_env_step": 3,
        "ready_time_ms": 90.0,
        "ready_step_float": 4.0,
        "latency_raw_frames": 1,
        "ready_raw_frame": 4,
    }
    assert ack == {
        "action_id": 5,
        "applied": True,
        "apply_raw_frame": 6,
    }


@pytest.mark.parametrize("error_class", [RuntimeError, InvalidAgentAction])
def test_inference_worker_reports_real_errors_when_pipe_is_open(monkeypatch, error_class):
    _patch_worker_dependencies(monkeypatch, FailingWorkerMeasurer)
    def fail(*args, **kwargs):
        raise error_class("policy boom")
    monkeypatch.setattr(FailingWorkerMeasurer, "run_and_measure", fail)
    task = RealtimeObservationTask(
        obs_id=8,
        observation=Observation(data=object(), env_step=0, sim_time_ms=0.0),
        submit_env_step=0,
        submit_time_ms=0.0,
        observation_start_time_ms=0.0,
    )
    connection = RecordingWorkerConnection(task)

    _inference_worker(
        connection,
        {"experiment": {"seed": 0}, "env": {}, "policy": {}, "executor": {"sync_cuda": True}},
        worker_id=0,
        device="cpu",
        start_method="spawn",
    )

    error_tag = "invalid_agent_action" if error_class is InvalidAgentAction else "error"
    assert [command for command, _payload in connection.sent] == ["ready", error_tag]
    assert "policy boom" in connection.sent[-1][1]["traceback"]
    pool = ProcessInferencePool.__new__(ProcessInferencePool)
    pool._connections = [SimpleNamespace(poll=lambda: True, recv=lambda: connection.sent[-1])]
    pool._busy_worker_tasks = {0: (task, 0.0)}
    with pytest.raises(error_class, match="policy boom"):
        pool._drain_all_ready()
    assert not pool._busy_worker_tasks


def test_policy_preparation_is_outside_the_episode_clock(tmp_path, monkeypatch):
    clock = FakeClock()
    pool = ScriptedInferencePool([], num_workers=1)
    logger = TrajectoryLogger(str(tmp_path), save_step_records=True,
                              save_action_records=True, save_latency_records=True)
    executor = _executor(pool=pool, fake_clock=clock, logger=logger,
                         env_fps=1, obs_fps=1, max_steps=2)
    monkeypatch.setattr(pool, "prepare_episode", lambda episode_id: clock.sleep(10))
    metrics = executor.run_episode(episode_id=0, seed=0, max_env_steps=2)
    logger.flush()
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [row["info"]["wall_elapsed_ms"] for row in rows] == [0, 1000]
    assert metrics.metadata["max_wall_clock_lag_ms"] == 0


def test_realtime_observes_on_stride_without_catchup_metadata():
    pool = ScriptedInferencePool(
        [
            CompletionScript(Action("a0"), ready_time_ms=1000.0, worker_id=0, device="cpu"),
            CompletionScript(Action("a1"), ready_time_ms=1000.0, worker_id=1, device="cpu"),
            CompletionScript(Action("a2"), ready_time_ms=1000.0, worker_id=2, device="cpu"),
        ],
        num_workers=3,
    )
    executor = _executor(pool=pool, env_fps=30.0, obs_fps=10.0, max_steps=7)

    metrics = executor.run_episode(episode_id=0, max_env_steps=7)

    assert [task.submit_env_step for task in pool.submitted] == [0, 3, 6]
    assert len(executor.env.applied_actions) == 7
    assert metrics.metadata["submitted_observation_frames"] == 3
    assert metrics.metadata["dropped_observation_count"] == 0
    assert "total_catchup_steps" not in metrics.metadata


def test_realtime_drops_observation_when_all_workers_are_busy():
    pool = ScriptedInferencePool(
        [CompletionScript(Action("slow"), ready_time_ms=1000.0, worker_id=0, device="cpu")],
        num_workers=1,
    )
    executor = _executor(pool=pool, max_steps=3)

    metrics = executor.run_episode(episode_id=0, max_env_steps=3)

    assert [task.submit_env_step for task in pool.submitted] == [0]
    assert metrics.metadata["submitted_observation_frames"] == 1
    assert metrics.metadata["dropped_observation_count"] == 2
    assert executor.logger is None


def test_realtime_logs_capacity_drops_and_tail_completion_in_attempt_order(tmp_path):
    pool = ScriptedInferencePool(
        [CompletionScript(Action("slow"), ready_time_ms=1000.0, worker_id=0, device="cpu")],
        num_workers=1,
    )
    logger = TrajectoryLogger(
        tmp_path,
        save_step_records=True,
        save_action_records=True,
        save_latency_records=True,
    )
    executor = _executor(pool=pool, max_steps=3, logger=logger)

    metrics = executor.run_episode(episode_id=4, max_env_steps=3)

    attempts = [
        json.loads(line)
        for line in (tmp_path / "observation_attempts.jsonl").read_text().splitlines()
    ]
    assert attempts == [
        {
            "episode_id": 4,
            "obs_env_step": 0,
            "obs_id": 0,
            "observation_to_action_latency_ms": 1000.0,
            "submitted": True,
            "worker_id": 0,
            "worker_service_time_ms": 1000.0,
        },
        {
            "episode_id": 4,
            "obs_env_step": 1,
            "obs_id": 1,
            "submitted": False,
        },
        {
            "episode_id": 4,
            "obs_env_step": 2,
            "obs_id": 2,
            "submitted": False,
        },
    ]
    assert metrics.num_actions == 0


def test_realtime_attempt_trace_uses_observation_order_for_out_of_order_results(tmp_path):
    pool = ScriptedInferencePool(
        [
            CompletionScript(Action("first"), ready_time_ms=80.0, worker_id=0, device="cpu:0"),
            CompletionScript(Action("second"), ready_time_ms=34.0, worker_id=1, device="cpu:1"),
        ],
        num_workers=2,
    )
    logger = TrajectoryLogger(
        tmp_path,
        save_step_records=True,
        save_action_records=True,
        save_latency_records=True,
    )
    executor = _executor(
        pool=pool,
        ordering_policy=OrderingPolicy.LATEST_READY,
        max_steps=4,
        logger=logger,
    )

    executor.run_episode(episode_id=0, max_env_steps=4)

    attempts = [
        json.loads(line)
        for line in (tmp_path / "observation_attempts.jsonl").read_text().splitlines()
    ]
    assert [row["obs_id"] for row in attempts[:2]] == [0, 1]
    assert [row["worker_id"] for row in attempts[:2]] == [0, 1]
    assert attempts[0]["observation_to_action_latency_ms"] == 80.0
    assert attempts[1]["observation_to_action_latency_ms"] == pytest.approx(
        34.0 - 1000.0 / 30.0
    )


def test_realtime_records_inference_worker_count_in_episode_metadata():
    pool = ScriptedInferencePool(
        [
            CompletionScript(Action("a0"), ready_time_ms=1000.0, worker_id=0, device="cpu"),
            CompletionScript(Action("a1"), ready_time_ms=1000.0, worker_id=1, device="cpu"),
        ],
        num_workers=2,
    )
    executor = _executor(pool=pool, max_steps=2)

    metrics = executor.run_episode(episode_id=0, max_env_steps=2)

    assert metrics.metadata["inference_worker_count"] == 2


def test_latest_ready_ignores_older_late_result():
    first = Action("first")
    second = Action("second")
    pool = ScriptedInferencePool(
        [
            CompletionScript(first, ready_time_ms=80.0, worker_id=0, device="cuda:0"),
            CompletionScript(second, ready_time_ms=34.0, worker_id=1, device="cuda:1"),
        ],
        num_workers=2,
    )
    executor = _executor(pool=pool, ordering_policy=OrderingPolicy.LATEST_READY, max_steps=4)

    metrics = executor.run_episode(episode_id=0, max_env_steps=4)

    applied_values = [event.action.value for event in executor.action_scheduler.get_applied_events()]
    assert applied_values == ["second"]
    assert metrics.metadata["ignored_late_result_count"] == 1
    assert metrics.num_dropped_actions == 1


def test_issue_order_fifo_blocks_later_ready_result_until_head_completes():
    first = Action("first")
    second = Action("second")
    pool = ScriptedInferencePool(
        [
            CompletionScript(first, ready_time_ms=80.0, worker_id=0, device="cuda:0"),
            CompletionScript(second, ready_time_ms=34.0, worker_id=1, device="cuda:1"),
        ],
        num_workers=2,
    )
    executor = _executor(pool=pool, ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO, max_steps=5)

    metrics = executor.run_episode(episode_id=0, max_env_steps=5)

    applied_values = [event.action.value for event in executor.action_scheduler.get_applied_events()]
    assert applied_values == ["first", "second"]
    assert metrics.metadata["fifo_blocked_count"] >= 1
    assert metrics.metadata["ignored_late_result_count"] == 0


def test_realtime_records_worker_id_once_on_latency_metadata(tmp_path):
    action = Action("jump")
    pool = ScriptedInferencePool(
        [CompletionScript(action, ready_time_ms=1.0, worker_id=1, device="cuda:1")],
        num_workers=2,
    )
    logger = TrajectoryLogger(
        tmp_path,
        save_step_records=True,
        save_action_records=True,
        save_latency_records=True,
    )
    executor = _executor(pool=pool, max_steps=2, logger=logger)

    executor.run_episode(episode_id=0, max_env_steps=2)

    # Records are flushed to disk (and cleared from memory) at episode end.
    latency = json.loads((tmp_path / "latencies.jsonl").read_text(encoding="utf-8").splitlines()[0])
    event = json.loads((tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert latency["device"] == "cuda:1"
    assert latency["metadata"]["worker_id"] == 1
    assert "worker_id" not in event["metadata"]
    assert "device" not in event["metadata"]


def test_realtime_overrun_uses_wall_readiness_for_actions_and_prefix(tmp_path):
    """Long-lived realtime regression: a slow env step cannot delay a returned action twice."""
    clock = FakeClock()
    pool = ScriptedInferencePool([
        CompletionScript(Action([value]), ready_time_ms=25 + 100 * index, worker_id=0, device="cpu")
        for index, value in enumerate([1, 2, 3, 4])
    ], num_workers=1)
    logger = TrajectoryLogger(tmp_path, save_step_records=True, save_action_records=True,
                              save_latency_records=True)
    executor = _executor(pool=pool, env_fps=20, obs_fps=20, max_steps=4,
                         fake_clock=clock, logger=logger)
    executor.action_scheduler.noop_action = Action([0], is_noop=True)
    executor.action_prefix = {"mode": "fixed_frames", "frames": 2, "action_horizon": 4, "action_dim": 1}
    step = executor.env.step

    def slow_step(action):
        clock.now += 0.1
        return step(action)

    executor.env.step = slow_step
    executor.run_episode(episode_id=0, max_env_steps=4)
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [row["active_action"]["value"] for row in rows] == [[0], [1], [2], [3]]
    np.testing.assert_array_equal(pool.submitted[2].observation.metadata["action_prefix"][:, 0], [2, 2, 0, 0])
    events = [json.loads(line) for line in (tmp_path / "actions.jsonl").read_text().splitlines()]
    assert [event["ready_raw_frame"] for event in events] == [1, 2, 3, None]
    assert [event["apply_raw_frame"] for event in events] == [1, 2, 3, None]


def test_environment_worker_uses_wall_readiness_after_slow_steps(monkeypatch):
    """Long-lived IPC-loop regression: delivered actions remain usable after a deadline miss."""
    from latency_bench.executors.realtime import env_worker

    time = FakeClock()
    clock = EnvClock(env_fps=20, obs_fps=20)
    env = DummyEnvAdapter(env_fps=20, max_steps=4, noop_action=Action(0, is_noop=True))
    step = env.step

    def slow_step(action):
        time.now += 0.1
        return step(action)

    env.step = slow_step
    env.reset()
    scheduler = ActionScheduler(clock, env.noop_action, HoldPolicy(), OrderingPolicy.LATEST_READY)
    monkeypatch.setattr(env_worker, "time", SimpleNamespace(perf_counter=time.time, sleep=time.sleep))
    monkeypatch.setattr(env_worker, "monotonic_ms", lambda: time.now * 1000)

    class Connection:
        def __init__(self):
            self.pending, self.records = [], []

        def poll(self, timeout=None):
            if timeout is not None:
                time.sleep(timeout)
            return bool(self.pending and self.pending[0][0] <= time.now * 1000)

        def recv(self):
            return self.pending.pop(0)[1]

        def send(self, message):
            tag, payload = message
            if tag == "observation":
                task = payload["task"]
                ready = task.submit_time_ms + 25
                event = ActionEvent(task.obs_id, Action(task.submit_env_step + 1), task.submit_env_step,
                                    ready, ready / clock.frame_ms, LatencyRecord(task.obs_id, 25))
                self.pending.extend([
                    (ready, ("release_observation", {"obs_id": task.shared_obs_id})),
                    (ready, ("action_dispatch", _action_event_dispatch_payload(event))),
                ])
            elif tag == "step":
                self.records.append(payload["record"])
            elif tag == "episode_done":
                self.pending = [item for item in self.pending if item[1][0] == "release_observation"]
                self.pending.append((time.now * 1000, ("finish_episode", {})))

    connection = Connection()
    owner = ObservationSlotPool.from_observation(env.observe(), capacity=4)
    try:
        assert env_worker._run_episode(
            connection, {"logging": {}, "policy": {"action_prefix": {"mode": "none"}}},
            env, clock, scheduler, owner, episode_id=0, seed=1, max_env_steps=4,
            video_output_dir=None, video_fps=None,
        )
    finally:
        owner.close()
    assert [record.active_action.value for record in connection.records] == [0, 1, 2, 3]


def test_realtime_records_current_episode_video(tmp_path):
    from latency_bench.logging.video import EpisodeVideoRecorder

    pool = ScriptedInferencePool(
        [CompletionScript(Action("jump"), ready_time_ms=1.0, worker_id=0, device="cpu")],
        num_workers=1,
    )
    recorder = EpisodeVideoRecorder(tmp_path / "videos", fps=30.0)
    executor = _executor(pool=pool, max_steps=2, video_recorder=recorder)

    executor.run_episode(episode_id=5, max_env_steps=2)

    manifest = [
        json.loads(line)
        for line in (tmp_path / "videos" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest) == 1
    assert manifest[0]["episode_id"] == 5
    assert manifest[0]["episode_return"] == 2.0
    assert manifest[0]["frame_count"] == 2
    assert (tmp_path / "videos" / manifest[0]["video_path"]).is_file()


@pytest.mark.parametrize("history", [False, True])
def test_realtime_warmup_writes_separate_artifacts(monkeypatch, tmp_path, history):
    from latency_bench.executors import realtime_warmup

    monkeypatch.setattr(realtime_warmup, "MIN_WARMUP_LATENCY_SAMPLES", 3)
    monkeypatch.setattr(realtime_warmup, "MAX_WARMUP_LATENCY_SAMPLES", 3)
    monkeypatch.setattr(realtime_warmup, "WARMUP_WINDOW_SIZE", 1)
    monkeypatch.setattr(realtime_warmup, "STABLE_WINDOW_COUNT", 3)

    action = Action(2)
    pool = ScriptedInferencePool(
        [
            CompletionScript(action, ready_time_ms=1.0, worker_id=0, device="cuda:0", ),
            CompletionScript(action, ready_time_ms=1.0, worker_id=0, device="cuda:0", ),
            CompletionScript(action, ready_time_ms=1.0, worker_id=0, device="cuda:0", ),
        ],
        num_workers=1,
    )
    history_snapshots = []
    submit = pool.submit

    def capture_warmup(task):
        cache = ObservationAttachmentCache(blocks={})
        try:
            attached = cache.attach(task.observation)
            if history:
                history_snapshots.append(attached.data["transport"].copy())
            return submit(task)
        finally:
            cache.close()

    pool.submit = capture_warmup
    formal_logger = TrajectoryLogger(
        tmp_path,
        save_step_records=True,
        save_action_records=True,
        save_latency_records=True,
    )
    executor = _executor(
        pool=pool,
        max_steps=2,
        logger=formal_logger,
        config={"logging": {"output_dir": str(tmp_path)},
                "env": {"action_history_decisions": 8 if history else None}},
        action_space=Discrete(4) if history else None,
    )

    warmup_dir = tmp_path / "warmup/episode_000000"
    summary = executor.run_warmup(artifact_dir=warmup_dir)

    assert summary["warmup_mode"] == "realtime_pipeline"
    assert summary["warmup_status"] == "stable"
    assert summary["warmup_samples"] == 3
    assert len(pool.submitted) == 3
    assert all(task.observation.metadata["episode_id"] == -1 for task in pool.submitted)
    assert all(task.observation.metadata["action_noise_seed"] == 0 for task in pool.submitted)
    assert len(formal_logger.latency_records) == 0
    assert (warmup_dir / "warmup_latencies.jsonl").is_file()
    assert (warmup_dir / "warmup_summary.json").is_file()
    assert (warmup_dir / "warmup_stability.png").is_file()
    assert not (tmp_path / "warmup_summary.json").exists()
    assert not (tmp_path / "latencies.jsonl").exists()
    if history:
        assert len(history_snapshots) == 3
        np.testing.assert_array_equal(history_snapshots[0], np.zeros(56))
        assert np.any(history_snapshots[1])
        np.testing.assert_array_equal(history_snapshots[2], np.zeros(56))


def test_realtime_warmup_uses_stress_policy_samples(monkeypatch, tmp_path):
    from latency_bench.executors import realtime_warmup

    monkeypatch.setattr(realtime_warmup, "MIN_WARMUP_LATENCY_SAMPLES", 3)
    monkeypatch.setattr(realtime_warmup, "MAX_WARMUP_LATENCY_SAMPLES", 4)
    monkeypatch.setattr(realtime_warmup, "WARMUP_WINDOW_SIZE", 1)
    monkeypatch.setattr(realtime_warmup, "STABLE_WINDOW_COUNT", 3)

    action = Action("jump")
    pool = ScriptedInferencePool(
        [
            CompletionScript(action, ready_time_ms=0.0, worker_id=0, device="cuda:0"),
            CompletionScript(action, ready_time_ms=0.0, worker_id=0, device="cuda:0"),
            CompletionScript(action, ready_time_ms=0.0, worker_id=0, device="cuda:0"),
            CompletionScript(action, ready_time_ms=0.0, worker_id=0, device="cuda:0"),
        ],
        num_workers=1,
    )
    executor = _executor(
        pool=pool,
        max_steps=2,
        config={"logging": {"output_dir": str(tmp_path)}},
    )

    summary = executor.run_warmup()

    assert summary["warmup_status"] == "stable"
    assert summary["warmup_samples"] == 4


def test_realtime_pipeline_warmup_keeps_steps_out_of_formal_episodes(monkeypatch, tmp_path):
    from latency_bench.executors import realtime_warmup

    monkeypatch.setattr(realtime_warmup, "MIN_WARMUP_LATENCY_SAMPLES", 2)
    monkeypatch.setattr(realtime_warmup, "MAX_WARMUP_LATENCY_SAMPLES", 2)
    monkeypatch.setattr(realtime_warmup, "WARMUP_WINDOW_SIZE", 1)
    monkeypatch.setattr(realtime_warmup, "STABLE_WINDOW_COUNT", 2)

    action = Action([9.0])
    pool = ScriptedInferencePool(
        [
            CompletionScript(action, ready_time_ms=0.0, worker_id=0, device="cuda:0"),
            CompletionScript(action, ready_time_ms=0.0, worker_id=0, device="cuda:0"),
        ],
        num_workers=1,
    )
    executor = _executor(
        pool=pool,
        max_steps=2,
        config={"logging": {"output_dir": str(tmp_path)}},
    )

    executor.action_scheduler.noop_action = Action([7.0])
    executor.action_prefix = {"mode": "profile_mean", "frames": 2, "action_horizon": 4, "action_dim": 1}
    snapshots = []
    original_submit = pool.submit

    def capture(task):
        cache = ObservationAttachmentCache(blocks={})
        try:
            attached = cache.attach(task.observation)
            snapshots.append(attached.metadata["action_prefix"].copy())
        finally:
            cache.close()
        return original_submit(task)

    monkeypatch.setattr(pool, "submit", capture)
    run_episode = executor.run_episode

    def fail_run_episode(*, episode_id, seed, max_env_steps):
        del episode_id, seed, max_env_steps
        raise AssertionError("stress warmup must not run realtime episodes")

    executor.run_episode = fail_run_episode

    summary = executor.run_warmup()

    assert summary["warmup_mode"] == "realtime_pipeline"
    assert summary["warmup_samples"] == 2

    assert len(snapshots) == 2
    for prefix in snapshots:
        np.testing.assert_array_equal(prefix[:, 0], [7, 7, 0, 0])
    executor.run_episode = run_episode
    executor.run_episode(episode_id=0, seed=1, max_env_steps=1)
    first = pool.submitted[0].observation
    assert first.env_step == 0
    np.testing.assert_array_equal(first.metadata["action_prefix"][:, 0], [7, 7, 0, 0])


def test_gc_paused_until_episode_end_restores_state_and_collects():
    from latency_bench.executors.realtime.executor import _gc_paused_until_episode_end

    class CallerOwnedCycle:
        pass

    value = CallerOwnedCycle()
    value.self_reference = value
    reference = weakref.ref(value)
    assert gc.isenabled()
    with _gc_paused_until_episode_end():
        assert not gc.isenabled()
    assert gc.isenabled()
    del value
    gc.collect()
    assert reference() is None


def test_run_episode_leaves_gc_enabled():
    pool = ScriptedInferencePool([], num_workers=1)
    executor = _executor(pool=pool, max_steps=3)

    executor.run_episode(episode_id=0, seed=0, max_env_steps=3)

    assert gc.isenabled()


def test_inference_worker_collect_command_runs_gc_and_keeps_serving(monkeypatch):
    _patch_worker_dependencies(monkeypatch, SuccessfulWorkerMeasurer)
    task = RealtimeObservationTask(
        obs_id=7,
        observation=Observation(data=object(), env_step=0, sim_time_ms=0.0),
        submit_env_step=0,
        submit_time_ms=0.0,
        observation_start_time_ms=0.0,
    )
    connection = RecordingWorkerConnection(task)
    connection.commands.insert(0, ("collect", {}))
    connection.commands.append(("close", {}))
    collect_calls = []
    real_collect = gc.collect
    monkeypatch.setattr(gc, "collect", lambda *args: collect_calls.append(args) or real_collect(*args))

    _inference_worker(
        connection,
        {"experiment": {"seed": 0}, "env": {}, "policy": {}, "executor": {"sync_cuda": True}},
        worker_id=0,
        device="cpu",
        start_method="spawn",
    )

    assert collect_calls
    assert [command for command, _payload in connection.sent] == ["ready", "result"]


def test_drain_until_idle_broadcasts_collect_to_idle_workers(monkeypatch):
    monkeypatch.setattr(ProcessInferencePool, "_start_workers", lambda self: None)
    pool = ProcessInferencePool(config={}, inference_devices=["cuda:0", "cuda:1"])

    class SendRecordingConnection:
        def __init__(self):
            self.sent = []

        def send(self, message):
            self.sent.append(message)

    connections = [SendRecordingConnection(), SendRecordingConnection()]
    pool._connections = list(connections)

    pool.drain_until_idle()

    assert all(connection.sent == [("collect", {})] for connection in connections)


def test_batch_inference_routes_slots_stably_and_restores_input_order():
    class BatchConnection:
        def __init__(self):
            self.sent = []

        def send(self, message):
            self.sent.append(message)

        def recv(self):
            command, payload = self.sent[-1]
            if command == "reset_state":
                return "state_reset", {}
            return "batch_result", {
                "policy_outputs": [
                    PolicyOutput(action=Action(observation.metadata["slot_id"]))
                    for observation in payload["observations"]
                ]
            }

    pool = ProcessInferencePool.__new__(ProcessInferencePool)
    pool.config = {"executor": {"inference_batch_size": 2}}
    pool.inference_devices = ["cuda:0", "cuda:1"]
    pool.closed = False
    pool._connections = [BatchConnection(), BatchConnection()]
    slots = [3, 0, 2, 1, 4]
    observations = [
        Observation(
            data=None,
            env_step=0,
            sim_time_ms=0.0,
            metadata={"slot_id": slot},
        )
        for slot in slots
    ]

    outputs = pool.predict_batch(observations)
    pool.reset_state(3)

    assert [output.action.value for output in outputs] == slots
    assert [
        [observation.metadata["slot_id"] for observation in payload["observations"]]
        for command, payload in pool._connections[0].sent
        if command == "predict_batch"
    ] == [[0, 2], [4]]
    assert pool._connections[1].sent[-1] == ("reset_state", {"slot_id": 3})
    assert pool._connections[0].sent[-1] == ("reset_state", {"slot_id": 3})
    del pool.config["executor"]["inference_batch_size"]
    with pytest.raises(KeyError, match="inference_batch_size"):
        pool.predict_batch(observations)


@pytest.mark.parametrize("ready_time_ms", [10.0, 100.0])
def test_realtime_prefix_is_frozen_before_completion(ready_time_ms):
    """Long-lived realtime request contract: measured completion cannot select K."""
    pool = ScriptedInferencePool([
        CompletionScript(Action([9.0]), ready_time_ms=ready_time_ms, worker_id=0, device="cuda:0")
    ], num_workers=1)
    executor = _executor(pool=pool, max_steps=1)
    executor.action_scheduler.noop_action = Action([7.0])
    executor.action_prefix = {"mode": "profile_mean", "frames": 2, "action_horizon": 4, "action_dim": 1}
    executor.run_episode(episode_id=0, seed=1, max_env_steps=1)
    metadata = pool.submitted[0].observation.metadata
    np.testing.assert_array_equal(metadata["action_prefix"][:, 0], [7, 7, 0, 0])
    np.testing.assert_array_equal(metadata["action_prefix_mask"], [True, True, False, False])


def test_environment_worker_freezes_prefix_before_shared_memory(monkeypatch):
    """Long-lived env-process boundary: the first transmitted observation includes K."""
    from latency_bench.executors.realtime.env_worker import _env_loop_worker
    from latency_bench.envs import factory as env_factory

    class Connection:
        snapshot = None

        def __init__(self):
            self.commands = [
                ("episode", {"episode_id": 0, "seed": 1, "max_env_steps": 1}),
                ("release_observation", {"obs_id": 0}),
                ("finish_episode", {}),
                ("close", {}),
            ]

        def poll(self, timeout=None):
            return False

        def recv(self):
            return self.commands.pop(0)

        def send(self, message):
            tag, payload = message
            if tag == "error":
                pytest.fail(payload["traceback"] if isinstance(payload, dict) else payload)
            if tag == "observation":
                cache = ObservationAttachmentCache(blocks={})
                try:
                    attached = cache.attach(payload["task"].observation)
                    self.snapshot = attached.metadata["action_prefix"].copy()
                    self.mask = attached.metadata["action_prefix_mask"].copy()
                finally:
                    cache.close()

    env = DummyEnvAdapter(env_fps=50, max_steps=1, noop_action=Action([7.0]))
    monkeypatch.setattr(env_factory, "build_env_adapter", lambda *args: env)
    config = {
        'env': {
            'env_fps': 50,
            'obs_fps': 50,
            'noop_action': [7.0],
            'action_space': {
                'type': 'box',
                'low': [0.0],
                'high': [10.0],
            },
        },
        'scheduler': {
            'hold_policy': 'hold',
            'ordering_policy': 'latest_ready',
        },
        'policy': {
            'action_prefix': {
                'mode': 'profile_mean',
                'frames': 2,
                'action_horizon': 4,
                'action_dim': 1,
            },
        },
        'logging': {
        },
    }
    connection = Connection()
    _env_loop_worker(connection, config, None, None)
    np.testing.assert_array_equal(connection.snapshot[:, 0], [7, 7, 0, 0])
    np.testing.assert_array_equal(connection.mask, [True, True, False, False])
    assert env.closed


def test_resident_environment_warms_first_step_and_resets_formal_episodes(monkeypatch):
    """Long-lived regression: cold step cost and warm state stay outside episodes."""
    from latency_bench.executors.realtime import env_worker
    from latency_bench.envs import factory as env_factory

    clock = FakeClock()

    class ColdEnv(DummyEnvAdapter):
        warmed = False

        def reset(self, seed=None):
            self.seed = seed
            return super().reset(seed)

        def observe(self):
            observation = super().observe()
            observation.metadata["seed"] = self.seed
            return observation

        def step(self, action):
            clock.now += 0.001 if self.warmed else 0.5
            self.warmed = True
            return super().step(action)

    env = ColdEnv(env_fps=50, max_steps=2, noop_action=Action(0))
    env.action_space = Discrete(2)
    monkeypatch.setattr(env_factory, "build_env_adapter", lambda *args: env)
    monkeypatch.setattr(env_worker, "time", SimpleNamespace(perf_counter=clock.time, sleep=clock.sleep))
    monkeypatch.setattr(env_worker, "monotonic_ms", lambda: clock.now * 1000)

    class Connection:
        def __init__(self):
            self.commands = [("observe", {}), ("release_observation", {"obs_id": 0}),
                             ("step", {"action": Action(1)})]
            for episode in range(2):
                self.commands.extend([
                    ("episode", {"episode_id": episode, "seed": 100 + episode, "max_env_steps": 2}),
                    ("release_observation", {"obs_id": 0}),
                    ("release_observation", {"obs_id": 1}), ("finish_episode", {}),
                ])
            self.commands.append(("close", {}))
            self.records, self.observations, self.warmup_results = [], [], []

        def recv(self):
            return self.commands.pop(0)

        def poll(self, timeout=None):
            if timeout is not None:
                clock.sleep(timeout)
            return False

        def send(self, message):
            tag, payload = message
            if tag == "error":
                pytest.fail(payload["traceback"])
            if tag == "observation":
                task = payload["task"]
                self.observations.append((task.observation.env_step, task.observation.metadata["seed"]))
            elif tag == "step":
                if "record" in payload:
                    self.records.append(payload["record"])
                else:
                    self.warmup_results.append(payload["result"])

    config = {
        'env': {
            'env_fps': 50,
            'obs_fps': 50,
            'noop_action': 0,
            'action_map': {
                'noop': 0,
                'jump': 1,
            },
        },
        'scheduler': {
            'hold_policy': 'hold',
            'ordering_policy': 'latest_ready',
        },
        'policy': {
            'action_prefix': {
                'mode': 'none',
            },
        },
        'logging': {
        },
    }
    connection = Connection()
    env_worker._env_loop_worker(connection, config, None, None)
    assert len(connection.warmup_results) == 1
    assert connection.observations == [(0, None), (0, 100), (1, 100), (0, 101), (1, 101)]
    assert [r.episode_id for r in connection.records] == [0, 0, 1, 1]
    assert max(r.info["wall_clock_lag_ms"] for r in connection.records) == pytest.approx(0)
    assert env.closed

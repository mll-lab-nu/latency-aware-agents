from __future__ import annotations

import gc
import json
from dataclasses import dataclass

import numpy as np
import pytest

import latency_bench.executors.realtime.executor as realtime_executor
import latency_bench.executors.realtime.inference_worker as realtime_inference_worker
import latency_bench.executors.realtime.pool as realtime_pool
from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, ActionEvent, LatencyRecord, Observation, OrderingPolicy, PolicyOutput, StepRecord
from latency_bench.envs.base import DummyEnvAdapter
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
from latency_bench.executors.realtime.shared_observation import ObservationSlotPool, attach_observation_arrays
from latency_bench.latency.samplers import ZeroSampler
from latency_bench.logging.realtime_pipeline import PipelineProfile
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.base import ConstantPolicyRunner
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
    profile: PipelineProfile | None = None


class ScriptedInferencePool:
    def __init__(self, scripts: list[CompletionScript], *, num_workers: int):
        self.scripts = list(scripts)
        self.num_workers = int(num_workers)
        self.submitted: list[RealtimeObservationTask] = []
        self.closed = False
        self._busy_worker_ids: set[int] = set()
        self._pending: list[tuple[RealtimeObservationTask, CompletionScript]] = []

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
        self.submitted.clear()
        self._busy_worker_ids.clear()
        self._pending.clear()
        return {
            "pre_episode_in_flight_count": 0,
            "pre_episode_drained_completion_count": 0,
            "pre_episode_drain_wait_ms": 0.0,
            "pre_episode_in_flight_after_drain": 0,
        }

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

    def drain_completed(self, now_ms: float) -> list[RealtimePolicyCompletion]:
        ready: list[RealtimePolicyCompletion] = []
        still_pending: list[tuple[RealtimeObservationTask, CompletionScript]] = []
        for task, script in self._pending:
            if script.ready_time_ms <= now_ms:
                self._busy_worker_ids.remove(script.worker_id)
                if script.profile is not None and task.profile is not None:
                    task.profile.merge_measured(script.profile)
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
                        worker_id=script.worker_id,
                        device=script.device,
                        profile=task.profile,
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


def test_official_gr00t_pool_uses_the_pinned_submodule_python(monkeypatch):
    starts = []

    class Connection:
        def recv(self):
            return "ready", {}

        def close(self):
            return None

    class Process:
        def start(self):
            starts.append(
                (realtime_pool.mp_spawn.get_executable(), realtime_pool.sys.path.copy())
            )

        def is_alive(self):
            return False

    class Context:
        def Pipe(self):
            return Connection(), Connection()

        def Process(self, **kwargs):
            return Process()

    monkeypatch.setattr(realtime_pool.mp, "get_context", lambda _method: Context())
    previous_executable = realtime_pool.mp_spawn.get_executable()
    previous_sys_path = realtime_pool.sys.path.copy()
    sys_path_object = realtime_pool.sys.path

    pool = ProcessInferencePool(
        config={"policy": {"type": "official_gr00t"}},
        inference_devices=["cuda:0"],
    )

    assert pool.worker_python_executable.endswith(
        "third_party/Isaac-GR00T/.venv/bin/python"
    )
    assert starts[0][0] == pool.worker_python_executable.encode()
    assert starts[0][1][0] == str(realtime_pool.OFFICIAL_GR00T_SITE_PACKAGES)
    assert realtime_pool.mp_spawn.get_executable() == previous_executable
    assert realtime_pool.sys.path is sys_path_object
    assert realtime_pool.sys.path == previous_sys_path


def test_process_inference_pool_does_not_accept_start_method(monkeypatch):
    monkeypatch.setattr(ProcessInferencePool, "_start_workers", lambda self: None)

    with pytest.raises(TypeError):
        ProcessInferencePool(config={}, inference_devices=["cuda:0"], start_method="fork")


def test_worker_runtime_metadata_records_spawn_context(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    metadata = _worker_runtime_metadata(worker_id=2, start_method="spawn")

    assert metadata["worker_id"] == 2
    assert metadata["worker_start_method"] == "spawn"
    assert metadata["worker_python_executable"]
    assert metadata["worker_torch_version"]
    assert "worker_torch_cuda_version" in metadata
    assert metadata["worker_cuda_visible_devices"] == "0,1"
    assert isinstance(metadata["worker_cuda_device_count"], int)
    assert "worker_cuda_flash_sdp_enabled" in metadata
    assert "worker_torch_float32_matmul_precision" in metadata


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


class ClosedActionEventPipeContext:
    def __init__(self, parent_conn: ClosedActionEventPipe):
        self.parent_conn = parent_conn

    def Pipe(self):
        return self.parent_conn, DummyChildPipe()

    def Process(self, *args, **kwargs):
        return AliveUntilTerminatedProcess(*args, **kwargs)


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


class QueuedEnvMessageContext:
    def __init__(self, parent_conn: QueuedEnvMessagePipe):
        self.parent_conn = parent_conn

    def Pipe(self):
        return self.parent_conn, DummyChildPipe()

    def Process(self, *args, **kwargs):
        return AliveUntilTerminatedProcess(*args, **kwargs)


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

    def submit(self, task: RealtimeObservationTask) -> bool:
        self.submitted.append(task)
        return True

    def drain_completed(self, now_ms: float) -> list[RealtimePolicyCompletion]:
        self.drain_calls += 1
        if self.drain_calls < 3:
            return []
        self.emitted = True
        task = self.submitted[0]
        return [
            RealtimePolicyCompletion(
                task=task,
                policy_output=PolicyOutput(action=Action("late")),
                completion_time_ms=now_ms,
                latency_record=LatencyRecord(obs_id=task.obs_id, latency_ms=1.0),
                worker_id=0,
                device="cpu",
                profile=task.profile,
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


class DummyWorkerPolicy:
    def predict(self, observation):
        return PolicyOutput(action=Action("unused"))


class SuccessfulWorkerMeasurer:
    def __init__(self, *, device: str, sync_cuda: bool):
        self.device = device
        self.sync_cuda = sync_cuda

    def run_and_measure(self, predict, observation, profile, *, obs_id: int):
        return (
            PolicyOutput(action=Action("jump")),
            LatencyRecord(obs_id=obs_id, latency_ms=1.0),
        )


class FailingWorkerMeasurer:
    def __init__(self, *, device: str, sync_cuda: bool):
        self.device = device
        self.sync_cuda = sync_cuda

    def run_and_measure(self, predict, observation, profile, *, obs_id: int):
        raise RuntimeError("policy boom")


class CallingWorkerMeasurer:
    def __init__(self, *, device: str, sync_cuda: bool):
        self.device = device
        self.sync_cuda = sync_cuda

    def run_and_measure(self, predict, observation, profile, *, obs_id: int):
        return predict(observation, profile), LatencyRecord(obs_id=obs_id, latency_ms=1.0)


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
) -> RealtimeLatencyExecutor:
    fake_clock = fake_clock or FakeClock()
    noop = Action("noop", name="noop", is_noop=True)
    env = DummyEnvAdapter(env_fps=env_fps, max_steps=max_steps, noop_action=noop)
    clock = EnvClock(env_fps=env_fps, obs_fps=obs_fps)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop,
        persistence_policy=HoldPolicy(duration_raw_frames=clock.obs_stride_raw_frames),
        ordering_policy=ordering_policy,
    )
    runtime = RealtimeRuntime(
        env=env,
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_source=ZeroSampler(clock=clock),
        action_scheduler=action_scheduler,
        clock=clock,
    )
    return RealtimeLatencyExecutor(
        runtime=runtime,
        runtime_factory=lambda: runtime,
        policy=ConstantPolicyRunner(Action("unused")),
        logger=logger,
        time_fn=fake_clock.time,
        sleep_fn=fake_clock.sleep,
        inference_pool=pool,
        video_recorder=video_recorder,
        config=config,
    )


def test_restart_runtime_rebuilds_owned_resources_and_preserves_session_state(
    monkeypatch,
    tmp_path,
):
    fake_clock = FakeClock()
    runtimes = []
    noop = Action("noop", name="noop", is_noop=True)

    def build_runtime():
        env = DummyEnvAdapter(env_fps=30.0, max_steps=2, noop_action=noop)
        clock = EnvClock(env_fps=30.0, obs_fps=30.0)
        runtime = RealtimeRuntime(
            env=env,
            decision_scheduler=ObservationCadenceDecisionScheduler(clock),
            latency_source=ZeroSampler(clock=clock),
            action_scheduler=ActionScheduler(
                clock=clock,
                noop_action=noop,
                persistence_policy=HoldPolicy(duration_raw_frames=clock.obs_stride_raw_frames),
                ordering_policy=OrderingPolicy.LATEST_READY,
            ),
            clock=clock,
        )
        runtimes.append(runtime)
        return runtime

    logger = TrajectoryLogger(
        tmp_path,
        save_step_records=True,
        save_action_records=True,
        save_latency_records=True,
    )
    old_pool = ScriptedInferencePool([], num_workers=1)
    new_pool = ScriptedInferencePool([], num_workers=1)
    monkeypatch.setattr(realtime_executor, "ProcessInferencePool", lambda **_kwargs: new_pool)
    executor = RealtimeLatencyExecutor(
        runtime=build_runtime(),
        runtime_factory=build_runtime,
        policy=None,
        logger=logger,
        time_fn=fake_clock.time,
        sleep_fn=fake_clock.sleep,
        inference_pool=old_pool,
        config={"logging": {"output_dir": str(tmp_path)}},
        inference_devices=["cuda:0"],
    )
    executor._next_obs_id = 17
    executor._next_action_id = 11

    executor.restart_runtime(60)

    assert old_pool.closed
    assert runtimes[0].env.closed
    assert fake_clock.sleeps == [60.0]
    assert executor.env is runtimes[1].env
    assert executor.inference_pool is None
    assert executor.logger is logger
    assert executor._next_obs_id == 17
    assert executor._next_action_id == 11
    assert executor._ensure_inference_pool() is new_pool

    executor.close()

    assert new_pool.closed
    assert runtimes[1].env.closed


def test_env_process_ignores_late_action_when_pipe_is_closed(monkeypatch):
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

    monkeypatch.setattr(realtime_executor.mp, "get_context", lambda _method: ClosedActionEventPipeContext(parent_conn))
    monkeypatch.setattr(executor, "_drain_and_release_results", fake_drain_and_release_results)

    metrics = executor._run_episode_with_env_process(
        pool=pool,
        episode_id=0,
        seed=0,
        eval_max_steps=1,
    )

    assert parent_conn.closed
    assert metrics.num_actions == 1
    assert metrics.mean_latency_ms == 1.0


def test_env_process_drains_ready_results_before_submitting_observation(monkeypatch):
    task = RealtimeObservationTask(
        obs_id=0,
        observation=Observation(data=None, env_step=0, sim_time_ms=0.0),
        submit_env_step=0,
        submit_time_ms=0.0,
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
                        info={"wall_clock_lag_ms": 0.0},
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

    monkeypatch.setattr(realtime_executor.mp, "get_context", lambda _method: QueuedEnvMessageContext(parent_conn))
    monkeypatch.setattr(executor, "_drain_and_release_results", fake_drain_and_release_results)

    metrics = executor._run_episode_with_env_process(
        pool=pool,
        episode_id=0,
        seed=0,
        eval_max_steps=1,
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
                        info={"wall_clock_lag_ms": 0.0},
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

    monkeypatch.setattr(realtime_executor.mp, "get_context", lambda _method: QueuedEnvMessageContext(parent_conn))

    metrics = executor._run_episode_with_env_process(
        pool=pool,
        episode_id=0,
        seed=0,
        eval_max_steps=1,
    )

    assert metrics.num_actions == 1
    assert executor.action_scheduler.get_pending_events() == []
    assert executor.action_scheduler.get_applied_events() == []
    assert ("release_observation", {"obs_id": 0}) in parent_conn.sent


def test_env_process_releases_in_flight_observations_before_close_after_episode_done(monkeypatch):
    task = RealtimeObservationTask(
        obs_id=0,
        observation=Observation(data=None, env_step=0, sim_time_ms=0.0),
        submit_env_step=0,
        submit_time_ms=0.0,
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
                        info={"wall_clock_lag_ms": 0.0},
                    )
                },
            ),
            ("episode_done", {}),
        ]
    )
    pool = LateCompletionAfterFinalizePool()
    executor = _executor(pool=pool, max_steps=1)

    monkeypatch.setattr(realtime_executor.mp, "get_context", lambda _method: QueuedEnvMessageContext(parent_conn))

    metrics = executor._run_episode_with_env_process(
        pool=pool,
        episode_id=0,
        seed=0,
        eval_max_steps=1,
    )

    assert metrics.num_actions == 0
    assert parent_conn.sent.index(("release_observation", {"obs_id": 42})) < parent_conn.sent.index(("close", {}))


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
    )
    connection = BrokenResultConnection(task)

    _inference_worker(
        connection,
        {"env": {}, "policy": {}, "latency": {}},
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
    )
    connection = RecordingWorkerConnection(task)
    connection.commands.append(("close", {}))

    _inference_worker(
        connection,
        {"env": {}, "policy": {}, "latency": {}},
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


def test_inference_worker_predicts_from_worker_local_observation_copy(monkeypatch):
    import latency_bench.core.actions as actions_mod
    import latency_bench.policy.factory as factory_mod

    class DummyActionResolver:
        def __init__(self, env_config):
            self.env_config = env_config

    class MutatingWorkerPolicy:
        def predict(self, observation):
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
        )
        connection = RecordingWorkerConnection(task)
        connection.commands.append(("close", {}))

        _inference_worker(
            connection,
            {"env": {}, "policy": {}, "latency": {}},
            worker_id=0,
            device="cpu",
            start_method="spawn",
        )

        attached = attach_observation_arrays(shared_observation)
        try:
            np.testing.assert_array_equal(attached.observation.data, observation.data)
            np.testing.assert_array_equal(
                attached.observation.metadata["env_raw_rgb_frame_stack"],
                observation.metadata["env_raw_rgb_frame_stack"],
            )
        finally:
            attached.close()
    finally:
        owner.close()


def test_action_dispatch_and_ack_payloads_carry_realtime_profile() -> None:
    event = ActionEvent(
        action_id=11,
        action=Action("jump"),
        obs_env_step=4,
        ready_time_ms=120.0,
        ready_step_float=3.6,
        latency=LatencyRecord(obs_id=5, latency_ms=20.0),
        metadata={"realtime_profile": PipelineProfile(parent_release_ms=1.5)},
    )
    event.latency_raw_frames = 2
    event.ready_raw_frame = 6

    payload = _action_event_dispatch_payload(event)
    dispatched = _action_event_from_dispatch_payload(payload)
    dispatched.applied = True
    dispatched.apply_raw_frame = 7
    ack = _action_event_ack_payload(dispatched)

    assert payload["profile"] == {"parent_release_ms": 1.5}
    assert dispatched.profile.to_dict() == {"parent_release_ms": 1.5}
    assert ack["profile"] == {"parent_release_ms": 1.5}


def test_env_action_drain_happens_before_scheduler_update() -> None:
    clock = EnvClock(env_fps=10.0, obs_fps=10.0)
    scheduler = ActionScheduler(
        clock=clock,
        noop_action=Action("noop", name="noop", is_noop=True),
        persistence_policy=HoldPolicy(duration_raw_frames=1),
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
    )
    event = ActionEvent(
        action_id=0,
        action=Action("jump"),
        obs_env_step=0,
        ready_time_ms=0.0,
        ready_step_float=0.0,
        latency=LatencyRecord(obs_id=0, latency_ms=0.0),
        metadata={"realtime_profile": PipelineProfile(parent_to_env_send_start_ms=9.0)},
    )
    event.latency_raw_frames = 0
    event.ready_raw_frame = 0
    pipe = QueuedEnvMessagePipe([("action_dispatch", _action_event_dispatch_payload(event))])

    closed = _drain_env_action_messages(
        connection=pipe,
        action_scheduler=scheduler,
        release_observation=lambda obs_id: None,
        now_ms_fn=lambda: 10.0,
    )
    active_action = scheduler.update(0, 0.0)
    applied = scheduler.get_applied_events()[0]

    assert closed is False
    assert active_action == Action("jump")
    assert applied.profile.parent_to_env_ipc_ms == 1.0
    assert applied.profile.env_action_recv_to_enqueue_ms is not None


def test_env_action_drain_releases_observations_with_explicit_callback() -> None:
    clock = EnvClock(env_fps=10.0, obs_fps=10.0)
    scheduler = ActionScheduler(
        clock=clock,
        noop_action=Action("noop", name="noop", is_noop=True),
        persistence_policy=HoldPolicy(duration_raw_frames=1),
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
        persistence_policy=HoldPolicy(duration_raw_frames=1),
        ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
    )
    event = ActionEvent(
        action_id=0,
        action=Action("jump"),
        obs_env_step=0,
        ready_time_ms=0.0,
        ready_step_float=0.0,
        latency=LatencyRecord(obs_id=0, latency_ms=0.0),
        metadata={"realtime_profile": PipelineProfile(parent_to_env_send_start_ms=999.0)},
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
        now_ms_fn=lambda: 1000.0,
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
        profile=PipelineProfile(parent_observation_handle_ms=2.0),
    )
    result = RealtimePolicyResult(
        policy_output=PolicyOutput(action=Action("jump")),
        completion_time_ms=135.0,
        latency_record=LatencyRecord(obs_id=11, latency_ms=15.0),
        profile=PipelineProfile(worker_recv_overhead_ms=3.0, worker_result_send_start_ms=17.0),
    )
    pool = ProcessInferencePool.__new__(ProcessInferencePool)
    pool.closed = False
    pool._connections = [ReadyResultConnection(result)]
    pool.inference_devices = ["cpu"]
    pool._busy_worker_tasks = {0: task}

    completions = pool._drain_all_ready()

    assert len(completions) == 1
    assert completions[0].task is task
    assert completions[0].policy_output.action.value == "jump"
    assert completions[0].worker_id == 0
    assert completions[0].device == "cpu"
    assert completions[0].profile.parent_observation_handle_ms == 2.0
    assert completions[0].profile.worker_recv_overhead_ms == 3.0
    assert completions[0].profile.worker_to_parent_ipc_ms == 3.0


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


def test_inference_worker_reports_real_errors_when_pipe_is_open(monkeypatch):
    _patch_worker_dependencies(monkeypatch, FailingWorkerMeasurer)
    task = RealtimeObservationTask(
        obs_id=8,
        observation=Observation(data=object(), env_step=0, sim_time_ms=0.0),
        submit_env_step=0,
        submit_time_ms=0.0,
    )
    connection = RecordingWorkerConnection(task)

    _inference_worker(
        connection,
        {"env": {}, "policy": {}, "latency": {}},
        worker_id=0,
        device="cpu",
        start_method="spawn",
    )

    assert [command for command, _payload in connection.sent] == ["ready", "error"]
    assert "policy boom" in connection.sent[-1][1]["traceback"]


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

    metrics = executor.run_episode(episode_id=0, eval_max_steps=7)

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

    metrics = executor.run_episode(episode_id=0, eval_max_steps=3)

    assert [task.submit_env_step for task in pool.submitted] == [0]
    assert metrics.metadata["submitted_observation_frames"] == 1
    assert metrics.metadata["dropped_observation_count"] == 2
    assert executor.logger is None


def test_realtime_records_inference_worker_count_in_episode_metadata():
    pool = ScriptedInferencePool(
        [
            CompletionScript(Action("a0"), ready_time_ms=1000.0, worker_id=0, device="cpu"),
            CompletionScript(Action("a1"), ready_time_ms=1000.0, worker_id=1, device="cpu"),
        ],
        num_workers=2,
    )
    executor = _executor(pool=pool, max_steps=2)

    metrics = executor.run_episode(episode_id=0, eval_max_steps=2)

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

    metrics = executor.run_episode(episode_id=0, eval_max_steps=4)

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

    metrics = executor.run_episode(episode_id=0, eval_max_steps=5)

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

    executor.run_episode(episode_id=0, eval_max_steps=2)

    # Records are flushed to disk (and cleared from memory) at episode end.
    latency = json.loads((tmp_path / "latencies.jsonl").read_text(encoding="utf-8").splitlines()[0])
    event = json.loads((tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert latency["device"] == "cuda:1"
    assert latency["metadata"]["worker_id"] == 1
    assert "worker_id" not in event["metadata"]
    assert "device" not in event["metadata"]


def test_realtime_records_current_episode_video(tmp_path):
    from latency_bench.logging.video import EpisodeVideoRecorder

    pool = ScriptedInferencePool(
        [CompletionScript(Action("jump"), ready_time_ms=1.0, worker_id=0, device="cpu")],
        num_workers=1,
    )
    recorder = EpisodeVideoRecorder(tmp_path / "videos", fps=30.0)
    executor = _executor(pool=pool, max_steps=2, video_recorder=recorder)

    executor.run_episode(episode_id=5, eval_max_steps=2)

    manifest = [
        json.loads(line)
        for line in (tmp_path / "videos" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(manifest) == 1
    assert manifest[0]["episode_id"] == 5
    assert manifest[0]["episode_return"] == 2.0
    assert manifest[0]["frame_count"] == 2
    assert (tmp_path / "videos" / manifest[0]["video_path"]).is_file()


def test_realtime_warmup_writes_separate_artifacts(monkeypatch, tmp_path):
    from latency_bench.executors import realtime_warmup

    monkeypatch.setattr(realtime_warmup, "MIN_WARMUP_LATENCY_SAMPLES", 3)
    monkeypatch.setattr(realtime_warmup, "MAX_WARMUP_LATENCY_SAMPLES", 3)
    monkeypatch.setattr(realtime_warmup, "WARMUP_WINDOW_SIZE", 1)
    monkeypatch.setattr(realtime_warmup, "STABLE_WINDOW_COUNT", 3)

    action = Action("jump")
    profile = PipelineProfile(
        worker_observation_attach_ms=0.1,
        worker_observation_localize_ms=0.2,
        policy_build_example_ms=0.3,
        policy_wrapper_predict_action_ms=0.4,
        policy_decode_ms=0.5,
        worker_recv_overhead_ms=0.6,
        worker_to_parent_ipc_ms=0.7,
        parent_to_worker_ipc_ms=0.8,
    )
    pool = ScriptedInferencePool(
        [
            CompletionScript(action, ready_time_ms=1.0, worker_id=0, device="cuda:0", profile=profile),
            CompletionScript(action, ready_time_ms=1.0, worker_id=0, device="cuda:0", profile=profile),
            CompletionScript(action, ready_time_ms=1.0, worker_id=0, device="cuda:0", profile=profile),
        ],
        num_workers=1,
    )
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
        config={"logging": {"output_dir": str(tmp_path)}},
    )

    warmup_dir = tmp_path / "warmup/episode_000000"
    summary = executor.run_warmup(artifact_dir=warmup_dir)

    assert summary["warmup_mode"] == "stress_policy"
    assert summary["warmup_status"] == "stable"
    assert summary["warmup_samples"] == 3
    assert summary["warmup_stage_summary"]["policy_wrapper_predict_action_ms"]["p50"] == pytest.approx(0.4)
    assert len(pool.submitted) == 3
    assert len(formal_logger.latency_records) == 0
    assert (warmup_dir / "warmup_latencies.jsonl").is_file()
    assert (warmup_dir / "warmup_pipeline_profile.jsonl").is_file()
    assert (warmup_dir / "warmup_summary.json").is_file()
    assert (warmup_dir / "warmup_stability.png").is_file()
    assert not (tmp_path / "warmup_summary.json").exists()
    assert not (tmp_path / "latencies.jsonl").exists()


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


def test_realtime_stress_warmup_does_not_run_episodes(monkeypatch, tmp_path):
    from latency_bench.executors import realtime_warmup

    monkeypatch.setattr(realtime_warmup, "MIN_WARMUP_LATENCY_SAMPLES", 2)
    monkeypatch.setattr(realtime_warmup, "MAX_WARMUP_LATENCY_SAMPLES", 2)
    monkeypatch.setattr(realtime_warmup, "WARMUP_WINDOW_SIZE", 1)
    monkeypatch.setattr(realtime_warmup, "STABLE_WINDOW_COUNT", 2)

    action = Action("jump")
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

    def fail_run_episode(*, episode_id, seed, eval_max_steps):
        del episode_id, seed, eval_max_steps
        raise AssertionError("stress warmup must not run realtime episodes")

    executor.run_episode = fail_run_episode

    summary = executor.run_warmup()

    assert summary["warmup_mode"] == "stress_policy"
    assert summary["warmup_samples"] == 2


def test_gc_paused_until_episode_end_restores_state_and_collects():
    from latency_bench.executors.realtime.executor import _gc_paused_until_episode_end

    assert gc.isenabled()
    with _gc_paused_until_episode_end():
        assert not gc.isenabled()
    assert gc.isenabled()


def test_run_episode_leaves_gc_enabled():
    pool = ScriptedInferencePool([], num_workers=1)
    executor = _executor(pool=pool, max_steps=3)

    executor.run_episode(episode_id=0, seed=0, eval_max_steps=3)

    assert gc.isenabled()


def test_inference_worker_collect_command_runs_gc_and_keeps_serving(monkeypatch):
    _patch_worker_dependencies(monkeypatch, SuccessfulWorkerMeasurer)
    task = RealtimeObservationTask(
        obs_id=7,
        observation=Observation(data=object(), env_step=0, sim_time_ms=0.0),
        submit_env_step=0,
        submit_time_ms=0.0,
    )
    connection = RecordingWorkerConnection(task)
    connection.commands.insert(0, ("collect", {}))
    connection.commands.append(("close", {}))
    collect_calls = []
    real_collect = gc.collect
    monkeypatch.setattr(gc, "collect", lambda *args: collect_calls.append(args) or real_collect(*args))

    _inference_worker(
        connection,
        {"env": {}, "policy": {}, "latency": {}},
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

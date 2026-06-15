from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, LatencyRecord, OrderingPolicy, PolicyOutput
from latency_bench.envs.base import DummyEnvAdapter
from latency_bench.executors.realtime import (
    ProcessInferencePool,
    RealtimeLatencyExecutor,
    RealtimeObservationTask,
    RealtimePolicyCompletion,
    _worker_runtime_metadata,
)
from latency_bench.latency.sources import ZeroLatencySource
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

    def reset_episode(self) -> None:
        self.submitted.clear()
        self._busy_worker_ids.clear()
        self._pending.clear()

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

    monkeypatch.setattr("latency_bench.executors.realtime.mp.get_context", fake_get_context)
    monkeypatch.setattr(ProcessInferencePool, "_start_workers", lambda self: None)

    pool = ProcessInferencePool(config={}, inference_devices=["cuda:0"])

    assert requested_methods == ["spawn"]
    assert pool.start_method == "spawn"


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
    return RealtimeLatencyExecutor(
        env=env,
        policy=ConstantPolicyRunner(Action("unused")),
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_source=ZeroLatencySource(),
        action_scheduler=action_scheduler,
        clock=clock,
        logger=logger,
        time_fn=fake_clock.time,
        sleep_fn=fake_clock.sleep,
        inference_pool=pool,
        video_recorder=video_recorder,
        config=config,
    )


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

    latency = logger.latency_records[0]
    event = logger.action_records[0]
    assert latency.device == "cuda:1"
    assert latency.metadata["worker_id"] == 1
    assert "worker_id" not in event.metadata
    assert "device" not in event.metadata


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
    monkeypatch.setattr(realtime_warmup, "MAX_WARMUP_LATENCY_SAMPLES", 4)
    monkeypatch.setattr(realtime_warmup, "WARMUP_WINDOW_SIZE", 1)
    monkeypatch.setattr(realtime_warmup, "STABLE_WINDOW_COUNT", 3)

    action = Action("jump")
    pool = ScriptedInferencePool(
        [CompletionScript(action, ready_time_ms=1.0, worker_id=0, device="cuda:0")],
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

    summary = executor.run_warmup(eval_max_steps=2)

    assert summary["warmup_status"] == "stable"
    assert summary["warmup_samples"] == 3
    assert len(formal_logger.latency_records) == 0
    assert (tmp_path / "warmup_latencies.jsonl").is_file()
    assert (tmp_path / "warmup_summary.json").is_file()
    assert (tmp_path / "warmup_stability.png").is_file()
    assert not (tmp_path / "latencies.jsonl").exists()


def test_realtime_warmup_uses_actual_samples_across_short_episodes(monkeypatch, tmp_path):
    from latency_bench.executors import realtime_warmup

    monkeypatch.setattr(realtime_warmup, "MIN_WARMUP_LATENCY_SAMPLES", 3)
    monkeypatch.setattr(realtime_warmup, "MAX_WARMUP_LATENCY_SAMPLES", 4)
    monkeypatch.setattr(realtime_warmup, "WARMUP_WINDOW_SIZE", 1)
    monkeypatch.setattr(realtime_warmup, "STABLE_WINDOW_COUNT", 3)

    action = Action("jump")
    pool = ScriptedInferencePool(
        [CompletionScript(action, ready_time_ms=0.0, worker_id=0, device="cuda:0")],
        num_workers=1,
    )
    executor = _executor(
        pool=pool,
        max_steps=1,
        config={"logging": {"output_dir": str(tmp_path)}},
    )

    summary = executor.run_warmup(eval_max_steps=4)

    assert summary["warmup_status"] == "stable"
    assert summary["warmup_samples"] == 3

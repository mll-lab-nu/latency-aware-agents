from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pytest

from latency_bench.core.clock import EnvClock
from latency_bench.core.timing import current_profiler
from latency_bench.core.types import Action, Observation, OrderingPolicy, PolicyOutput, StepResult
from latency_bench.envs.base import DummyEnvAdapter, EnvAdapter
from latency_bench.envs.factory import register_env_adapter
from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor
from latency_bench.executors.env_step_backend import SerialEnvStepBackend
from latency_bench.latency.sample import LatencySample
from latency_bench.latency.samplers import FixedSampler, LatencySampler, ZeroSampler
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.utils.io import load_jsonl_records
from latency_bench.policy.base import ConstantPolicyRunner, PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler
from latency_bench.scheduler.persistence import HoldPolicy
from latency_bench.envs.raw_rgb import ENV_RAW_RGB_FRAME_STACK_INFO_KEY

_CLOCK = EnvClock(env_fps=30.0, obs_fps=30.0)


class SlotIdPolicy(PolicyRunner):
    def predict(self, observation: Observation) -> PolicyOutput:
        raise AssertionError("batched executor must call predict_batch")

    def predict_batch(self, observations):
        return [
            PolicyOutput(
                action=Action(
                    value=f"slot-{observation.metadata['slot_index']}",
                    name=f"slot-{observation.metadata['slot_index']}",
                )
            )
            for observation in observations
        ]


class SlotLatencySource(LatencySampler):
    def __init__(self, slot_id: int, *, clock: EnvClock = _CLOCK):
        self.slot_id = int(slot_id)
        self.calls = 0
        self.clock = clock

    def sample(self, context=None) -> LatencySample:
        self.calls += 1
        return LatencySample.from_latency_ms(float(self.slot_id), clock=self.clock)


class RawFrameLatencySource(LatencySampler):
    def __init__(self, raw_frames: list[int], *, clock: EnvClock = _CLOCK):
        self.raw_frames = list(raw_frames)
        self.calls: list[dict] = []
        self.clock = clock

    def sample(self, context=None) -> LatencySample:
        self.calls.append(dict(context or {}))
        raw = self.raw_frames.pop(0)
        return LatencySample.from_latency_ms(raw * self.clock.frame_ms, clock=self.clock, latency_raw_frames=raw)


class TupleActionPolicy(PolicyRunner):
    def predict(self, observation: Observation) -> PolicyOutput:
        raise AssertionError("batched executor must call predict_batch")

    def predict_batch(self, observations):
        return [
            PolicyOutput(
                action=Action(
                    value=(
                        int(observation.metadata["slot_index"]),
                        int(observation.env_step),
                    ),
                    name=str((int(observation.metadata["slot_index"]), int(observation.env_step))),
                )
            )
            for observation in observations
        ]


class ResetTrackingBatchPolicy(PolicyRunner):
    def __init__(self, action: Action):
        self.action = action
        self.reset_calls: list[int | None] = []

    def predict(self, observation: Observation) -> PolicyOutput:
        raise AssertionError("batched executor must call predict_batch")

    def predict_batch(self, observations):
        return [PolicyOutput(action=self.action) for _observation in observations]

    def reset_state(self, slot_id: int | None = None) -> None:
        self.reset_calls.append(slot_id)


class MetadataCapturePolicy(PolicyRunner):
    def __init__(self):
        self.batches: list[list[dict]] = []

    def predict(self, observation: Observation) -> PolicyOutput:
        raise AssertionError("batched executor must call predict_batch")

    def predict_batch(self, observations):
        self.batches.append([dict(observation.metadata) for observation in observations])
        return [
            PolicyOutput(action=Action("jump", name="jump"))
            for _observation in observations
        ]


class TimingBatchPolicy(PolicyRunner):
    """Records fixed per-stage timings into the ambient profiler (like StarVla)."""

    def predict(self, observation: Observation) -> PolicyOutput:
        raise AssertionError("batched executor must call predict_batch")

    def predict_batch(self, observations):
        profiler = current_profiler()
        if profiler.enabled:
            profiler.timings.update(
                {
                    "policy_build_example_ms": 1.0,
                    "policy_wrapper_predict_action_ms": 2.0,
                    "policy_decode_ms": 3.0,
                    "starvla_fake_internal_ms": 4.0,
                }
            )
        return [
            PolicyOutput(action=Action("jump", name="jump"))
            for _observation in observations
        ]


class RawRgbShapeCapturePolicy(PolicyRunner):
    def __init__(self):
        self.raw_stack_shapes: list[tuple[int, ...]] = []

    def predict(self, observation: Observation) -> PolicyOutput:
        raise AssertionError("batched executor must call predict_batch")

    def predict_batch(self, observations):
        outputs = []
        for observation in observations:
            raw_stack = np.asarray(observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY])
            self.raw_stack_shapes.append(tuple(raw_stack.shape))
            outputs.append(
                PolicyOutput(
                    action=Action("jump", name="jump"),
                    metadata={
                        "input_frame_shape": [list(raw_stack[-1].shape)],
                    },
                )
            )
        return outputs


class TupleActionEnvAdapter(EnvAdapter):
    def __init__(self, *, slot_id: int, max_steps: int, noop_action: Action):
        import gymnasium as gym

        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(4), gym.spaces.Discrete(4)))
        self.gym_env = self
        self.slot_id = int(slot_id)
        self.env_fps = 30.0
        self.max_steps = int(max_steps)
        self.noop_action = noop_action
        self.env_step = 0
        self.applied_actions: list[Action] = []

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        self.applied_actions.clear()
        return self._observation(seed=seed)

    def step(self, action: Action) -> StepResult:
        self.applied_actions.append(action)
        self.env_step += 1
        return StepResult(
            observation=self._observation(),
            reward=1.0,
            done=self.env_step >= self.max_steps,
            truncated=False,
            info={"applied_action": action.value},
        )

    def observe(self) -> Observation:
        return self._observation()

    def _observation(self, seed: int | None = None) -> Observation:
        metadata = {"slot_index": self.slot_id}
        if seed is not None:
            metadata["seed"] = seed
        return Observation(
            data={"state": self.env_step},
            env_step=self.env_step,
            sim_time_ms=self.env_step * (1000.0 / self.env_fps),
            metadata=metadata,
        )


class RawRgbStepInfoSlotEnvAdapter(TupleActionEnvAdapter):
    def step(self, action: Action) -> StepResult:
        self.applied_actions.append(action)
        self.env_step += 1
        return StepResult(
            observation=self._observation(),
            reward=1.0,
            done=self.env_step >= self.max_steps,
            truncated=False,
            info={
                "applied_action": action.value,
                ENV_RAW_RGB_FRAME_STACK_INFO_KEY: np.full((1, 2, 3, 3), self.slot_id, dtype=np.uint8),
            },
        )

    def _observation(self, seed: int | None = None) -> Observation:
        observation = super()._observation(seed=seed)
        observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY] = np.full(
            (1, 2, 3, 3),
            self.slot_id + 7,
            dtype=np.uint8,
        )
        return observation


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


class FailingStepEnvAdapter(DummyEnvAdapter):
    def step(self, action: Action) -> StepResult:
        raise RuntimeError("subproc step boom")


class MultiDiscreteSubprocEnvAdapter(DummyEnvAdapter):
    def __init__(self, *, env_fps: float, max_steps: int, noop_action: Action):
        import gymnasium as gym

        super().__init__(env_fps=env_fps, max_steps=max_steps, noop_action=noop_action)
        self.action_space = gym.spaces.MultiDiscrete([2, 3])
        self.gym_env = self


class DiscreteSubprocEnvAdapter(DummyEnvAdapter):
    def __init__(self, *, env_fps: float, max_steps: int, noop_action: Action):
        import gymnasium as gym

        super().__init__(env_fps=env_fps, max_steps=max_steps, noop_action=noop_action)
        self.action_space = gym.spaces.Discrete(3)
        self.gym_env = self


class RawRgbMetadataSubprocEnvAdapter(EnvAdapter):
    def __init__(self, *, env_fps: float, max_steps: int, noop_action: Action):
        self.env_fps = float(env_fps)
        self.max_steps = int(max_steps)
        self.noop_action = noop_action
        self.env_step = 0

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.env_fps

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        return self.observe()

    def step(self, action: Action) -> StepResult:
        self.env_step += 1
        return StepResult(
            observation=self.observe(),
            reward=1.0,
            done=self.env_step >= self.max_steps,
            truncated=False,
            info={"applied_action": action.value},
        )

    def observe(self) -> Observation:
        raw_stack = np.full((4, 5, 6, 3), self.env_step + 11, dtype=np.uint8)
        return Observation(
            data=np.zeros((4, 84, 84), dtype=np.uint8),
            env_step=self.env_step,
            sim_time_ms=self.env_step * self.frame_ms,
            metadata={ENV_RAW_RGB_FRAME_STACK_INFO_KEY: raw_stack},
        )


@register_env_adapter("subproc_failure_test")
def _build_subproc_failure_test_env(config: dict, noop_action: Action) -> EnvAdapter:
    return FailingStepEnvAdapter(
        env_fps=float(config["env"]["env_fps"]),
        max_steps=int(config["evaluation"]["eval_max_steps"]),
        noop_action=noop_action,
    )


@register_env_adapter("discrete_subproc_test")
def _build_discrete_subproc_test_env(config: dict, noop_action: Action) -> EnvAdapter:
    return DiscreteSubprocEnvAdapter(
        env_fps=float(config["env"]["env_fps"]),
        max_steps=int(config["evaluation"]["eval_max_steps"]),
        noop_action=noop_action,
    )


@register_env_adapter("multidiscrete_subproc_test")
def _build_multidiscrete_subproc_test_env(config: dict, noop_action: Action) -> EnvAdapter:
    return MultiDiscreteSubprocEnvAdapter(
        env_fps=float(config["env"]["env_fps"]),
        max_steps=int(config["evaluation"]["eval_max_steps"]),
        noop_action=noop_action,
    )


@register_env_adapter("raw_rgb_metadata_subproc_test")
def _build_raw_rgb_metadata_subproc_test_env(config: dict, noop_action: Action) -> EnvAdapter:
    return RawRgbMetadataSubprocEnvAdapter(
        env_fps=float(config["env"]["env_fps"]),
        max_steps=int(config["evaluation"]["eval_max_steps"]),
        noop_action=noop_action,
    )


def _executor(
    tmp_path: Path,
    *,
    num_slots: int = 2,
    env_max_steps: int = 2,
    policy: PolicyRunner | None = None,
    latency_sources: list[LatencySampler] | None = None,
    simulated_worker_capacity: int | None = None,
    ordering_policy: OrderingPolicy = OrderingPolicy.ISSUE_ORDER_FIFO,
) -> tuple[BatchedSimulatedLatencyExecutor, list[DummyEnvAdapter]]:
    noop = Action("noop", name="noop", is_noop=True)
    envs = [
        DummyEnvAdapter(env_fps=30.0, max_steps=env_max_steps, noop_action=noop)
        for _ in range(num_slots)
    ]
    for slot, env in enumerate(envs):
        original_make_observation = env._make_observation

        def make_observation(metadata=None, *, _slot=slot, _original=original_make_observation):
            metadata = dict(metadata or {})
            metadata["slot_index"] = _slot
            return _original(metadata=metadata)

        env._make_observation = make_observation

    clock = EnvClock(env_fps=30.0, obs_fps=30.0)
    action_schedulers = [
        ActionScheduler(
            clock=clock,
            noop_action=env.noop_action,
            persistence_policy=HoldPolicy(),
            ordering_policy=ordering_policy,
        )
        for env in envs
    ]
    backend = SerialEnvStepBackend(envs)
    executor = BatchedSimulatedLatencyExecutor(
        env_backend=backend,
        policy=policy or ConstantPolicyRunner(Action("jump", name="jump")),
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_sources=latency_sources or [ZeroSampler(clock=_CLOCK) for _ in envs],
        action_schedulers=action_schedulers,
        clock=clock,
        logger=TrajectoryLogger(
            tmp_path,
            save_step_records=True,
            save_action_records=True,
            save_latency_records=True,
        ),
        simulated_worker_capacity=simulated_worker_capacity,
    )
    return executor, envs


def _subproc_config(
    tmp_path: Path,
    *,
    env_name: str = "dummy",
    num_episodes: int = 4,
    eval_max_steps: int = 2,
    noop_value="noop",
    jump_value="jump",
) -> dict:
    return {
        "experiment": {
            "name": "subproc_batched_test",
            "seed": 0,
        },
        "executor": {
            "mode": "simulated",
        },
        "env": {
            "name": env_name,
            "env_fps": 30,
            "obs_fps": 30,
            "noop_action": noop_value,
            "action_map": {
                "noop": noop_value,
                "jump": jump_value,
            },
        },
        "latency": {
            "method": "zero",
        },
        "scheduler": {
            "hold_policy": "one_frame_then_noop",
            "ordering_policy": "latest_ready",
        },
        "policy": {
            "type": "constant",
            "action": "jump",
        },
        "evaluation": {
            "eval_episodes": num_episodes,
            "eval_max_steps": eval_max_steps,
            "latency_bench_env_backend": "subproc",
        },
        "logging": {
            "output_dir": str(tmp_path),
            "save_step_records": True,
            "save_action_records": True,
            "save_latency_records": True,
        },
    }


def _subproc_executor(
    tmp_path: Path,
    *,
    num_slots: int = 2,
    env_name: str = "dummy",
    num_episodes: int = 4,
    eval_max_steps: int = 2,
    noop_value="noop",
    jump_value="jump",
    start_method: str | None = None,
    policy: PolicyRunner | None = None,
) -> BatchedSimulatedLatencyExecutor:
    from latency_bench.executors.env_step_backend import SubprocEnvStepBackend

    config = _subproc_config(
        tmp_path,
        env_name=env_name,
        num_episodes=num_episodes,
        eval_max_steps=eval_max_steps,
        noop_value=noop_value,
        jump_value=jump_value,
    )
    noop_action = Action(noop_value, name="noop", is_noop=True)
    backend = SubprocEnvStepBackend(
        config=config,
        noop_action=noop_action,
        num_slots=num_slots,
        start_method=start_method,
    )
    clock = EnvClock(env_fps=backend.env_fps, obs_fps=backend.env_fps)
    action_schedulers = [
        ActionScheduler(
            clock=clock,
            noop_action=env.noop_action,
            persistence_policy=HoldPolicy(),
            ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
        )
        for env in backend.slot_handles
    ]
    return BatchedSimulatedLatencyExecutor(
        env_backend=backend,
        policy=policy or ConstantPolicyRunner(Action(jump_value, name="jump")),
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_sources=[ZeroSampler(clock=_CLOCK) for _env in backend.slot_handles],
        action_schedulers=action_schedulers,
        clock=clock,
        logger=TrajectoryLogger(
            tmp_path,
            save_step_records=True,
            save_action_records=True,
            save_latency_records=True,
        ),
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_pids_to_exit(pids: list[int], timeout_sec: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if all(not _pid_alive(pid) for pid in pids):
            return True
        time.sleep(0.02)
    return all(not _pid_alive(pid) for pid in pids)


def _tuple_executor(
    tmp_path: Path,
    *,
    num_slots: int = 2,
    env_max_steps: int = 2,
    env_cls=TupleActionEnvAdapter,
    policy: PolicyRunner | None = None,
) -> tuple[BatchedSimulatedLatencyExecutor, list[TupleActionEnvAdapter]]:
    noop = Action((0, 0), name="noop", is_noop=True)
    envs = [
        env_cls(slot_id=slot, max_steps=env_max_steps, noop_action=noop)
        for slot in range(num_slots)
    ]
    clock = EnvClock(env_fps=30.0, obs_fps=30.0)
    action_schedulers = [
        ActionScheduler(
            clock=clock,
            noop_action=env.noop_action,
            persistence_policy=HoldPolicy(),
            ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
        )
        for env in envs
    ]
    backend = SerialEnvStepBackend(envs)
    executor = BatchedSimulatedLatencyExecutor(
        env_backend=backend,
        policy=policy or TupleActionPolicy(),
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_sources=[ZeroSampler(clock=_CLOCK) for _env in envs],
        action_schedulers=action_schedulers,
        clock=clock,
        logger=TrajectoryLogger(
            tmp_path,
            save_step_records=True,
            save_action_records=True,
            save_latency_records=True,
        ),
    )
    return executor, envs


def test_batched_executor_completes_exact_episode_ids(tmp_path: Path):
    executor, _envs = _executor(tmp_path, num_slots=4, env_max_steps=2)

    try:
        metrics = executor.run_episodes(
            episode_ids=range(5),
            seeds=[100, 101, 102, 103, 104],
            eval_max_steps=3,
        )
    finally:
        executor.close()

    assert [metric.episode_id for metric in metrics] == [0, 1, 2, 3, 4]
    assert executor.started_episodes == 5
    assert executor.completed_episodes == 5


def test_batched_executor_parallel_envs_gt_episodes_is_safe(tmp_path: Path):
    executor, _envs = _executor(tmp_path, num_slots=5, env_max_steps=1)

    try:
        metrics = executor.run_episodes(
            episode_ids=range(2),
            seeds=[7, 8],
            eval_max_steps=3,
        )
    finally:
        executor.close()

    assert [metric.episode_id for metric in metrics] == [0, 1]
    assert executor.started_episodes == 2
    assert executor.completed_episodes == 2


def test_batched_executor_zero_eval_max_steps_still_completes_all_episodes(tmp_path: Path):
    executor, _envs = _executor(tmp_path, num_slots=2, env_max_steps=5)

    try:
        metrics = executor.run_episodes(
            episode_ids=range(5),
            seeds=[1, 2, 3, 4, 5],
            eval_max_steps=0,
        )
    finally:
        executor.close()

    assert [metric.episode_id for metric in metrics] == [0, 1, 2, 3, 4]
    assert [metric.survival_steps for metric in metrics] == [0, 0, 0, 0, 0]
    assert executor.started_episodes == 5
    assert executor.completed_episodes == 5


def test_batched_executor_global_obs_and_action_ids_are_unique(tmp_path: Path):
    executor, _envs = _executor(tmp_path, num_slots=3, env_max_steps=3)

    try:
        executor.run_episodes(
            episode_ids=range(4),
            seeds=[1, 2, 3, 4],
            eval_max_steps=3,
        )
        # Records are flushed to disk (and cleared from memory) at episode end.
        action_rows = load_jsonl_records(tmp_path / "actions.jsonl")
        latency_rows = load_jsonl_records(tmp_path / "latencies.jsonl")
        action_ids = [row["action_id"] for row in action_rows]
        action_obs_ids = [row["latency"]["obs_id"] for row in action_rows]
        latency_obs_ids = [row["obs_id"] for row in latency_rows]
    finally:
        executor.close()

    assert action_ids
    assert len(action_ids) == len(set(action_ids))
    assert len(action_obs_ids) == len(set(action_obs_ids))
    assert sorted(action_obs_ids) == sorted(latency_obs_ids)


def test_batched_executor_adds_slot_id_to_policy_observation_metadata(tmp_path: Path):
    policy = MetadataCapturePolicy()
    executor, _envs = _executor(tmp_path, num_slots=2, env_max_steps=1, policy=policy)

    try:
        executor.run_episodes(
            episode_ids=range(2),
            seeds=[1, 2],
            eval_max_steps=1,
        )
    finally:
        executor.close()

    assert policy.batches[0] == [
        {
            "env_step": 0,
            "sim_time_ms": 0.0,
            "slot_index": 0,
            "slot_id": 0,
            "episode_id": 0,
            "action_noise_seed": 1,
            "obs_id": 0,
        },
        {
            "env_step": 0,
            "sim_time_ms": 0.0,
            "slot_index": 1,
            "slot_id": 1,
            "episode_id": 1,
            "action_noise_seed": 2,
            "obs_id": 1,
        },
    ]
    assert [
        (row["obs_id"], row["latency_ms"])
        for row in load_jsonl_records(tmp_path / "latencies.jsonl")
    ] == [(0, 0.0), (1, 0.0)]


def test_batched_executor_writes_simulated_pipeline_summary_when_enabled(tmp_path: Path):
    executor, _envs = _executor(tmp_path, num_slots=2, env_max_steps=1, policy=TimingBatchPolicy())
    executor.profile_pipeline = True

    try:
        executor.run_episodes(
            episode_ids=[0, 1],
            seeds=[0, 1],
            eval_max_steps=1,
        )
    finally:
        executor.close()

    summary = json.loads((tmp_path / "simulated_pipeline_summary.json").read_text(encoding="utf-8"))
    assert summary["num_profiled_batches"] == 1
    assert summary["batch_size"]["mean"] == 2.0
    assert summary["policy_build_example_ms"]["mean"] == 1.0
    assert summary["policy_wrapper_predict_action_ms"]["mean"] == 2.0
    assert summary["policy_decode_ms"]["mean"] == 3.0
    assert summary["starvla_fake_internal_ms"]["mean"] == 4.0
    assert summary["observe_slots_ms"]["count"] == 1
    assert summary["env_step_ms"]["count"] == 1


def test_batched_executor_keeps_slot_action_queues_independent(tmp_path: Path):
    clock = EnvClock(env_fps=30.0, obs_fps=30.0)
    executor, envs = _executor(
        tmp_path,
        num_slots=2,
        env_max_steps=3,
        latency_sources=[
            ZeroSampler(clock=clock),
            FixedSampler(clock.frame_ms * 2, clock=clock),
        ],
    )

    try:
        executor.run_episodes(
            episode_ids=range(2),
            seeds=[1, 2],
            eval_max_steps=3,
        )
    finally:
        executor.close()

    assert envs[0].applied_actions[0].value == "jump"
    assert [action.value for action in envs[1].applied_actions[:2]] == ["noop", "noop"]


def test_batched_executor_ignores_late_stale_result_like_realtime(tmp_path: Path):
    # obs at frame 0 takes 5 frames, obs at frame 1 takes 2. Releasing the newer
    # result at frame 3 permanently voids the older in-flight one, so when it
    # matures alone at frame 5 it must be dropped, never applied over the newer
    # action -- the same rule the realtime result coordinator enforces.
    class TwoObservationScheduler:
        def should_observe(self, env_step: int, sim_time_ms: float) -> bool:
            return int(env_step) in {0, 1}

    executor, envs = _executor(
        tmp_path,
        num_slots=1,
        env_max_steps=8,
        policy=TupleActionPolicy(),
        latency_sources=[RawFrameLatencySource([5, 2])],
        ordering_policy=OrderingPolicy.LATEST_READY,
    )
    executor.decision_scheduler = TwoObservationScheduler()

    try:
        metrics = executor.run_episodes(episode_ids=[0], seeds=[0], eval_max_steps=8)
    finally:
        executor.close()

    applied_values = [action.value for action in envs[0].applied_actions]
    assert (0, 1) in applied_values
    assert (0, 0) not in applied_values
    assert metrics[0].num_dropped_actions == 1


def test_batched_executor_soft_reset_continues_same_episode_and_clears_slot_latency_state(tmp_path: Path):
    clock = EnvClock(env_fps=30.0, obs_fps=30.0)
    noop = Action("noop", name="noop", is_noop=True)
    env = SoftResetEnvAdapter(noop_action=noop)
    policy = ResetTrackingBatchPolicy(Action("jump", name="jump"))
    backend = SerialEnvStepBackend([env])
    executor = BatchedSimulatedLatencyExecutor(
        env_backend=backend,
        policy=policy,
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_sources=[FixedSampler(clock.frame_ms * 2, clock=clock)],
        action_schedulers=[
            ActionScheduler(
                clock=clock,
                noop_action=noop,
                persistence_policy=HoldPolicy(),
                ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
            )
        ],
        clock=clock,
        logger=TrajectoryLogger(
            tmp_path,
            save_step_records=True,
            save_action_records=True,
            save_latency_records=True,
        ),
    )

    try:
        metrics = executor.run_episodes(
            episode_ids=[7],
            seeds=[5],
            eval_max_steps=10,
        )
    finally:
        executor.close()

    assert metrics[0].episode_id == 7
    assert metrics[0].survival_steps == 4
    assert metrics[0].episode_return == 4.0
    assert [action.value for action in env.applied_actions] == ["noop", "noop", "noop", "noop"]
    assert policy.reset_calls == [0, 0]


def test_batched_executor_keeps_slot_latency_sources_independent(tmp_path: Path):
    latency_sources = [SlotLatencySource(10), SlotLatencySource(20)]
    executor, _envs = _executor(
        tmp_path,
        num_slots=2,
        env_max_steps=2,
        policy=SlotIdPolicy(),
        latency_sources=latency_sources,
    )

    try:
        executor.run_episodes(
            episode_ids=range(2),
            seeds=[1, 2],
            eval_max_steps=2,
        )
        values_by_slot: dict[int, set[float]] = {}
        for row in load_jsonl_records(tmp_path / "latencies.jsonl"):
            values_by_slot.setdefault(int(row["metadata"]["slot_id"]), set()).add(row["latency_ms"])
    finally:
        executor.close()

    assert values_by_slot == {0: {10.0}, 1: {20.0}}
    assert [source.calls for source in latency_sources] == [2, 2]


def test_batched_executor_worker_capacity_is_independent_per_slot(tmp_path: Path):
    latency_sources = [
        RawFrameLatencySource([3, 3]),
        RawFrameLatencySource([3, 3]),
    ]
    executor, _ = _executor(
        tmp_path,
        num_slots=2,
        env_max_steps=5,
        policy=MetadataCapturePolicy(),
        latency_sources=latency_sources,
        simulated_worker_capacity=1,
    )

    metrics = executor.run_episodes(
        episode_ids=[10, 11],
        seeds=[None, None],
        eval_max_steps=5,
    )

    assert [[call["raw_frame"] for call in source.calls] for source in latency_sources] == [[0, 3], [0, 3]]
    assert [item.metadata["submitted_observation_frames"] for item in metrics] == [2, 2]
    assert [item.metadata["dropped_observation_count"] for item in metrics] == [3, 3]
    assert [item.metadata["simulated_worker_capacity"] for item in metrics] == [1, 1]


def test_batched_executor_output_records_remain_schema_compatible(tmp_path: Path):
    executor, _envs = _executor(tmp_path, num_slots=2, env_max_steps=1)

    try:
        executor.run_episodes(
            episode_ids=range(2),
            seeds=[1, 2],
            eval_max_steps=1,
        )
        executor.logger.flush()
    finally:
        executor.close()

    step = json.loads((tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    action = json.loads((tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    latency = json.loads((tmp_path / "latencies.jsonl").read_text(encoding="utf-8").splitlines()[0])
    metrics = json.loads((tmp_path / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert {"episode_id", "env_step", "sim_time_ms", "active_action", "reward", "done", "info"} <= set(step)
    assert {
        "action_id",
        "action",
        "ready_time_ms",
        "latency",
        "latency_raw_frames",
        "ready_raw_frame",
        "apply_raw_frame",
        "metadata",
    } <= set(action)
    assert {"obs_id", "latency_ms", "metadata"} <= set(action["latency"])
    assert action["obs_env_step"] == 0
    assert action["latency_raw_frames"] == 0
    assert action["ready_raw_frame"] == 0
    assert action["apply_raw_frame"] == 0
    assert {"obs_id", "latency_ms", "metadata"} <= set(latency)
    assert "source" not in latency
    assert {"episode_id", "episode_return", "survival_steps", "metadata"} <= set(metrics)


def test_batched_executor_no_trace_does_not_retain_hot_path_records(tmp_path: Path):
    noop = Action("noop", name="noop", is_noop=True)
    envs = [
        DummyEnvAdapter(env_fps=30.0, max_steps=3, noop_action=noop)
        for _slot in range(2)
    ]
    clock = EnvClock(env_fps=30.0, obs_fps=30.0)
    executor = BatchedSimulatedLatencyExecutor(
        env_backend=SerialEnvStepBackend(envs),
        policy=ConstantPolicyRunner(Action("jump", name="jump")),
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_sources=[ZeroSampler(clock=_CLOCK) for _env in envs],
        action_schedulers=[
            ActionScheduler(
                clock=clock,
                noop_action=env.noop_action,
                persistence_policy=HoldPolicy(),
                ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
            )
            for env in envs
        ],
        clock=clock,
        logger=TrajectoryLogger(
            tmp_path,
            save_step_records=False,
            save_action_records=False,
            save_latency_records=False,
        ),
    )

    try:
        metrics = executor.run_episodes(
            episode_ids=range(2),
            seeds=[1, 2],
            eval_max_steps=3,
        )
        executor.logger.flush()
    finally:
        executor.close()

    assert [metric.episode_return for metric in metrics] == [3.0, 3.0]
    assert [metric.survival_steps for metric in metrics] == [3, 3]
    assert [metric.num_actions for metric in metrics] == [3, 3]
    assert [metric.mean_latency_ms for metric in metrics] == [0.0, 0.0]
    assert executor.logger.step_records == []
    assert executor.logger.action_records == []
    assert executor.logger.latency_records == []
    assert not (tmp_path / "steps.jsonl").exists()
    assert not (tmp_path / "actions.jsonl").exists()
    assert not (tmp_path / "latencies.jsonl").exists()
    assert executor._completed_buffers[0].step_records is None
    assert executor._completed_buffers[0].action_events is None
    assert executor._completed_buffers[0].latency_records is None


def test_batched_executor_omits_raw_rgb_frame_stack_from_logged_step_info(tmp_path: Path):
    policy = RawRgbShapeCapturePolicy()
    executor, _envs = _tuple_executor(
        tmp_path,
        num_slots=2,
        env_max_steps=1,
        env_cls=RawRgbStepInfoSlotEnvAdapter,
        policy=policy,
    )

    try:
        executor.run_episodes(
            episode_ids=range(2),
            seeds=[1, 2],
            eval_max_steps=1,
        )
        executor.logger.flush()
    finally:
        executor.close()

    assert policy.raw_stack_shapes == [(1, 2, 3, 3), (1, 2, 3, 3)]
    assert load_jsonl_records(tmp_path / "actions.jsonl")[0]["metadata"]["input_frame_shape"] == [[2, 3, 3]]
    assert "env_raw_rgb_frame_stack" not in load_jsonl_records(tmp_path / "steps.jsonl")[0]["info"]
    step = json.loads((tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "env_raw_rgb_frame_stack" not in step["info"]


def test_batched_executor_steps_tuple_actions_and_serializes_schema_compatibly(tmp_path: Path):
    executor, envs = _tuple_executor(tmp_path, num_slots=2, env_max_steps=1)

    try:
        metrics = executor.run_episodes(
            episode_ids=range(2),
            seeds=[1, 2],
            eval_max_steps=1,
        )
        executor.logger.flush()
    finally:
        executor.close()

    assert [metric.episode_id for metric in metrics] == [0, 1]
    assert [env.applied_actions[0].value for env in envs] == [(0, 0), (1, 0)]
    action = json.loads((tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert {"action_id", "action", "ready_time_ms", "latency", "metadata"} <= set(action)
    assert isinstance(action["action"]["value"], list)


def test_subproc_backend_step_omits_observation_payload_and_observe_reads_worker_state(tmp_path: Path):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork start method is not available")

    from latency_bench.executors.env_step_backend import SubprocEnvStepBackend

    config = _subproc_config(tmp_path, num_episodes=1, eval_max_steps=3)
    backend = SubprocEnvStepBackend(
        config=config,
        noop_action=Action("noop", name="noop", is_noop=True),
        num_slots=1,
        start_method="fork",
    )

    try:
        backend.reset_slot(0, episode_id=0, seed=11)
        response = backend.step_slots({0: Action("jump", name="jump")})[0]
        observed = backend.observe_slots([0])[0]
    finally:
        backend.close()

    assert response.result.observation is None
    assert observed.data == {"state": 1}
    assert observed.env_step == 1


def test_subproc_step_action_payload_round_trips_without_dataclass():
    from latency_bench.executors.env_step_backend import (
        _action_from_step_payload,
        _step_payload_from_action,
    )

    action = Action((1, 2), name="jump", is_noop=False, is_oneshot=True)
    payload = _step_payload_from_action(action)
    restored = _action_from_step_payload(payload)

    assert payload == ((1, 2), "jump", False, True)
    assert restored == action


def test_subproc_batched_policy_receives_env_raw_rgb_metadata(tmp_path: Path):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork start method is not available")

    policy = RawRgbShapeCapturePolicy()
    executor = _subproc_executor(
        tmp_path,
        num_slots=1,
        env_name="raw_rgb_metadata_subproc_test",
        num_episodes=1,
        eval_max_steps=1,
        start_method="fork",
        policy=policy,
    )

    try:
        executor.run_episodes(
            episode_ids=[0],
            seeds=[123],
            eval_max_steps=1,
        )
        executor.logger.flush()
    finally:
        executor.close()

    action = json.loads((tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert policy.raw_stack_shapes == [(4, 5, 6, 3)]
    assert action["metadata"]["input_frame_shape"] == [[5, 6, 3]]


def test_subproc_backend_prefers_fork_start_method_by_default(monkeypatch):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork start method is not available")

    from latency_bench.executors.env_step_backend import START_METHOD_ENV, resolve_start_method

    monkeypatch.delenv(START_METHOD_ENV, raising=False)

    assert resolve_start_method() == "fork"


def test_subproc_backend_completes_exact_episode_ids_and_uses_child_pids(tmp_path: Path):
    executor = _subproc_executor(tmp_path, num_slots=3, num_episodes=5, eval_max_steps=2)
    worker_pids = executor.env_backend.worker_pids()

    try:
        metrics = executor.run_episodes(
            episode_ids=range(5),
            seeds=[10, 11, 12, 13, 14],
            eval_max_steps=2,
        )
        step_worker_pids = {
            metadata["worker_pid"]
            for metadata in executor.env_backend.last_step_metadata_by_slot.values()
        }
    finally:
        executor.close()

    assert [metric.episode_id for metric in metrics] == [0, 1, 2, 3, 4]
    assert sorted(metric.episode_id for metric in metrics) == list(range(5))
    assert len(set(metric.episode_id for metric in metrics)) == 5
    assert executor.started_episodes == 5
    assert executor.completed_episodes == 5
    assert worker_pids
    assert all(pid != os.getpid() for pid in worker_pids)
    assert step_worker_pids <= set(worker_pids)
    assert step_worker_pids


def test_subproc_backend_parallel_envs_gt_episodes_is_safe(tmp_path: Path):
    executor = _subproc_executor(tmp_path, num_slots=5, num_episodes=2, eval_max_steps=1)

    try:
        metrics = executor.run_episodes(
            episode_ids=range(2),
            seeds=[7, 8],
            eval_max_steps=3,
        )
    finally:
        executor.close()

    assert [metric.episode_id for metric in metrics] == [0, 1]
    assert executor.started_episodes == 2
    assert executor.completed_episodes == 2


def test_subproc_backend_output_records_remain_schema_compatible(tmp_path: Path):
    executor = _subproc_executor(tmp_path, num_slots=2, num_episodes=3, eval_max_steps=1)

    try:
        executor.run_episodes(
            episode_ids=range(3),
            seeds=[1, 2, 3],
            eval_max_steps=1,
        )
        executor.logger.flush()
    finally:
        executor.close()

    step = json.loads((tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    action = json.loads((tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()[0])
    latency = json.loads((tmp_path / "latencies.jsonl").read_text(encoding="utf-8").splitlines()[0])
    metrics = json.loads((tmp_path / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert {"episode_id", "env_step", "sim_time_ms", "active_action", "reward", "done", "info"} <= set(step)
    assert {"action_id", "action", "ready_time_ms", "latency", "metadata"} <= set(action)
    assert {"obs_id", "latency_ms", "metadata"} <= set(action["latency"])
    assert {"obs_id", "latency_ms", "metadata"} <= set(latency)
    assert "source" not in latency
    assert {"episode_id", "episode_return", "survival_steps", "metadata"} <= set(metrics)


def test_subproc_backend_worker_exception_does_not_hang_and_closes_workers(tmp_path: Path):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork debug start method is required for test-registered env workers")

    executor = _subproc_executor(
        tmp_path,
        num_slots=2,
        env_name="subproc_failure_test",
        num_episodes=2,
        eval_max_steps=1,
        start_method="fork",
    )
    worker_pids = executor.env_backend.worker_pids()

    with pytest.raises(RuntimeError, match="subproc step boom"):
        executor.run_episodes(
            episode_ids=range(2),
            seeds=[1, 2],
            eval_max_steps=1,
        )

    assert executor.env_backend.closed
    assert _wait_for_pids_to_exit(worker_pids)


def test_subproc_backend_workers_close_cleanly_after_normal_completion(tmp_path: Path):
    executor = _subproc_executor(tmp_path, num_slots=2, num_episodes=2, eval_max_steps=1)
    worker_pids = executor.env_backend.worker_pids()

    executor.run_episodes(
        episode_ids=range(2),
        seeds=[1, 2],
        eval_max_steps=1,
    )
    executor.close()

    assert executor.env_backend.closed
    assert _wait_for_pids_to_exit(worker_pids)


def test_subproc_backend_discrete_action_path(tmp_path: Path):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork debug start method is required for test-registered env workers")

    executor = _subproc_executor(
        tmp_path,
        num_slots=2,
        env_name="discrete_subproc_test",
        num_episodes=2,
        eval_max_steps=1,
        noop_value=0,
        jump_value=1,
        start_method="fork",
    )

    try:
        metrics = executor.run_episodes(
            episode_ids=range(2),
            seeds=[1, 2],
            eval_max_steps=1,
        )
        executor.logger.flush()
    finally:
        executor.close()

    assert [metric.episode_id for metric in metrics] == [0, 1]
    step = json.loads((tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert step["active_action"]["value"] == 1


def test_subproc_backend_multidiscrete_action_path(tmp_path: Path):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork debug start method is required for test-registered env workers")

    executor = _subproc_executor(
        tmp_path,
        num_slots=2,
        env_name="multidiscrete_subproc_test",
        num_episodes=2,
        eval_max_steps=1,
        noop_value=(0, 0),
        jump_value=(1, 2),
        start_method="fork",
    )

    try:
        metrics = executor.run_episodes(
            episode_ids=range(2),
            seeds=[1, 2],
            eval_max_steps=1,
        )
        executor.logger.flush()
    finally:
        executor.close()

    assert [metric.episode_id for metric in metrics] == [0, 1]
    step = json.loads((tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert step["active_action"]["value"] == [1, 2]

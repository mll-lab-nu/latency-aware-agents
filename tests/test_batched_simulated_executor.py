from __future__ import annotations

import json
from pathlib import Path

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, Observation, PolicyOutput, StepResult
from latency_bench.envs.base import DummyEnvAdapter, EnvAdapter
from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor
from latency_bench.latency.sources import FixedLatencySource, LatencySource, ZeroLatencySource
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.observation.builder import ObservationBuilder
from latency_bench.observation.history import HistoryBuffer
from latency_bench.policy.base import ConstantPolicyRunner, PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import EveryFrameDecisionScheduler
from latency_bench.scheduler.persistence import HoldPreviousActionPolicy


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


class SlotLatencySource(LatencySource):
    def __init__(self, slot_id: int):
        self.slot_id = int(slot_id)
        self.calls = 0

    def next_latency_ms(self, context=None) -> float:
        self.calls += 1
        return float(self.slot_id)


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


class TupleActionEnvAdapter(EnvAdapter):
    def __init__(self, *, slot_id: int, max_steps: int, noop_action: Action):
        import gymnasium as gym

        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(4), gym.spaces.Discrete(4)))
        self.gym_env = self
        self.slot_id = int(slot_id)
        self.fps_env = 30.0
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

    def _observation(self, seed: int | None = None) -> Observation:
        metadata = {"slot_index": self.slot_id}
        if seed is not None:
            metadata["seed"] = seed
        return Observation(
            data={"state": self.env_step},
            env_step=self.env_step,
            sim_time_ms=self.env_step * (1000.0 / self.fps_env),
            metadata=metadata,
        )


def _executor(
    tmp_path: Path,
    *,
    num_slots: int = 2,
    env_max_steps: int = 2,
    policy: PolicyRunner | None = None,
    latency_sources: list[LatencySource] | None = None,
) -> tuple[BatchedSimulatedLatencyExecutor, list[DummyEnvAdapter]]:
    noop = Action("noop", name="noop", is_noop=True)
    envs = [
        DummyEnvAdapter(fps_env=30.0, max_steps=env_max_steps, noop_action=noop)
        for _ in range(num_slots)
    ]
    for slot, env in enumerate(envs):
        original_make_observation = env._make_observation

        def make_observation(metadata=None, *, _slot=slot, _original=original_make_observation):
            metadata = dict(metadata or {})
            metadata["slot_index"] = _slot
            return _original(metadata=metadata)

        env._make_observation = make_observation

    clock = EnvClock(30.0)
    action_schedulers = [
        ActionScheduler(
            clock=clock,
            noop_action=env.noop_action,
            persistence_policy=HoldPreviousActionPolicy(),
        )
        for env in envs
    ]
    executor = BatchedSimulatedLatencyExecutor(
        envs=envs,
        policy=policy or ConstantPolicyRunner(Action("jump", name="jump")),
        observation_builder=ObservationBuilder(),
        histories=[HistoryBuffer() for _ in envs],
        decision_scheduler=EveryFrameDecisionScheduler(),
        latency_sources=latency_sources or [ZeroLatencySource() for _ in envs],
        action_schedulers=action_schedulers,
        clock=clock,
        logger=TrajectoryLogger(tmp_path),
    )
    return executor, envs


def _tuple_executor(
    tmp_path: Path,
    *,
    num_slots: int = 2,
    env_max_steps: int = 2,
) -> tuple[BatchedSimulatedLatencyExecutor, list[TupleActionEnvAdapter]]:
    noop = Action((0, 0), name="noop", is_noop=True)
    envs = [
        TupleActionEnvAdapter(slot_id=slot, max_steps=env_max_steps, noop_action=noop)
        for slot in range(num_slots)
    ]
    clock = EnvClock(30.0)
    action_schedulers = [
        ActionScheduler(
            clock=clock,
            noop_action=env.noop_action,
            persistence_policy=HoldPreviousActionPolicy(),
        )
        for env in envs
    ]
    executor = BatchedSimulatedLatencyExecutor(
        envs=envs,
        policy=TupleActionPolicy(),
        observation_builder=ObservationBuilder(),
        histories=[HistoryBuffer() for _ in envs],
        decision_scheduler=EveryFrameDecisionScheduler(),
        latency_sources=[ZeroLatencySource() for _env in envs],
        action_schedulers=action_schedulers,
        clock=clock,
        logger=TrajectoryLogger(tmp_path),
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
        action_ids = [event.action_id for event in executor.logger.action_records]
        action_obs_ids = [event.obs_id for event in executor.logger.action_records]
        latency_obs_ids = [record.obs_id for record in executor.logger.latency_records]
    finally:
        executor.close()

    assert len(action_ids) == len(set(action_ids))
    assert len(action_obs_ids) == len(set(action_obs_ids))
    assert sorted(action_obs_ids) == sorted(latency_obs_ids)


def test_batched_executor_keeps_slot_action_queues_independent(tmp_path: Path):
    clock = EnvClock(30.0)
    executor, envs = _executor(
        tmp_path,
        num_slots=2,
        env_max_steps=3,
        latency_sources=[
            ZeroLatencySource(),
            FixedLatencySource(clock.frame_ms * 2),
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
        for record in executor.logger.latency_records:
            values_by_slot.setdefault(int(record.metadata["slot_id"]), set()).add(record.latency_ms)
    finally:
        executor.close()

    assert values_by_slot == {0: {10.0}, 1: {20.0}}
    assert [source.calls for source in latency_sources] == [2, 2]


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
    assert {"action_id", "obs_id", "action", "latency_ms", "ready_time_ms", "metadata"} <= set(action)
    assert {"obs_id", "latency_ms", "source", "metadata"} <= set(latency)
    assert {"episode_id", "episode_return", "survival_steps", "metadata"} <= set(metrics)


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
    assert {"action_id", "obs_id", "action", "latency_ms", "ready_time_ms", "metadata"} <= set(action)
    assert isinstance(action["action"]["value"], list)

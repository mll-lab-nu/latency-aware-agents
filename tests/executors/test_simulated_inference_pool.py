"""Pooled simulated inference must be draw-for-draw identical to inline predict.

The pool only decides WHERE the forward pass physically runs; every virtual
timestamp comes from the sampled latency, so wall-clock completion order,
worker count, and result delays must leave no trace in the outputs.
"""
from __future__ import annotations

from pathlib import Path

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, LatencyRecord, Observation, OrderingPolicy, PolicyOutput
from latency_bench.envs.base import DummyEnvAdapter
from latency_bench.executors.realtime.messages import RealtimePolicyCompletion
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.latency.sample import LatencySample
from latency_bench.latency.samplers import LatencySampler
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.base import PolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler
from latency_bench.scheduler.persistence import OneFrameThenNoopPolicy
from latency_bench.utils.io import load_jsonl_records


class PureObsPolicy(PolicyRunner):
    """Action is a pure function of the observation, so prediction order is irrelevant."""

    def predict(self, observation: Observation) -> PolicyOutput:
        obs_id = observation.metadata["obs_id"]
        return PolicyOutput(
            action=Action(value=f"a{obs_id}", name=f"a{obs_id}"),
            metadata={"predicted_obs_id": obs_id},
        )


class ScriptedRawFrameSampler(LatencySampler):
    latency_type = "profile_sample"

    def __init__(self, raw_frames: list[int], *, env_fps: float = 30.0):
        self.raw_frames = list(raw_frames)
        self.frame_ms = 1000.0 / env_fps

    def sample(self, context=None) -> LatencySample:
        raw = self.raw_frames.pop(0)
        return LatencySample(
            latency_ms=raw * self.frame_ms,
            latency_raw_frames=raw,
            action_ready_raw_frames=raw,
            worker_service_raw_frames=raw,
            worker_service_ms=raw * self.frame_ms,
        )


class FakeInferencePool:
    """ProcessInferencePool stand-in: finite workers, results after N drain polls.

    Uneven per-observation delays return results out of submission order,
    exercising the reorder/blocking discipline in the executor.
    """

    def __init__(self, policy: PolicyRunner, *, worker_count: int = 2, delay_for_obs=None):
        self.policy = policy
        self._worker_count = worker_count
        self._busy: dict[int, list] = {}
        self._delay_for_obs = delay_for_obs or (lambda obs_id: 1)
        self.closed = False
        self.drain_until_idle_calls = 0

    @property
    def worker_count(self) -> int:
        return self._worker_count

    @property
    def in_flight_count(self) -> int:
        return len(self._busy)

    @property
    def idle_worker_count(self) -> int:
        return self._worker_count - len(self._busy)

    def submit(self, task) -> bool:
        for worker_id in range(self._worker_count):
            if worker_id not in self._busy:
                self._busy[worker_id] = [task, int(self._delay_for_obs(task.obs_id))]
                return True
        return False

    def drain_completed(self, now_ms: float) -> list[RealtimePolicyCompletion]:
        del now_ms
        completed = []
        for worker_id in sorted(self._busy):
            entry = self._busy[worker_id]
            entry[1] -= 1
            if entry[1] <= 0:
                task, _ = entry
                del self._busy[worker_id]
                completed.append(
                    RealtimePolicyCompletion(
                        task=task,
                        policy_output=self.policy.predict(task.observation),
                        completion_time_ms=0.0,
                        latency_record=LatencyRecord(obs_id=task.obs_id, latency_ms=0.0),
                        worker_id=worker_id,
                        device=f"fake:{worker_id}",
                    )
                )
        return completed

    def drain_until_idle(self) -> dict[str, float | int]:
        self.drain_until_idle_calls += 1
        while self._busy:
            self.drain_completed(0.0)
        return {}

    def close(self) -> None:
        self.closed = True


# Long dwells next to short ones force latest_ready voiding; repeats keep the
# script long enough for every submitted observation across the episodes.
LATENCY_SCRIPT = [2, 5, 1, 2, 3, 1, 4, 1, 2, 2] * 30


def _build_executor(tmp_path: Path, *, pool: FakeInferencePool | None) -> SimulatedLatencyExecutor:
    clock = EnvClock(env_fps=30.0, obs_fps=30.0)
    env = DummyEnvAdapter(env_fps=30.0, max_steps=40)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=env.noop_action,
        persistence_policy=OneFrameThenNoopPolicy(),
        ordering_policy=OrderingPolicy.LATEST_READY,
    )
    return SimulatedLatencyExecutor(
        env=env,
        policy=None if pool is not None else PureObsPolicy(),
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_source=ScriptedRawFrameSampler(list(LATENCY_SCRIPT)),
        action_scheduler=action_scheduler,
        clock=clock,
        logger=TrajectoryLogger(str(tmp_path), save_step_records=True, save_action_records=True, save_latency_records=True),
        simulated_worker_capacity=2,
        inference_pool=pool,
    )


def _run_episodes(executor: SimulatedLatencyExecutor, n_episodes: int = 3):
    metrics = [
        executor.run_episode(episode_id=episode_id, seed=episode_id, eval_max_steps=60)
        for episode_id in range(n_episodes)
    ]
    executor.logger.flush()
    return metrics


def _comparable_metrics(metrics) -> list[dict]:
    return [
        {
            "episode_id": item.episode_id,
            "episode_return": item.episode_return,
            "num_actions": item.num_actions,
            "num_dropped_actions": item.num_dropped_actions,
            "survival_steps": item.survival_steps,
            "latencies": item.metadata.get("submitted_observation_frames"),
            "dropped_observations": item.metadata.get("dropped_observation_count"),
        }
        for item in metrics
    ]


def test_pooled_execution_is_identical_to_inline(tmp_path):
    inline_dir = tmp_path / "inline"
    pooled_dir = tmp_path / "pooled"
    inline_metrics = _run_episodes(_build_executor(inline_dir, pool=None))

    # Delay pattern: every third observation takes 4 polls, others 1, so
    # results habitually return out of submission order.
    pool = FakeInferencePool(PureObsPolicy(), worker_count=2, delay_for_obs=lambda obs_id: 4 if obs_id % 3 == 0 else 1)
    pooled_metrics = _run_episodes(_build_executor(pooled_dir, pool=pool))

    assert _comparable_metrics(pooled_metrics) == _comparable_metrics(inline_metrics)
    for name in ("steps.jsonl", "actions.jsonl", "latencies.jsonl"):
        inline_records = load_jsonl_records(inline_dir / name)
        pooled_records = load_jsonl_records(pooled_dir / name)
        assert pooled_records == inline_records, f"{name} diverged between pooled and inline execution"


def test_pooled_execution_with_single_worker_blocks_and_matches(tmp_path):
    # One fake worker (fewer than the virtual capacity of 2) forces the
    # parent-side submit queue and the blocking wait to engage.
    inline_metrics = _run_episodes(_build_executor(tmp_path / "inline", pool=None))
    pool = FakeInferencePool(PureObsPolicy(), worker_count=1, delay_for_obs=lambda obs_id: 2)
    pooled_metrics = _run_episodes(_build_executor(tmp_path / "pooled", pool=pool))

    assert _comparable_metrics(pooled_metrics) == _comparable_metrics(inline_metrics)


def test_pool_lifecycle_is_driven_by_the_executor(tmp_path):
    pool = FakeInferencePool(PureObsPolicy(), worker_count=2)
    executor = _build_executor(tmp_path, pool=pool)
    _run_episodes(executor, n_episodes=2)
    assert pool.drain_until_idle_calls == 2  # one pre-episode drain per episode
    executor.close()
    assert pool.closed
    assert executor.env.closed

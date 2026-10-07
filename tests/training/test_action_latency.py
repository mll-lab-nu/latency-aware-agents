"""Long-lived action latency contracts shared with scalar and batched evaluation."""

from types import SimpleNamespace

import pytest
import torch

from latency_bench.latency.sample import LatencySample
from training.common.action_latency import ActionLatencyBatch


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("hold_tail", [False, True])
def test_replica_admission_stale_prefix_and_episode_reset(hold_tail, device):
    config = {
        "env": {"env_fps": 50, "obs_fps": 50},
        "executor": {"simulated_worker_capacity": 1},
        "latency": {"method": "fixed", "fixed_latency_ms": 40, "seed": 0},
        "scheduler": {"ordering_policy": "latest_ready", "hold_policy": "hold", "hold_last_chunk_action": hold_tail},
    }
    queue = ActionLatencyBatch(config, num_envs=2, device=device, noop_command=[0])
    queue.reset()
    assert queue.episodes == [0, 1]
    chunks = torch.tensor([[[1.], [2.], [3.], [4.]], [[10.], [20.], [30.], [40.]]], device=device)
    assert queue.submit(chunks).tolist() == [True, True]
    first = queue.actions()
    torch.testing.assert_close(first.cpu(), torch.zeros(2, 1))
    chunks.fill_(-100)
    queue.advance()
    assert queue.submit(chunks).tolist() == [False, False]
    queue.reset([0])
    fresh_chunks = torch.tensor([[[1.], [2.], [3.], [4.]], [[10.], [20.], [30.], [40.]]], device=device)
    assert queue.submit(fresh_chunks, [0]).tolist() == [True, False]
    queue.advance()
    torch.testing.assert_close(queue.actions().cpu(), torch.tensor([[0.], [30.]]))
    queue.advance()
    torch.testing.assert_close(queue.actions().cpu(), torch.tensor([[3.], [40.]]))
    assert queue.last_application == [
        {"source_raw_frame": 0, "source_obs_id": 0, "chunk_index": 2},
        {"source_raw_frame": 0, "source_obs_id": 0, "chunk_index": 3},
    ]
    queue.advance()
    torch.testing.assert_close(queue.actions().cpu(), torch.tensor([[4.], [40. if hold_tail else 0.]]))
    assert queue.last_application[1] == ({
        "source_raw_frame": 0, "source_obs_id": 0, "chunk_index": 3,
    } if hold_tail else {
        "source_raw_frame": -1, "source_obs_id": -1, "chunk_index": -1,
    })
    queue.reset()
    torch.testing.assert_close(queue.actions().cpu(), torch.zeros(2, 1))
    torch.testing.assert_close(first.cpu(), torch.zeros(2, 1))


@pytest.mark.parametrize(("fps", "latency_frames"), [(20, 2), (50, 5)])
def test_single_action_holds_until_replacement_and_reset_at_control_rate(fps, latency_frames):
    config = {
        "env": {"env_fps": fps, "obs_fps": fps},
        "executor": {"simulated_worker_capacity": 1},
        "latency": {"method": "fixed", "fixed_latency_ms": 100, "seed": 0},
        "scheduler": {"ordering_policy": "latest_ready", "hold_policy": "hold"},
    }
    queue = ActionLatencyBatch(config, num_envs=2, device="cpu", noop_command=[0, 0])
    commands = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
    queue.reset()

    assert queue.submit(commands).tolist() == [True, True]
    torch.testing.assert_close(queue.actions(), torch.zeros(2, 2))
    for _ in range(latency_frames):
        queue.advance()
        actions = queue.actions()
    torch.testing.assert_close(actions, commands)
    assert queue.last_application == [
        {"source_raw_frame": 0, "source_obs_id": 0, "chunk_index": 0}
        for _ in range(2)
    ]

    queue.reset([0])
    torch.testing.assert_close(queue.actions(), torch.tensor([[0.0, 0.0], [10.0, 20.0]]))
    replacement = torch.tensor([[3.0, 4.0], [30.0, 40.0]])
    assert queue.submit(replacement, [0]).tolist() == [True, False]
    torch.testing.assert_close(queue.actions(), torch.tensor([[0.0, 0.0], [10.0, 20.0]]))
    for _ in range(latency_frames):
        queue.advance()
        actions = queue.actions()
    torch.testing.assert_close(actions, torch.tensor([[3.0, 4.0], [10.0, 20.0]]))

    queue.reset()
    torch.testing.assert_close(queue.actions(), torch.zeros(2, 2))
    assert queue.last_application == [
        {"source_raw_frame": -1, "source_obs_id": -1, "chunk_index": -1}
        for _ in range(2)
    ]


def test_submission_metrics_capture_latency_and_capacity_drop():
    config = {
        "env": {"env_fps": 50, "obs_fps": 50},
        "executor": {"simulated_worker_capacity": 1},
        "latency": {"method": "fixed", "fixed_latency_ms": 100, "seed": 0},
        "scheduler": {"ordering_policy": "latest_ready", "hold_policy": "hold"},
    }
    queue = ActionLatencyBatch(config, num_envs=1, device="cpu", noop_command=[0])
    queue.reset()

    assert queue.submit(torch.tensor([[1.0]])).tolist() == [True]
    assert queue.last_submission[0]["latency_ms"] == 100.0
    assert queue.last_dropped == [False]
    queue.advance()

    assert queue.submit(torch.tensor([[2.0]])).tolist() == [False]
    assert queue.last_submission == [None]
    assert queue.last_dropped == [True]

    for _ in range(4):
        queue.advance()
    torch.testing.assert_close(queue.actions(), torch.tensor([[1.0]]))
    assert queue.submit(torch.tensor([[2.0]])).tolist() == [True]
    for _ in range(4):
        queue.advance()
        assert queue.submit(torch.tensor([[3.0]])).tolist() == [False]
        torch.testing.assert_close(queue.actions(), torch.tensor([[1.0]]))
    queue.advance()
    torch.testing.assert_close(queue.actions(), torch.tensor([[2.0]]))


@pytest.mark.parametrize("ordering", ["latest_ready", "issue_order_fifo"])
def test_worker_release_precedes_readiness_and_late_results_keep_issue_order(ordering):
    """One worker can leave several pending results; newer results may arrive first."""
    config = {
        "env": {"env_fps": 50, "obs_fps": 50},
        "executor": {"simulated_worker_capacity": 1},
        "latency": {"method": "zero", "seed": 0},
        "scheduler": {"ordering_policy": ordering, "hold_policy": "hold"},
    }
    queue = ActionLatencyBatch(config, num_envs=1, device="cpu", noop_command=[0])
    queue.reset()
    samples = iter([
        LatencySample.from_action_ready(
            action_ready_raw_frames=ready, worker_service_raw_frames=1,
            worker_service_ms=20, clock=queue.clock,
        ) for ready in (5, 1, 0)
    ])
    queue.samplers[0] = SimpleNamespace(sample=lambda context: next(samples))
    trace = []
    for frame in range(8):
        if frame < 3:
            assert queue.submit(torch.tensor([[10.0 * (frame + 1)]])).tolist() == [True]
        trace.append(queue.actions().item())
        queue.advance()
    assert trace == (
        [0, 0, 30, 30, 30, 30, 30, 30]
        if ordering == "latest_ready" else [0, 0, 0, 0, 0, 10, 20, 30]
    )


def test_load_state_dict_preserves_sampler_continuation_and_clears_transport():
    config = {
        "env": {"env_fps": 50, "obs_fps": 50},
        "executor": {"simulated_worker_capacity": 1},
        "latency": {
            "method": "uniform",
            "min_latency_ms": 0,
            "max_latency_ms": 100,
            "seed": 7,
        },
        "scheduler": {"ordering_policy": "latest_ready", "hold_policy": "hold"},
    }
    queue = ActionLatencyBatch(config, num_envs=1, device="cpu", noop_command=[0])
    queue.reset()
    commands = torch.tensor([[[1.], [2.]]])
    context = {
        "episode_id": 0,
        "worker_slot": 0,
        "obs_id": 0,
        "raw_frame": 0,
        "env_step": 0,
        "sim_time_ms": 0.0,
        "in_flight_count": 0,
        "idle_worker_count": 1,
        "recent_drop_count": 0,
    }
    queue.submit(commands)
    queue.advance()
    queue.samplers[0].sample(context)
    state = queue.state_dict()
    expected = queue.samplers[0].sample(context).latency_ms

    resumed = ActionLatencyBatch(config, num_envs=1, device="cpu", noop_command=[0])
    resumed.load_state_dict(state)

    assert resumed.frames == [0]
    assert resumed.submitted == [0]
    assert resumed.episodes == state["episodes"]
    assert resumed.samplers[0].sample(context).latency_ms == expected
    torch.testing.assert_close(resumed.actions(), torch.zeros(1, 1))


def test_training_episode_stream_matches_evaluator_after_independent_resets():
    from latency_bench.latency.samplers import build_episode_sampler_factory

    config = {
        "env": {"env_fps": 50, "obs_fps": 50},
        "executor": {"simulated_worker_capacity": 1},
        "latency": {"method": "uniform", "min_latency_ms": 0, "max_latency_ms": 100, "seed": 7},
        "scheduler": {"ordering_policy": "latest_ready", "hold_policy": "hold"},
    }
    queue = ActionLatencyBatch(config, num_envs=2, device="cpu", noop_command=[0])
    evaluator_factory = build_episode_sampler_factory(config["latency"], queue.clock)
    queue.reset()
    queue.reset([1])
    queue.reset([0])
    assert queue.episodes == [3, 2]
    for slot, episode_id in enumerate(queue.episodes):
        evaluator = evaluator_factory(episode_id)
        assert [queue.samplers[slot].sample({}).latency_ms for _ in range(8)] == [
            evaluator.sample({}).latency_ms for _ in range(8)
        ]


def test_training_scalar_and_batched_eval_apply_identical_command_trace():
    """Transport contract: admission, stale H40 prefixes and resets match evaluation."""
    from latency_bench.core.types import Action, PolicyOutput
    from latency_bench.envs.adapters.dummy import DummyEnvAdapter
    from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor
    from latency_bench.executors.env_step_backend import SerialEnvStepBackend
    from latency_bench.executors.simulated import SimulatedLatencyExecutor
    from latency_bench.latency.samplers import build_episode_sampler_factory, build_sampler
    from latency_bench.policy.base import PolicyRunner
    from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler
    from latency_bench.scheduler.factory import build_action_scheduler

    class CommandPolicy(PolicyRunner):
        def predict(self, observation):
            chunk = torch.arange(40, dtype=torch.float32).reshape(40, 1) + observation.env_step * 100
            return PolicyOutput(action=Action(chunk[0], name="command"), action_chunk=chunk)

        def predict_batch(self, observations):
            return [self.predict(observation) for observation in observations]

    config = {
        "env": {"env_fps": 50, "obs_fps": 10},
        "executor": {"simulated_worker_capacity": 2},
        "latency": {"method": "uniform", "min_latency_ms": 0, "max_latency_ms": 400, "seed": 7},
        "scheduler": {"ordering_policy": "latest_ready", "hold_policy": "hold"},
    }
    queue = ActionLatencyBatch(config, num_envs=2, device="cpu", noop_command=[-1])
    env = DummyEnvAdapter(env_fps=50, max_steps=100, noop_action=Action(torch.tensor([-1.])))
    evaluator = SimulatedLatencyExecutor(
        env=env, policy=CommandPolicy(), clock=queue.clock,
        decision_scheduler=ObservationCadenceDecisionScheduler(queue.clock),
        latency_source=build_sampler(config["latency"], clock=queue.clock),
        action_scheduler=build_action_scheduler(config["scheduler"], clock=queue.clock, noop_action=env.noop_action),
        episode_latency_source_factory=build_episode_sampler_factory(config["latency"], queue.clock),
        simulated_worker_capacity=2,
    )
    batch_envs = [DummyEnvAdapter(env_fps=50, max_steps=100, noop_action=env.noop_action) for _ in range(2)]
    batched = BatchedSimulatedLatencyExecutor(
        env_backend=SerialEnvStepBackend(batch_envs), policy=CommandPolicy(), clock=queue.clock,
        decision_scheduler=ObservationCadenceDecisionScheduler(queue.clock),
        latency_sources=[build_sampler(config["latency"], clock=queue.clock) for _ in range(2)],
        action_schedulers=[build_action_scheduler(config["scheduler"], clock=queue.clock, noop_action=env.noop_action) for _ in range(2)],
        episode_latency_source_factory=build_episode_sampler_factory(config["latency"], queue.clock),
        simulated_worker_capacity=2,
    )
    for batch in range(2):
        queue.reset()
        training_actions = []
        for frame in range(100):
            queue.submit((torch.arange(40, dtype=torch.float32).reshape(1, 40, 1) + frame * 100).expand(2, 40, 1))
            training_actions.append(queue.actions())
            queue.advance()
        batched.run_episodes(episode_ids=[2 * batch, 2 * batch + 1], seeds=[0, 1], eval_max_steps=100)
        for slot in range(2):
            evaluator.run_episode(2 * batch + slot, eval_max_steps=100)
            expected = torch.stack(training_actions)[:, slot]
            torch.testing.assert_close(expected, torch.stack([a.value for a in env.applied_actions]))
            torch.testing.assert_close(expected, torch.stack([a.value for a in batch_envs[slot].applied_actions]))

"""Long-lived native transport contracts shared with scalar and batched evaluation."""

import pytest
import torch

from training.common.command_latency import CommandLatencyBatch


@pytest.mark.parametrize("hold_tail", [False, True])
def test_replica_admission_stale_prefix_and_episode_reset(hold_tail):
    config = {
        "env": {"env_fps": 50, "obs_fps": 50},
        "executor": {"simulated_worker_capacity": 1},
        "latency": {"method": "fixed", "fixed_latency_ms": 40, "seed": 0},
        "scheduler": {"ordering_policy": "latest_ready", "hold_policy": "hold", "hold_last_chunk_action": hold_tail},
    }
    queue = CommandLatencyBatch(config, num_envs=2, device="cpu", noop_command=[0])
    queue.reset()
    assert queue.episodes == [0, 1]
    chunks = torch.tensor([[[1.], [2.], [3.], [4.]], [[10.], [20.], [30.], [40.]]])
    assert queue.submit(chunks).tolist() == [True, True]
    torch.testing.assert_close(queue.actions(), torch.zeros(2, 1))
    queue.advance()
    assert queue.submit(chunks).tolist() == [False, False]
    queue.reset([0])
    assert queue.submit(chunks, [0]).tolist() == [True, False]
    queue.advance()
    torch.testing.assert_close(queue.actions(), torch.tensor([[0.], [30.]]))
    queue.advance()
    torch.testing.assert_close(queue.actions(), torch.tensor([[3.], [40.]]))
    assert queue.last_application == [
        {"source_raw_frame": 0, "source_obs_id": 0, "chunk_index": 2},
        {"source_raw_frame": 0, "source_obs_id": 0, "chunk_index": 3},
    ]
    queue.advance()
    torch.testing.assert_close(queue.actions(), torch.tensor([[4.], [40. if hold_tail else 0.]]))
    assert queue.last_application[1] == ({
        "source_raw_frame": 0, "source_obs_id": 0, "chunk_index": 3,
    } if hold_tail else {
        "source_raw_frame": -1, "source_obs_id": -1, "chunk_index": -1,
    })
    queue.reset()
    torch.testing.assert_close(queue.actions(), torch.zeros(2, 1))


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
    queue = CommandLatencyBatch(config, num_envs=1, device="cpu", noop_command=[0])
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

    resumed = CommandLatencyBatch(config, num_envs=1, device="cpu", noop_command=[0])
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
    queue = CommandLatencyBatch(config, num_envs=2, device="cpu", noop_command=[0])
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
    from latency_bench.envs.base import DummyEnvAdapter
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
    queue = CommandLatencyBatch(config, num_envs=2, device="cpu", noop_command=[-1])
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

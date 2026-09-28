from __future__ import annotations

from types import SimpleNamespace

import torch


def test_gpu_sampling_patch_only_enables_env_with_cpu_tensor_channel():
    from training.common.sf_gpu_sampling_compat import _is_enabled_for_sample_factory_env

    enabled_env_names = {"latency_flappy"}
    cpu_runner = SimpleNamespace(
        cfg=SimpleNamespace(env="latency_flappy"),
        vec_env=SimpleNamespace(),
    )
    gpu_runner = SimpleNamespace(
        cfg=SimpleNamespace(env="latency_flappy"),
        vec_env=SimpleNamespace(sample_factory_step_cpu_tensors={}),
    )

    assert _is_enabled_for_sample_factory_env(cpu_runner, enabled_env_names) is False
    assert _is_enabled_for_sample_factory_env(gpu_runner, enabled_env_names) is True


def test_gpu_sampling_patch_supports_multiple_gpu_env_names():
    from training.common.sf_gpu_sampling_compat import _is_enabled_for_sample_factory_env

    enabled_env_names = {"latency_flappy", "latency_demon_attack"}
    demon_runner = SimpleNamespace(
        cfg=SimpleNamespace(env="latency_demon_attack"),
        vec_env=SimpleNamespace(sample_factory_step_cpu_tensors={}),
    )
    unknown_runner = SimpleNamespace(
        cfg=SimpleNamespace(env="latency_unknown"),
        vec_env=SimpleNamespace(sample_factory_step_cpu_tensors={}),
    )

    assert _is_enabled_for_sample_factory_env(demon_runner, enabled_env_names) is True
    assert _is_enabled_for_sample_factory_env(unknown_runner, enabled_env_names) is False


def test_gpu_sampling_patch_prefers_fast_step_through_wrapper_chain():
    from training.common.sf_gpu_sampling_compat import _sample_factory_gpu_step

    calls = []

    class FastEnv:
        sample_factory_step_cpu_tensors = {}

        def sample_factory_gpu_step(self, actions):
            calls.append(actions)
            return "obs", "rewards", "terminated", "truncated", "infos"

    wrapped = SimpleNamespace(env=FastEnv())

    step = _sample_factory_gpu_step(wrapped)

    assert step("actions") == ("obs", "rewards", "terminated", "truncated", "infos")
    assert calls == ["actions"]


def test_gpu_sampling_patched_rollout_writes_actor_admission(monkeypatch):
    from sample_factory.algo.sampling import batched_sampling
    from sample_factory.algo.utils.tensor_dict import TensorDict
    from sample_factory.utils.timing import Timing
    from training.common.sf_gpu_sampling_compat import _patch_batched_advance_rollouts

    monkeypatch.setattr(batched_sampling, "preprocess_actions", lambda env_info, actions: actions)

    class Runner:
        def advance_rollouts(self, policy_id, timing):
            raise AssertionError("original rollout path used")

    _patch_batched_advance_rollouts(Runner, "latency_flappy")

    class Env:
        num_agents = 1

        def __init__(self):
            self.sample_factory_step_cpu_tensors = {}

        def step(self, actions):
            self.sample_factory_step_cpu_tensors = {
                "rewards": torch.tensor([1.0]),
                "dones": torch.tensor([False]),
                "command_admitted": torch.tensor([False]),
            }
            return {"obs": torch.zeros(1, 1)}, torch.tensor([1.0]), torch.tensor([False]), torch.tensor([False]), [{}]

    runner = Runner()
    runner.cfg = SimpleNamespace(env="latency_flappy", mask_unadmitted_actor=True, rollout=2)
    runner.vec_env = Env()
    runner.env_info = SimpleNamespace()
    runner.curr_step = TensorDict(
        actions=torch.zeros(1, 1),
        rewards=torch.zeros(1),
        dones=torch.zeros(1, dtype=torch.bool),
        time_outs=torch.zeros(1, dtype=torch.bool),
        policy_id=torch.zeros(1, dtype=torch.int32),
        command_admitted=torch.ones(1, dtype=torch.bool),
    )
    runner.policy_output_tensors = TensorDict(actions=torch.zeros(1, 1), new_rnn_states=torch.zeros(1, 1))
    runner.policy_id_buffer = torch.zeros(1, dtype=torch.int32)
    runner.policy_id = 0
    runner.last_rnn_state = torch.ones(1, 1)
    runner.rollout_step = 0
    runner._process_rewards = lambda rewards, cpu_rewards: rewards
    runner._process_env_step = lambda rewards, dones, infos: []

    runner.advance_rollouts(0, Timing())

    assert runner.curr_step["command_admitted"].tolist() == [False]
    assert runner.curr_step["rewards"].tolist() == [1.0]
    assert runner.curr_step["dones"].tolist() == [False]

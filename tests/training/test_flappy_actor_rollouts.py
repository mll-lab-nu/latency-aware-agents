from types import SimpleNamespace

import torch

from sample_factory.algo.sampling import batched_sampling
from sample_factory.algo.sampling.non_batched_sampling import ActorState
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.timing import Timing


def test_non_batched_actor_records_command_admission():
    actor = ActorState.__new__(ActorState)
    actor.cfg = SimpleNamespace(algo="APPO", summaries_use_frameskip=False, mask_unadmitted_actor=True)
    actor.env_info = SimpleNamespace(frameskip=1)
    actor.curr_traj_buffer = TensorDict(
        rewards=torch.zeros(1),
        dones=torch.zeros(1, dtype=torch.bool),
        time_outs=torch.zeros(1, dtype=torch.bool),
        policy_id=torch.zeros(1, dtype=torch.int32),
        command_admitted=torch.ones(1, dtype=torch.bool),
    )
    actor.is_active = True
    actor.curr_policy_id = 0
    actor.last_episode_duration = 0

    actor.record_env_step(1.0, False, False, {"command_admitted": False}, 0)

    assert actor.curr_traj_buffer["command_admitted"].tolist() == [False]
    assert actor.curr_traj_buffer["policy_id"].tolist() == [0]
    assert actor.curr_traj_buffer["rewards"].tolist() == [1.0]


def test_standard_batched_runner_records_command_admission(monkeypatch):
    monkeypatch.setattr(batched_sampling, "preprocess_actions", lambda env_info, actions: actions)

    class Env:
        num_agents = 1

        def step(self, actions):
            return {"obs": torch.zeros(1, 1)}, torch.tensor([1.0]), torch.tensor([False]), torch.tensor([False]), [
                {"command_admitted": False}
            ]

    runner = SimpleNamespace()
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
    runner.device = torch.device("cpu")
    runner.last_rnn_state = torch.ones(1, 1)
    runner.rollout_step = 0
    runner._process_rewards = lambda rewards, cpu_rewards: rewards
    runner._process_env_step = lambda rewards, dones, infos: []

    batched_sampling.BatchedVectorEnvRunner.advance_rollouts(runner, 0, Timing())

    assert runner.curr_step["command_admitted"].tolist() == [False]
    assert runner.curr_step["rewards"].tolist() == [1.0]

"""Long-lived native PPO contracts: full episodes, MC targets, and actor/V isolation."""
import copy
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from latency_bench.core.config import load_config
from sample_factory.algo.mc_ppo.env import EpisodeEnv
from sample_factory.algo.mc_ppo.evaluation import comparison, episode_summary
from sample_factory.algo.mc_ppo.learner import MCPPOLearner, prepare_batch, update
from sample_factory.algo.mc_ppo.models import MCPPOActorCritic
from sample_factory.algo.mc_ppo.sampling import EpisodeSchedule
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.context import global_env_registry
from sample_factory.algo.utils.shared_buffers import alloc_trajectory_tensors
from sample_factory.train import run_rl
from sample_factory.utils.dicts import iterate_recursively
from training.gymnasium_sf.integration import build_train_cfg_from_config, make_gymnasium_sf_env


class CountingEpisode(gym.Env):
    """Small deterministic task with unequal seeded episode lengths."""

    observation_space = gym.spaces.Dict({
        "obs": gym.spaces.Box(-10, 10, (2,), dtype=np.float64),
        "scheduled_chunk": gym.spaces.Box(-1, 1, (16, 1), dtype=np.float32),
        "scheduled_index": gym.spaces.Box(0, 1, (16,), dtype=np.float32),
    })
    action_space = gym.spaces.Box(-1, 1, (16,), dtype=np.float32)

    def __init__(self):
        self.actions = []

    def observation(self):
        return {"obs": np.array([self.frames, self.length], dtype=np.float64),
                "scheduled_chunk": np.zeros((16, 1), dtype=np.float32),
                "scheduled_index": np.zeros(16, dtype=np.float32)}

    def reset(self, *, seed=None, options=None):
        self.frames, self.length = 0, 6 + seed % 3
        return self.observation(), {}

    def step(self, action):
        self.actions.append(action.copy())
        self.frames += 1
        return self.observation(), 0.5, self.frames == self.length, False, {}


def make_counting_episode(full_env_name, cfg, env_config, render_mode=None):
    return EpisodeEnv(CountingEpisode(), cfg.ppo)


def test_parallel_sf_collects_complete_episodes_and_resumes(tmp_path, monkeypatch):
    """Spawned workers must write shared trajectories, including unequal terminal lengths."""
    config = load_config("configs/examples/gymnasium/g1_sonic_balance_simple/ppo_train.yaml")
    config["training"]["ppo"].update(workers=2, episodes_per_round=5, rounds=1, episode_frames=10)
    config["backend"].update(device="cpu", train_dir=str(tmp_path))
    config["experiment"]["name"] = "complete_episodes"
    cfg = build_train_cfg_from_config(config)
    cfg.env, cfg.benchmark, cfg.with_wandb = "test_mc_episodes", True, False
    monkeypatch.setitem(global_env_registry(), cfg.env, make_counting_episode)
    env = make_counting_episode(cfg.env, cfg, None)
    model = MCPPOActorCritic(env.observation_space, env.action_space, cfg)
    cfg.initial_model_path = str(tmp_path / "bc.pth")
    torch.save({"model": model.policy.state_dict()}, cfg.initial_model_path)
    env.close()
    assert run_rl(cfg) == 0
    cfg.restart_behavior = "resume"
    cfg.ppo["rounds"] = 2
    cfg.cli_args = {"restart_behavior": "resume", "ppo": cfg.ppo}
    assert run_rl(cfg) == 0
    directory = tmp_path / cfg.experiment
    rounds = [json.loads(path.read_text()) for path in sorted(directory.glob("round_*.json"))]
    assert [row["round"] for row in rounds] == [0, 1, 2]
    for row in rounds:
        assert row["sampling"]["n"] == 5
        assert row["sampling"]["frames"] == sum(6 + episode["seed"] % 3 for episode in row["sampling"]["episodes"])
    assert rounds[-1]["counters"]["episodes"] == 15
    # Resuming an exhausted budget must terminate without collecting another round.
    modified = {path: path.stat().st_mtime_ns for path in directory.glob("round_*.json")}
    assert run_rl(cfg) == 0
    assert {path: path.stat().st_mtime_ns for path in directory.glob("round_*.json")} == modified


def test_complete_episode_mc_and_separate_value_updates():
    torch.manual_seed(1)
    torch.set_num_threads(1)
    settings = {"chunk_horizon": 16, "active_heads": [5, 10], "action_std": .02,
                "decision_frames": 5, "episode_frames": 10, "minibatch_size": 4,
                "target_kl": .02, "clip_ratio": .2, "inference_batch_size": 64,
                "normalize_advantages": True, "kl_control": "rollback"}
    cfg = build_train_cfg_from_config(load_config("configs/examples/gymnasium/g1_sonic_balance_simple/ppo_train.yaml"))
    cfg.ppo = settings
    env = EpisodeEnv(CountingEpisode(), settings)
    model = MCPPOActorCritic(env.observation_space, env.action_space, cfg)
    policy, value = model.policy, model.value
    for parameter in value.parameters():
        nn.init.zeros_(parameter)
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=1e-6)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=3e-4)
    original = {key: tensor.clone() for key, tensor in policy.state_dict().items()}
    # Capacity exceeds actual lengths: padding must never enter the MC objective.
    buffer = alloc_trajectory_tensors(SimpleNamespace(obs_space=env.observation_space, action_space=env.action_space),
                                     5, 3, 1, "cpu", False, "PPO")
    for _, _, tensor in iterate_recursively(buffer):
        tensor.zero_()
    try:
        for row, seed in enumerate(range(1, 6)):
            observation, _ = env.reset(seed=seed)
            step = 0
            while True:
                buffer["obs"][row, step] = observation
                with torch.no_grad():
                    output = model({key: torch.as_tensor(item)[None] for key, item in observation.items()}, torch.zeros(1, 1))
                buffer[row, step] = {key: tensor[0] for key, tensor in output.items() if key != "new_rnn_states"}
                observation, reward, terminated, truncated, info = env.step(output["actions"][0].numpy())
                buffer["raw_frames"][row, step] = info["frames"]
                buffer["raw_rewards"][row, step] = info["raw_reward"]
                buffer["rewards"][row, step] = reward
                buffer["valids"][row, step] = True
                buffer["dones"][row, step] = terminated or truncated
                step += 1
                if terminated or truncated:
                    break
        data, episodes = prepare_batch(model, buffer, settings["episode_frames"])
        assert [row["seed"] for row in episodes] == list(range(1, 6))
        assert [row["length"] for row in episodes] == [7, 8, 6, 7, 8]
        assert sum(row["length"] for row in episodes) == 36
        assert episode_summary(episodes)["mean_length"] == 7.2
        assert comparison(episodes, list(reversed(episodes))) == {"mean_length_gain": 0, "paired_95_interval": [0, 0], "accepted": False}
        assert len(data["obs"]) == 10
        torch.testing.assert_close(data["returns"], (data["obs"][:, 1] - data["obs"][:, 0]) * 1.01 / 10)
        torch.testing.assert_close(data["advantages"], data["returns"])
        torch.testing.assert_close(policy.distribution(data["means"]).log_prob(data["latent"]), data["log_prob"])
        result = update(policy, value, actor_optimizer, value_optimizer, data, settings, epochs=2, actor=False)
        assert result["actor_updates"] == 0
        assert result["value_updates"] == 6
        for key, tensor in policy.state_dict().items():
            torch.testing.assert_close(tensor, original[key], rtol=0, atol=0)
        result = update(policy, value, actor_optimizer, value_optimizer, data, settings, epochs=2, actor=True)
        assert 1 <= result["actor_updates"] <= 6
        assert result["value_updates"] == 6
        assert torch.isfinite(torch.tensor(result["actor_loss"]))
        assert result["kl"] <= settings["target_kl"]
        assert policy.empirical_obs_normalizer.count == 0
        # A rejected step restores existing Adam moments and leaves V learning active.
        with torch.no_grad():
            data["means"] = policy.logits(data["obs"])[:, policy.active].clone()
            data["log_prob"] = policy.distribution(data["means"]).log_prob(data["latent"])
        actor_optimizer.param_groups[0]["lr"] = .1
        parameters = copy.deepcopy(policy.state_dict())
        optimizer_state = copy.deepcopy(actor_optimizer.state_dict())
        result = update(policy, value, actor_optimizer, value_optimizer, data, settings, epochs=2, actor=True)
        assert result["actor_updates"] == 0
        assert result["rejected_updates"] == 1
        assert result["value_updates"] == 6
        assert result["kl"] <= settings["target_kl"]
        for key, tensor in policy.state_dict().items():
            torch.testing.assert_close(tensor, parameters[key], rtol=0, atol=0)
        for key, state in actor_optimizer.state_dict()["state"].items():
            for name, tensor in state.items():
                torch.testing.assert_close(tensor, optimizer_state["state"][key][name], rtol=0, atol=0)
        json.dumps(result, allow_nan=False)
    finally:
        env.close()


def test_checkpoint_roundtrip_preserves_optimizer_and_rng(tmp_path):
    torch.manual_seed(9)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    cfg = SimpleNamespace(serial_mode=True)
    learner = MCPPOLearner(cfg, None, torch.zeros(1, dtype=torch.int64), 0, ParameterServer(0, torch.zeros(1, dtype=torch.int64), True))
    learner.actor_critic = model
    learner.actor_optimizer = optimizer
    learner.value_optimizer = optimizer
    learner.device = torch.device("cpu")
    learner.round, learner.train_step, learner.env_steps = 7, 23, 1234
    learner.value_updates, learner.episodes, learner.best_round = 45, 128, 5
    state = learner._get_checkpoint_dict()
    path = tmp_path / "checkpoint.pth"
    torch.save(state, path)
    expected = torch.rand(4)
    loaded = torch.load(path, weights_only=False)
    learner._load_state(loaded)
    torch.testing.assert_close(torch.rand(4), expected)
    restored = nn.Linear(2, 1)
    restored.load_state_dict(loaded["model"])
    restored_optimizer = torch.optim.Adam(restored.parameters())
    restored_optimizer.load_state_dict(loaded["actor_optimizer"])
    assert loaded["round"] == 7
    torch.testing.assert_close(restored(torch.ones(1, 2)), model(torch.ones(1, 2)))
    assert restored_optimizer.state[restored.weight]["step"] == 1
    assert (learner.round, learner.train_step, learner.env_steps, learner.value_updates) == (7, 23, 1234, 45)


def test_round_seed_schedule_resumes_without_reusing_episodes():
    cfg = SimpleNamespace(ppo={"episodes_per_round": 5, "train_seed_base": 101}, benchmark=False, ppo_start_round=0)
    schedule = EpisodeSchedule(cfg)
    assert [schedule.claim(0) for _ in range(5)] == list(range(96, 101))
    assert schedule.claim(0) is None
    assert [schedule.claim(1) for _ in range(5)] == list(range(101, 106))
    cfg.ppo_start_round = 2
    resumed = EpisodeSchedule(cfg)
    assert [resumed.claim(2) for _ in range(5)] == list(range(106, 111))


def test_native_ppo_observation_exposes_only_committed_heads():
    """Long-lived PPO/latency contract: returned heads 5:10 are visible before admission."""
    config = load_config("configs/examples/gymnasium/g1_sonic_balance_simple/ppo_train.yaml")
    cfg = build_train_cfg_from_config(config)
    cfg.episode_metrics_path = ""
    env = make_gymnasium_sf_env(cfg.env, cfg, SimpleNamespace(worker_index=0, vector_index=0, env_id=0)).env
    chunk = np.repeat(np.arange(16, dtype=np.float32)[:, None] / 100, 43, axis=1)
    try:
        observation, _ = env.reset(seed=104730)
        assert np.count_nonzero(observation["scheduled_chunk"]) == 0
        admissions = []
        for frame in range(5):
            observation, _, _, _, info = env.step(chunk.reshape(-1))
            admissions.append(observation["replay_clock"][0, 4])
            assert observation["replay_clock"][0, 2] == -1
        expected = np.zeros_like(chunk)
        expected[5:10] = chunk[5:10]
        np.testing.assert_allclose(observation["scheduled_chunk"], expected, atol=2e-7)
        np.testing.assert_array_equal(np.flatnonzero(observation["scheduled_index"]), [5])
        observation, _, _, _, info = env.step((-chunk).reshape(-1))
        admissions.append(observation["replay_clock"][0, 4])
        assert admissions == [True, False, False, False, False, True]
        np.testing.assert_array_equal(observation["replay_clock"][0, 2:4], [0, 5])
        expected[5] = 0
        np.testing.assert_allclose(observation["scheduled_chunk"], expected, atol=2e-7)
    finally:
        env.close()

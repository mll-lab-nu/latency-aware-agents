"""Long-lived native PPO contracts: full episodes, MC targets, and actor/V isolation."""
from functools import partial
import json
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from latency_bench.core.config import load_config
from training.gymnasium_sf import ppo


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


def test_complete_episode_mc_and_separate_value_updates(monkeypatch):
    torch.manual_seed(1)
    monkeypatch.setattr(ppo, "make_gymnasium_sf_env", lambda *args: CountingEpisode())
    settings = {"chunk_horizon": 16, "active_heads": [5, 10], "action_std": .02,
                "decision_frames": 5, "episode_frames": 10, "minibatch_size": 4,
                "target_kl": .02, "clip_ratio": .2}
    cfg = SimpleNamespace(env="counting", episode_metrics_path="")
    policy = ppo.PPOPolicy(CountingEpisode.observation_space, CountingEpisode.action_space, settings)
    value = nn.Linear(policy.input_dim + 1, 1)
    nn.init.zeros_(value.weight)
    nn.init.zeros_(value.bias)
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=1e-5)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=3e-4)
    original = {key: tensor.clone() for key, tensor in policy.state_dict().items()}
    env = gym.vector.SyncVectorEnv([partial(ppo.EpisodeWorker, cfg, i, 2, settings) for i in range(2)])
    try:
        episodes, metrics, data = ppo.collect(env, policy, value, range(1, 6), settings, stochastic=True, training=True)
        assert [row["seed"] for row in episodes] == list(range(1, 6))
        assert [row["length"] for row in episodes] == [7, 8, 6, 7, 8]
        assert metrics["frames"] == 36
        json.dumps(metrics, allow_nan=False)
        assert ppo.episode_summary(episodes)["mean_length"] == 7.2
        assert ppo.comparison(episodes, episodes) == {"mean_length_gain": 0, "paired_95_interval": [0, 0], "accepted": False}
        assert len(data["obs"]) == 10
        torch.testing.assert_close(data["returns"], (data["obs"][:, 1] - data["obs"][:, 0]) * 1.01 / 10)
        torch.testing.assert_close(data["advantages"], data["returns"])
        torch.testing.assert_close(policy.distribution(data["means"]).log_prob(data["latent"]), data["log_prob"])
        np.testing.assert_allclose(env.envs[0].unwrapped.actions[0][policy.active], data["latent"][0].tanh().numpy())
        result = ppo.update(policy, value, actor_optimizer, value_optimizer, data, settings, epochs=2, actor=False)
        assert result["actor_updates"] == 0
        assert result["value_updates"] == 6
        for key, tensor in policy.state_dict().items():
            torch.testing.assert_close(tensor, original[key], rtol=0, atol=0)
        result = ppo.update(policy, value, actor_optimizer, value_optimizer, data, settings, epochs=2, actor=True)
        assert 1 <= result["actor_updates"] <= 6
        assert result["value_updates"] == 6
        assert torch.isfinite(torch.tensor(result["actor_loss"]))
        assert policy.empirical_obs_normalizer.count == 0
        # A violated trust region stops actor work, not the remaining V minibatches.
        data["means"] += 1
        result = ppo.update(policy, value, actor_optimizer, value_optimizer, data, settings, epochs=2, actor=True)
        assert result["actor_updates"] == 0
        assert result["value_updates"] == 6
    finally:
        env.close()


def test_checkpoint_roundtrip_preserves_optimizer_and_rng(tmp_path):
    torch.manual_seed(9)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "round": 7, "rng": torch.get_rng_state()}
    path = tmp_path / "checkpoint.pth"
    ppo.save_checkpoint(path, state)
    expected = torch.rand(4)
    loaded = torch.load(path, weights_only=False)
    torch.set_rng_state(loaded["rng"])
    torch.testing.assert_close(torch.rand(4), expected)
    restored = nn.Linear(2, 1)
    restored.load_state_dict(loaded["model"])
    restored_optimizer = torch.optim.Adam(restored.parameters())
    restored_optimizer.load_state_dict(loaded["optimizer"])
    assert loaded["round"] == 7
    torch.testing.assert_close(restored(torch.ones(1, 2)), model(torch.ones(1, 2)))
    assert restored_optimizer.state[restored.weight]["step"] == 1


def test_native_ppo_observation_exposes_only_committed_heads():
    """Long-lived PPO/latency contract: returned heads 5:10 are visible before admission."""
    config = load_config("configs/examples/gymnasium/g1_sonic_balance_simple/ppo_train.yaml")
    cfg = ppo.build_train_cfg_from_config(config)
    cfg.episode_metrics_path = ""
    env = ppo.make_gymnasium_sf_env(cfg.env, cfg, SimpleNamespace(worker_index=0, vector_index=0, env_id=0))
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

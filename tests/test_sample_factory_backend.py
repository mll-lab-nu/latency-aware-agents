from __future__ import annotations

from latency_bench.core.config import load_config
from scripts.generate_sample_factory_same_latency_sweep import _build_eval_config, _build_train_config
from training.sample_factory_common.config_adapter import (
    apply_training_overrides,
    build_common_sf_args,
    load_training_config,
)


def test_sample_factory_training_configs_load():
    cfg = load_training_config("configs/experiments/demon_attack/sf_fixed_latency_4.yaml")
    assert cfg["backend"]["type"] == "sample_factory"
    assert cfg["env"]["name"] == "demon_attack"
    assert cfg["latency"]["train_values"] == [4]


def test_build_common_sf_args_emits_non_zero_latency_and_eval_settings():
    cfg = load_config("configs/experiments/deadly_corridor/sf_fixed_latency_4.yaml")
    args = build_common_sf_args(cfg, sf_env_name="doom_deadly_corridor")
    assert "--queue-latency-values" in args
    assert "4" in args
    assert "--eval-queue-latency-values" in args
    assert "--experiment" in args
    assert "deadly_corridor_sf_fixed_latency_4" in args


def test_apply_training_overrides_updates_run_name_train_dir_and_mode():
    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    updated = apply_training_overrides(
        cfg,
        run_name="override_run",
        train_dir="tmp/sample_factory",
        mode="train",
    )
    assert updated["experiment"]["name"] == "override_run"
    assert updated["backend"]["train_dir"] == "tmp/sample_factory"
    assert updated["backend"]["run_mode"] == "train"


def test_queue_latency_wrapper_exposes_fixed_non_zero_latency():
    import gymnasium as gym
    from training.sample_factory_common.latency_wrapper import QueueLatencyWrapper

    env = gym.make("CartPole-v1")
    wrapped = QueueLatencyWrapper(
        env,
        queue_latency_values=[4],
        latency_mode="episode_fixed",
        latency_info_keys=("queue_latency",),
    )
    _, info = wrapped.reset(seed=123)
    assert info["queue_latency"] == 4
    _, _, _, _, step_info = wrapped.step(0)
    assert step_info["queue_latency"] == 4
    wrapped.close()


def test_queue_latency_wrapper_auto_seeds_underlying_env_resets():
    import gymnasium as gym
    import numpy as np
    from gymnasium.spaces import Box, Discrete

    from training.sample_factory_common.latency_wrapper import QueueLatencyWrapper

    class SeedEchoEnv(gym.Env):
        observation_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        action_space = Discrete(2)

        def __init__(self):
            self.seeds: list[int | None] = []

        def reset(self, *, seed=None, options=None):
            self.seeds.append(seed)
            return np.zeros((1,), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}

    env = SeedEchoEnv()
    wrapped = QueueLatencyWrapper(
        env,
        queue_latency_values=[0],
        latency_mode="episode_fixed",
        seed=11,
        env_reset_seed_base=500,
    )
    wrapped.reset()
    wrapped.reset()
    wrapped.reset(seed=999)
    assert env.seeds == [500, 501, 999]


def test_same_latency_sweep_generator_emits_sample_factory_queue_eval_configs():
    train_cfg = _build_train_config(
        env_name="demon_attack",
        seed=0,
        latency=4,
        max_steps=3600,
        train_dir="/workspace/sample_factory_runs",
        output_root="/workspace/latency_sweeps",
        device="gpu",
        restart_behavior="resume",
        train_for_env_steps=1234,
    )
    eval_cfg = _build_eval_config(
        train_config=train_cfg,
        env_name="demon_attack",
        seed=0,
        latency=4,
        max_steps=3600,
        output_root="/workspace/latency_sweeps",
        eval_episodes=10,
    )
    assert train_cfg["training"]["train_for_env_steps"] == 1234
    assert train_cfg["latency"]["train_values"] == [4]
    assert eval_cfg["backend"]["run_mode"] == "eval"
    assert eval_cfg["latency"]["train_values"] == [4]
    assert eval_cfg["evaluation"]["eval_episodes"] == 10
    assert eval_cfg["evaluation"]["eval_latency_values"] == [4]
    assert eval_cfg["evaluation"]["eval_max_steps"] == 3600

from __future__ import annotations

from types import SimpleNamespace

import pytest

from latency_bench.core.config import load_config


def test_demon_attack_gpu_simulator_config_is_passed_to_training_adapter():
    from training.demon_attack_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/base/demon_attack.yaml")
    cfg["env"]["simulator"] = "gpu"
    cfg["env"]["gpu_env_device"] = "cuda"
    cfg["training"]["num_envs_per_worker"] = 64
    cfg["training"]["worker_num_splits"] = 1

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--simulator") + 1] == "gpu"
    assert args[args.index("--gpu-env-device") + 1] == "cuda"
    # GPU batched-env parallelism is driven by num_envs_per_worker.
    assert args[args.index("--gpu-env-batch-size") + 1] == "64"
    assert args[args.index("--num_envs_per_worker") + 1] == "1"
    assert args[args.index("--worker_num_splits") + 1] == "1"
    assert "--gpu-env-profile" not in args
    assert "--gpu-env-profile-interval" not in args
    assert args[args.index("--env_gpu_observations") + 1] == "True"
    assert args[args.index("--env_gpu_actions") + 1] == "True"


def test_demon_attack_gpu_simulator_splits_sample_factory_layout():
    from training.demon_attack_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/base/demon_attack.yaml")
    cfg["env"]["simulator"] = "gpu"
    cfg["env"]["gpu_env_device"] = "cuda"
    cfg["training"]["num_envs_per_worker"] = 64
    cfg["training"]["worker_num_splits"] = 2

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--num_envs_per_worker") + 1] == "2"
    assert args[args.index("--worker_num_splits") + 1] == "2"
    assert args[args.index("--gpu-env-batch-size") + 1] == "32"
    assert args[args.index("--env_gpu_observations") + 1] == "True"
    assert args[args.index("--env_gpu_actions") + 1] == "True"


def test_demon_attack_gpu_simulator_requires_even_split_layout():
    from training.demon_attack_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/base/demon_attack.yaml")
    cfg["env"]["simulator"] = "gpu"
    cfg["training"]["num_envs_per_worker"] = 65
    cfg["training"]["worker_num_splits"] = 2

    with pytest.raises(ValueError, match="num_envs_per_worker=65"):
        build_cli_args_from_config(cfg)


def test_demon_attack_gpu_split_layout_preserves_global_env_ids_and_seeds():
    from latency_bench.core.types import LatencySourceType
    from training.common.latency_config import sample_factory_latency_seed_from_flat_config
    from training.demon_attack_sf.batched_gpu_env import DemonAttackBatchedGpuEnv

    def fake_env(*, seed, latency_seed, worker_index, vector_index, env_id, num_agents):
        env = object.__new__(DemonAttackBatchedGpuEnv)
        env.cfg = SimpleNamespace(seed=seed, latency_seed=latency_seed, latency_type=LatencySourceType.ZERO.value)
        env.env_config = SimpleNamespace(worker_index=worker_index, vector_index=vector_index, env_id=env_id)
        env.num_agents = num_agents
        return env

    split1 = fake_env(seed=7, latency_seed=13, worker_index=0, vector_index=0, env_id=0, num_agents=512)
    split2_a = fake_env(seed=7, latency_seed=13, worker_index=0, vector_index=0, env_id=0, num_agents=256)
    split2_b = fake_env(seed=7, latency_seed=13, worker_index=0, vector_index=1, env_id=1, num_agents=256)

    split1_env_ids = [split1._sub_env_id(agent_idx) for agent_idx in range(512)]
    split2_env_ids = [split2_a._sub_env_id(agent_idx) for agent_idx in range(256)] + [
        split2_b._sub_env_id(agent_idx) for agent_idx in range(256)
    ]
    split1_reset_seeds = [split1._sub_seed(agent_idx) for agent_idx in range(512)]
    split2_reset_seeds = [split2_a._sub_seed(agent_idx) for agent_idx in range(256)] + [
        split2_b._sub_seed(agent_idx) for agent_idx in range(256)
    ]
    split1_latency_seeds = [
        sample_factory_latency_seed_from_flat_config(
            split1.cfg,
            fallback_seed=split1_reset_seeds[agent_idx],
            worker_index=0,
            vector_index=0,
            env_id=split1_env_ids[agent_idx],
        )
        for agent_idx in range(512)
    ]
    split2_latency_seeds = [
        sample_factory_latency_seed_from_flat_config(
            split2_a.cfg,
            fallback_seed=split2_reset_seeds[agent_idx],
            worker_index=0,
            vector_index=0,
            env_id=split2_env_ids[agent_idx],
        )
        for agent_idx in range(256)
    ] + [
        sample_factory_latency_seed_from_flat_config(
            split2_b.cfg,
            fallback_seed=split2_reset_seeds[256 + agent_idx],
            worker_index=0,
            vector_index=0,
            env_id=split2_env_ids[256 + agent_idx],
        )
        for agent_idx in range(256)
    ]

    assert split2_env_ids == split1_env_ids
    assert split2_reset_seeds == split1_reset_seeds
    assert split2_latency_seeds == split1_latency_seeds


def test_demon_attack_base_config_uses_gpu_simulator():
    cfg = load_config("configs/base/demon_attack.yaml")

    assert cfg["env"]["simulator"] == "gpu"
    # gpu_env_batch_size is deprecated; parallelism comes from training.num_envs_per_worker.
    assert "gpu_env_batch_size" not in cfg["env"]
    assert "gpu_env_profile" not in cfg["env"]
    assert "gpu_env_profile_interval" not in cfg["env"]


def test_deadly_corridor_base_config_defaults_to_cpu_simulator():
    cfg = load_config("configs/base/deadly_corridor.yaml")

    assert cfg["env"]["simulator"] == "cpu"
    assert "gpu_env_device" not in cfg["env"]
    assert "gpu_env_profile" not in cfg["env"]
    assert "gpu_env_profile_interval" not in cfg["env"]


def test_demon_attack_gpu_simulator_constructs_gpu_env_without_cpu_fallback(monkeypatch):
    from training.demon_attack_sf import integration

    def fail_cpu_env(**_kwargs):
        raise AssertionError("simulator=gpu must not construct the CPU Demon Attack env")

    class FakeGpuEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(integration, "make_demon_attack_benchmark_env", fail_cpu_env)
    monkeypatch.setattr(
        "training.demon_attack_sf.batched_gpu_env.DemonAttackBatchedGpuEnv",
        FakeGpuEnv,
    )

    env = integration.make_demon_attack_sf_env(
        "latency_demon_attack",
        cfg=SimpleNamespace(
            simulator="gpu",
            env_fps=60,
            obs_fps=15,
            screen_size=84,
            noop_max=30,
            frame_stack=4,
            terminal_on_life_loss=True,
            clip_reward=True,
            repeat_action_probability=0.0,
            full_action_space=False,
            env_id="ALE/DemonAttack-v5",
            max_episode_steps=25000,
            gpu_env_device="cuda",
            gpu_env_batch_size=64,
        ),
        env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
    )

    assert isinstance(env, FakeGpuEnv)
    assert env.kwargs["gpu_env_batch_size"] == 64


def test_demon_attack_sample_factory_probe_env_config_is_explicit(monkeypatch):
    from training.demon_attack_sf import integration

    class FakeGpuEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "training.demon_attack_sf.batched_gpu_env.DemonAttackBatchedGpuEnv",
        FakeGpuEnv,
    )

    env = integration.make_demon_attack_sf_env(
        "latency_demon_attack",
        cfg=SimpleNamespace(
            simulator="gpu",
            env_fps=60,
            obs_fps=15,
            screen_size=84,
            noop_max=30,
            frame_stack=4,
            terminal_on_life_loss=True,
            clip_reward=True,
            repeat_action_probability=0.0,
            full_action_space=False,
            env_id="ALE/DemonAttack-v5",
            max_episode_steps=25000,
            gpu_env_device="cuda",
            gpu_env_batch_size=64,
        ),
        env_config=None,
    )

    assert env.kwargs["env_config"].worker_index == 0
    assert env.kwargs["env_config"].vector_index == 0
    assert env.kwargs["env_config"].env_id == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("screen_size", 96, "screen_size=84"),
        ("repeat_action_probability", 0.25, "repeat_action_probability=0.0"),
        ("full_action_space", True, "full_action_space=False"),
        ("env_id", "ALE/Pong-v5", "ALE/DemonAttack-v5"),
    ],
)
def test_demon_attack_gpu_training_rejects_unsupported_semantics(field, value, message):
    from training.demon_attack_sf import integration

    cfg = SimpleNamespace(
        simulator="gpu",
        env_fps=60,
        obs_fps=15,
        screen_size=84,
        noop_max=30,
        frame_stack=4,
        terminal_on_life_loss=True,
        clip_reward=True,
        repeat_action_probability=0.0,
        full_action_space=False,
        env_id="ALE/DemonAttack-v5",
        max_episode_steps=25000,
        gpu_env_device="cuda",
        gpu_env_batch_size=64,
    )
    setattr(cfg, field, value)

    with pytest.raises(ValueError, match=message):
        integration.make_demon_attack_sf_env(
            "latency_demon_attack",
            cfg=cfg,
            env_config=SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        )

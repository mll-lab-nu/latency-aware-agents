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

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--simulator") + 1] == "gpu"
    assert args[args.index("--gpu-env-device") + 1] == "cuda"
    # GPU batched-env parallelism is driven by num_envs_per_worker.
    assert args[args.index("--gpu-env-batch-size") + 1] == "64"
    assert "--gpu-env-profile" not in args
    assert "--gpu-env-profile-interval" not in args
    assert args[args.index("--env_gpu_observations") + 1] == "True"
    assert args[args.index("--env_gpu_actions") + 1] == "True"


def test_deadly_corridor_gpu_simulator_config_is_passed_to_training_adapter():
    from training.deadly_corridor_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/base/deadly_corridor.yaml")
    cfg["env"]["simulator"] = "gpu"
    cfg["env"]["gpu_env_device"] = "cuda"
    cfg["training"]["num_envs_per_worker"] = 32
    cfg["env"]["gpu_env_profile"] = True
    cfg["env"]["gpu_env_profile_interval"] = 17

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--simulator") + 1] == "gpu"
    assert args[args.index("--gpu-env-device") + 1] == "cuda"
    # GPU batched-env parallelism is driven by num_envs_per_worker.
    assert args[args.index("--gpu-env-batch-size") + 1] == "32"
    assert args[args.index("--gpu-env-profile") + 1] == "True"
    assert args[args.index("--gpu-env-profile-interval") + 1] == "17"


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


def test_deadly_corridor_gpu_simulator_does_not_construct_cpu_env(monkeypatch):
    from training.deadly_corridor_sf import integration

    def fail_cpu_env(*_args, **_kwargs):
        raise AssertionError("simulator=gpu must not construct the CPU Deadly Corridor env")

    monkeypatch.setattr(
        "sf_examples.vizdoom.doom.doom_utils.make_doom_env_from_spec",
        fail_cpu_env,
    )
    monkeypatch.setattr(
        "training.deadly_corridor_sf.deadly_corridor_cuda_extension.load_deadly_corridor_cuda_extension",
        lambda: SimpleNamespace(__name__="deadly_corridor_cuda_kernels"),
    )

    with pytest.raises(NotImplementedError, match="Deadly Corridor CUDA simulator"):
        integration.make_latency_doom_env_from_spec(
            SimpleNamespace(name="doom_deadly_corridor", reward_scaling=0.01),
            "doom_deadly_corridor",
            SimpleNamespace(
                simulator="gpu",
                env_fps=35,
                obs_fps=8.75,
                gpu_env_device="cuda",
                gpu_env_batch_size=32,
                gpu_env_profile=False,
                gpu_env_profile_interval=1000,
            ),
            SimpleNamespace(worker_index=0, vector_index=0, env_id=0),
        )

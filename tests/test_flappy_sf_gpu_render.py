from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from latency_bench.core.config import load_config
from training.sample_factory_common.config_adapter import load_training_config, validate_training_config


def _flappy_gpu_cfg(**overrides):
    values = {
        "gym_id": "FlappyBird-v0",
        "use_lidar": False,
        "normalize_obs": True,
        "audio_on": False,
        "pipe_gap": 100,
        "bird_color": "yellow",
        "pipe_color": "green",
        "background": "day",
        "debug": False,
        "screen_size": "",
        "score_limit": -1,
        "obs_resize": "84,84",
        "obs_mode": "image_only",
        "frame_stack": 4,
        "max_episode_steps": 0,
        "queue_latency_values": "0",
        "latency_mode": "episode_fixed",
        "latency_sampling_mode": "uniform_all",
        "episodes_per_stage": 10,
        "randomize_within_stage": False,
        "within_episode_change_interval": 1,
        "within_episode_change_prob": 1.0,
        "add_latency_info": False,
        "seed": 0,
        "episode_metrics_path": "",
        "use_gpu_render": True,
        "gpu_render_device": "cpu",
        "gpu_render_profile": False,
        "gpu_render_profile_interval": 1000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _env_config():
    return SimpleNamespace(worker_index=0, vector_index=0, env_id=0)


def test_flappy_sf_gpu_render_cpu_reset_contract_image_only():
    torch = pytest.importorskip("torch")
    from gymnasium.spaces import Dict as GymDict

    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env("latency_flappy", cfg=_flappy_gpu_cfg(), env_config=_env_config())
    try:
        assert getattr(env, "num_agents") == 1
        assert getattr(env, "is_multiagent") is True
        assert isinstance(env.observation_space, GymDict)
        assert env.observation_space.spaces["obs"].shape == (12, 84, 84)

        obs, infos = env.reset(seed=123)

        assert isinstance(obs, dict)
        assert isinstance(obs["obs"], torch.Tensor)
        assert obs["obs"].shape == (1, 12, 84, 84)
        assert obs["obs"].dtype == torch.uint8
        assert obs["obs"].device.type == "cpu"
        assert isinstance(infos, list)
        assert len(infos) == 1
        assert "numeric_obs" in infos[0]
    finally:
        env.close()


def test_flappy_sf_gpu_render_cpu_latency_channel_contract():
    torch = pytest.importorskip("torch")

    from modeling.flappy_ppo.preprocessing import latency_channel_value
    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env(
        "latency_flappy",
        cfg=_flappy_gpu_cfg(obs_mode="latency_channel", queue_latency_values="3"),
        env_config=_env_config(),
    )
    try:
        obs, _ = env.reset(seed=123)

        assert isinstance(obs["obs"], torch.Tensor)
        assert obs["obs"].shape == (1, 13, 84, 84)
        assert torch.all(obs["obs"][0, -1] == int(latency_channel_value(3, 3)))
    finally:
        env.close()


def test_flappy_sf_gpu_render_auto_resets_on_done():
    pytest.importorskip("torch")
    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env("latency_flappy", cfg=_flappy_gpu_cfg(), env_config=_env_config())
    try:
        env.reset(seed=123)
        env.unwrapped._score_limit = 0

        obs, rewards, terminated, truncated, infos = env.step(np.asarray([0], dtype=np.int64))

        assert obs["obs"].shape == (1, 12, 84, 84)
        assert rewards.shape == (1,)
        assert terminated.shape == (1,)
        assert truncated.shape == (1,)
        assert bool(terminated[0] or truncated[0])
        assert isinstance(infos, list)
        assert "reset_info" in infos[0]
        assert "numeric_obs" in infos[0]["reset_info"]
    finally:
        env.close()


def test_flappy_gpu_render_profile_disabled_by_default(capsys):
    pytest.importorskip("torch")
    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env("latency_flappy", cfg=_flappy_gpu_cfg(), env_config=_env_config())
    try:
        assert env.gpu_render_profiler.enabled is False
        env.reset(seed=123)
        env.step(np.asarray([0], dtype=np.int64))

        captured = capsys.readouterr()
        assert "[FlappyGpuRenderProfile]" not in captured.out
    finally:
        env.close()


def test_flappy_gpu_render_profile_preserves_step_contract(capsys):
    torch = pytest.importorskip("torch")
    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env(
        "latency_flappy",
        cfg=_flappy_gpu_cfg(gpu_render_profile=True, gpu_render_profile_interval=1),
        env_config=_env_config(),
    )
    try:
        obs, infos = env.reset(seed=123)
        assert isinstance(obs["obs"], torch.Tensor)
        assert obs["obs"].shape == (1, 12, 84, 84)
        assert isinstance(infos, list)
        assert len(infos) == 1

        obs, rewards, terminated, truncated, infos = env.step(np.asarray([0], dtype=np.int64))

        assert isinstance(obs["obs"], torch.Tensor)
        assert obs["obs"].shape == (1, 12, 84, 84)
        assert rewards.shape == (1,)
        assert terminated.shape == (1,)
        assert truncated.shape == (1,)
        assert isinstance(infos, list)
        assert len(infos) == 1

        captured = capsys.readouterr()
        assert "[FlappyGpuRenderProfile]" in captured.out
        for field in (
            "env_step_s",
            "get_render_state_s",
            "state_batch_from_states_s",
            "render_into_s",
            "pipeline_s",
            "reset_obs_total_s",
            "step_obs_total_s",
        ):
            assert f"{field}_avg_ms=" in captured.out
    finally:
        env.close()


def test_flappy_gpu_render_profile_does_not_touch_cpu_render_path(capsys):
    from gymnasium.spaces import Box

    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env(
        "latency_flappy",
        cfg=_flappy_gpu_cfg(use_gpu_render=False, gpu_render_profile=True, gpu_render_profile_interval=1),
        env_config=_env_config(),
    )
    try:
        assert "gpu_render_profiler" not in vars(env)
        assert isinstance(env.observation_space, Box)

        obs, _ = env.reset(seed=123)
        assert isinstance(obs, np.ndarray)
        env.step(0)

        captured = capsys.readouterr()
        assert "[FlappyGpuRenderProfile]" not in captured.out
    finally:
        env.close()


def test_flappy_gpu_render_device_resolution_fails_fast_without_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    from training.flappy_sf.gpu_rendering import resolve_gpu_render_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_gpu_render_device("cpu").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_gpu_render_device("cuda")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_gpu_render_device("auto")


def test_flappy_gpu_render_config_emits_cuda_actor_worker_args():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["use_gpu_render"] = True
    cfg["env"]["gpu_render_device"] = "cuda"
    cfg["backend"].pop("actor_worker_gpus", None)

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--use-gpu-render") + 1] == "True"
    assert args[args.index("--gpu-render-device") + 1] == "cuda"
    assert args[args.index("--actor_worker_gpus") + 1] == "0"
    assert args[args.index("--env_gpu_observations") + 1] == "True"
    assert args[args.index("--env_gpu_actions") + 1] == "False"
    assert args[args.index("--batched_sampling") + 1] == "True"


def test_flappy_gpu_render_config_cpu_does_not_force_actor_worker_gpu():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["use_gpu_render"] = True
    cfg["env"]["gpu_render_device"] = "cpu"
    cfg["backend"].pop("actor_worker_gpus", None)

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--use-gpu-render") + 1] == "True"
    assert args[args.index("--gpu-render-device") + 1] == "cpu"
    assert "--actor_worker_gpus" not in args
    assert args[args.index("--env_gpu_observations") + 1] == "False"
    assert args[args.index("--env_gpu_actions") + 1] == "False"
    assert args[args.index("--batched_sampling") + 1] == "True"


def test_flappy_gpu_render_config_maps_profile_args():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["use_gpu_render"] = True
    cfg["env"]["gpu_render_device"] = "cpu"
    cfg["env"]["gpu_render_profile"] = True
    cfg["env"]["gpu_render_profile_interval"] = 7

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--gpu-render-profile") + 1] == "True"
    assert args[args.index("--gpu-render-profile-interval") + 1] == "7"


def test_flappy_gpu_render_validation_requires_square_obs_resize():
    cfg = load_training_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["use_gpu_render"] = True
    cfg["env"]["gpu_render_device"] = "cpu"
    cfg["env"]["obs_resize"] = [84, 96]

    with pytest.raises(ValueError, match="square"):
        validate_training_config(cfg)


def test_flappy_gpu_render_validation_rejects_explicit_non_batched_sampling():
    cfg = load_training_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["use_gpu_render"] = True
    cfg["env"]["gpu_render_device"] = "cpu"
    cfg["training"]["batched_sampling"] = False

    with pytest.raises(ValueError, match="batched_sampling"):
        validate_training_config(cfg)


def test_flappy_gpu_render_validation_rejects_non_positive_profile_interval():
    cfg = load_training_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["gpu_render_profile_interval"] = 0

    with pytest.raises(ValueError, match="gpu_render_profile_interval"):
        validate_training_config(cfg)

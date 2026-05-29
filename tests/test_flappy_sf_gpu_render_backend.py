from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from latency_bench.core.config import load_config


def _gpu_queue_cfg(**overrides):
    values = {
        "gym_id": "FlappyBird-v0",
        "use_lidar": False,
        "normalize_obs": True,
        "audio_on": False,
        "screen_size": "",
        "policy_screen_size": "84,84",
        "pipe_gap": 100,
        "bird_color": "yellow",
        "pipe_color": "green",
        "background": "day",
        "score_limit": -1,
        "debug": False,
        "max_episode_steps": 0,
        "queue_latency_values": "2",
        "latency_mode": "episode_fixed",
        "latency_sampling_mode": "uniform_all",
        "latency_distribution": "discrete",
        "latency_normal_mean": 0.0,
        "latency_normal_std": 0.0,
        "latency_normal_min": 0,
        "latency_normal_max": 0,
        "latency_uniform_min": 0,
        "latency_uniform_max": 0,
        "episodes_per_stage": 10,
        "randomize_within_stage": False,
        "within_episode_change_interval": 1,
        "within_episode_change_prob": 1.0,
        "add_latency_info": True,
        "seed": 0,
        "episode_metrics_path": "",
        "frame_stack": 4,
        "use_gpu_render": True,
        "gpu_render_device": "cpu",
        "gpu_render_batch_size": 2,
        "gpu_render_profile": False,
        "gpu_render_profile_interval": 1000,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _env_config():
    return SimpleNamespace(worker_index=0, vector_index=0, env_id=0)


def test_flappy_gpu_render_config_uses_batched_env_without_changing_queue_latency_args():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["use_gpu_render"] = True
    cfg["env"]["gpu_render_device"] = "cpu"
    cfg["env"]["gpu_render_profile"] = True
    cfg["env"]["gpu_render_profile_interval"] = 7
    cfg["training"]["num_envs_per_worker"] = 8
    cfg["training"]["worker_num_splits"] = 1

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--use-gpu-render") + 1] == "True"
    assert args[args.index("--gpu-render-device") + 1] == "cpu"
    assert args[args.index("--gpu-render-batch-size") + 1] == "8"
    assert args[args.index("--gpu-render-profile") + 1] == "True"
    assert args[args.index("--gpu-render-profile-interval") + 1] == "7"
    assert args[args.index("--num_envs_per_worker") + 1] == "1"
    assert args[args.index("--batched_sampling") + 1] == "True"
    assert "--queue-latency-values" in args
    assert "--latency-type" not in args


def test_flappy_gpu_render_env_preserves_queue_latency_step_semantics():
    torch = pytest.importorskip("torch")
    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env("latency_flappy", cfg=_gpu_queue_cfg(), env_config=_env_config())
    try:
        assert getattr(env, "is_multiagent") is True
        assert getattr(env, "num_agents") == 2

        obs, infos = env.reset(seed=123)

        assert isinstance(obs, dict)
        assert isinstance(obs["obs"], torch.Tensor)
        assert obs["obs"].shape == (2, 12, 84, 84)
        assert len(infos) == 2
        assert infos[0]["queue_latency"] == 2
        assert "latency_raw_frames" not in infos[0]

        _obs, _reward, _terminated, _truncated, infos = env.step(np.asarray([1, 0], dtype=np.int64))
        assert infos[0]["queue_latency"] == 2
        assert infos[0]["frame_skip"] == 2
        assert infos[0]["latency_frame_skip"] == 2
        assert infos[0]["queued_action"] == 1
        assert infos[0]["applied_action"] == 0

        env.step(np.asarray([0, 0], dtype=np.int64))
        _obs, _reward, _terminated, _truncated, infos = env.step(np.asarray([0, 0], dtype=np.int64))

        assert infos[0]["applied_action"] == 1
    finally:
        env.close()

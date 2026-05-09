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
        "debug_timelimit_diagnostics": False,
        "screen_size": "",
        "score_limit": -1,
        "obs_resize": "84,84",
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
        "env_fps": 30,
        "obs_fps": 30,
        "episode_metrics_path": "",
        "use_gpu_render": True,
        "gpu_render_device": "cpu",
        "gpu_render_batch_size": 1,
        "gpu_render_profile": False,
        "gpu_render_profile_interval": 1000,
        "hold_policy": "one_frame_then_noop",
        "ordering_policy": "latest_ready",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _env_config():
    return SimpleNamespace(worker_index=0, vector_index=0, env_id=0)


def test_flappy_sf_gpu_render_cpu_reset_contract_image_only():
    torch = pytest.importorskip("torch")
    from gymnasium.spaces import Dict as GymDict

    from training.flappy_sf.batched_gpu_env import FlappyBatchedGpuRenderedEnv
    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env("latency_flappy", cfg=_flappy_gpu_cfg(), env_config=_env_config())
    try:
        assert isinstance(env, FlappyBatchedGpuRenderedEnv)
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


def test_flappy_sf_timelimit_diagnostics_default_quiet(capsys):
    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env("latency_flappy", cfg=_flappy_gpu_cfg(), env_config=_env_config())
    try:
        captured = capsys.readouterr()
        assert "[flappy_sf][timelimit]" not in captured.out
    finally:
        env.close()


def test_flappy_sf_timelimit_diagnostics_prints_essential_fields_when_enabled(capsys):
    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env(
        "latency_flappy",
        cfg=_flappy_gpu_cfg(debug_timelimit_diagnostics=True, max_episode_steps=17),
        env_config=_env_config(),
    )
    try:
        captured = capsys.readouterr()
        assert "[flappy_sf][timelimit]" in captured.out
        assert "cfg.max_episode_steps=17" in captured.out
        assert "wrapper_chain=" in captured.out
        assert "obs_shape=(12, 84, 84)" in captured.out
        assert "gpu_render_batch_size=1" in captured.out
    finally:
        env.close()


def test_flappy_sf_gpu_render_has_no_timelimit_when_max_episode_steps_absent():
    pytest.importorskip("torch")
    from training.flappy_sf.integration import _time_limit_max_episode_steps, make_flappy_sf_env

    env = make_flappy_sf_env(
        "latency_flappy",
        cfg=_flappy_gpu_cfg(max_episode_steps=0),
        env_config=_env_config(),
    )
    try:
        assert _time_limit_max_episode_steps(env.sub_envs[0]) is None
    finally:
        env.close()


def test_flappy_sf_gpu_render_batched_reset_contract_image_only():
    torch = pytest.importorskip("torch")

    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env(
        "latency_flappy",
        cfg=_flappy_gpu_cfg(gpu_render_batch_size=4),
        env_config=_env_config(),
    )
    try:
        assert getattr(env, "num_agents") == 4
        assert getattr(env, "is_multiagent") is True
        assert env.observation_space.spaces["obs"].shape == (12, 84, 84)

        obs, infos = env.reset(seed=123)

        assert isinstance(obs["obs"], torch.Tensor)
        assert obs["obs"].shape == (4, 12, 84, 84)
        assert obs["obs"].dtype == torch.uint8
        assert obs["obs"].device.type == "cpu"
        assert isinstance(infos, list)
        assert len(infos) == 4
        assert all("numeric_obs" in info for info in infos)
    finally:
        env.close()


def test_flappy_sf_gpu_render_auto_resets_on_done():
    pytest.importorskip("torch")
    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env("latency_flappy", cfg=_flappy_gpu_cfg(), env_config=_env_config())
    try:
        env.reset(seed=123)
        env.sub_envs[0].unwrapped._score_limit = 0

        obs, rewards, terminated, truncated, infos = env.step(np.asarray([0], dtype=np.int64))

        assert obs["obs"].shape == (1, 12, 84, 84)
        assert rewards.shape == (1,)
        assert terminated.shape == (1,)
        assert truncated.shape == (1,)
        assert bool(terminated[0] or truncated[0])
        assert isinstance(infos, list)
        assert "reset_info" in infos[0]
        assert "numeric_obs" in infos[0]["reset_info"]
        assert infos[0]["episode_extra_stats"].keys() >= {
            "episode/return",
            "episode/raw_return",
            "episode/clipped_return",
            "episode/length",
            "episode/raw_frames",
            "episode/index",
            "episode/queue_latency",
        }
    finally:
        env.close()


def _stack_frames_equal(frame_stack: np.ndarray, agent_idx: int, *, stack_frames: int, base_channels: int = 3) -> bool:
    stack = frame_stack[agent_idx]
    chunks = [
        stack[:, :, idx * base_channels : (idx + 1) * base_channels]
        for idx in range(stack_frames)
    ]
    return all(np.array_equal(chunks[0], item) for item in chunks[1:])


def test_flappy_sf_gpu_render_auto_reset_resets_only_done_agent_frame_stack():
    pytest.importorskip("torch")
    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env(
        "latency_flappy",
        cfg=_flappy_gpu_cfg(gpu_render_batch_size=2, frame_stack=3),
        env_config=_env_config(),
    )
    try:
        env.reset(seed=123)

        for _ in range(5):
            env.step(np.asarray([1, 1], dtype=np.int64))
            stack = env.pipeline.current_frame_stack
            assert stack is not None
            if not _stack_frames_equal(stack, 0, stack_frames=3) and not _stack_frames_equal(
                stack, 1, stack_frames=3
            ):
                break
        stack = env.pipeline.current_frame_stack
        assert stack is not None
        assert not _stack_frames_equal(stack, 0, stack_frames=3)
        assert not _stack_frames_equal(stack, 1, stack_frames=3)

        env.sub_envs[0].unwrapped._score_limit = 0
        obs, rewards, terminated, truncated, infos = env.step(np.asarray([0, 1], dtype=np.int64))

        assert obs["obs"].shape == (2, 9, 84, 84)
        assert rewards.shape == (2,)
        assert bool(terminated[0] or truncated[0])
        assert not bool(terminated[1] or truncated[1])
        assert "reset_info" in infos[0]
        assert "reset_info" not in infos[1]
        assert "episode_extra_stats" in infos[0]
        assert infos[0]["episode_extra_stats"].keys() >= {
            "episode/return",
            "episode/raw_return",
            "episode/clipped_return",
            "episode/length",
            "episode/raw_frames",
            "episode/index",
            "episode/queue_latency",
        }

        stack = env.pipeline.current_frame_stack
        assert stack is not None
        assert _stack_frames_equal(stack, 0, stack_frames=3)
        assert not _stack_frames_equal(stack, 1, stack_frames=3)
    finally:
        env.close()


def test_flappy_sf_gpu_render_manual_reset_preserves_terminal_info_and_resets_requested_agent_only():
    pytest.importorskip("torch")
    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env(
        "latency_flappy",
        cfg=_flappy_gpu_cfg(gpu_render_batch_size=2, frame_stack=3),
        env_config=_env_config(),
    )
    try:
        env.reset(seed=123)
        env.set_auto_reset_done(False)

        for _ in range(5):
            env.step(np.asarray([1, 1], dtype=np.int64))
            stack = env.pipeline.current_frame_stack
            assert stack is not None
            if not _stack_frames_equal(stack, 0, stack_frames=3) and not _stack_frames_equal(
                stack, 1, stack_frames=3
            ):
                break

        env.sub_envs[0].unwrapped._score_limit = 0
        _obs, rewards, terminated, truncated, infos = env.step(np.asarray([0, 1], dtype=np.int64))

        assert rewards.shape == (2,)
        assert bool(terminated[0] or truncated[0])
        assert not bool(terminated[1] or truncated[1])
        assert "episode_extra_stats" in infos[0]
        assert "reset_info" not in infos[0]
        assert "reset_info" not in infos[1]

        _obs, reset_infos = env.reset_agents({0: 777}, active_mask=[True, True])
        assert "numeric_obs" in reset_infos[0]
        stack = env.pipeline.current_frame_stack
        assert stack is not None
        assert _stack_frames_equal(stack, 0, stack_frames=3)
        assert not _stack_frames_equal(stack, 1, stack_frames=3)
    finally:
        env.close()


def test_flappy_sf_gpu_render_inactive_agent_does_not_emit_done_reward_or_reset():
    pytest.importorskip("torch")
    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env(
        "latency_flappy",
        cfg=_flappy_gpu_cfg(gpu_render_batch_size=2),
        env_config=_env_config(),
    )
    try:
        env.reset(seed=123)
        env.set_auto_reset_done(False)
        env.set_active_agents([False, True])
        env.sub_envs[0].unwrapped._score_limit = 0

        _obs, rewards, terminated, truncated, infos = env.step(np.asarray([0, 1], dtype=np.int64))

        assert rewards[0] == 0.0
        assert not bool(terminated[0])
        assert not bool(truncated[0])
        assert infos[0]["inactive"] is True
        assert "reset_info" not in infos[0]
    finally:
        env.close()


def test_flappy_sf_gpu_render_accepts_sample_factory_discrete_action_format():
    pytest.importorskip("torch")
    from training.flappy_sf.integration import make_flappy_sf_env

    env = make_flappy_sf_env(
        "latency_flappy",
        cfg=_flappy_gpu_cfg(gpu_render_batch_size=4),
        env_config=_env_config(),
    )
    try:
        env.reset(seed=123)

        obs, rewards, terminated, truncated, infos = env.step(
            np.asarray([0, 1, 0, 1], dtype=np.int32)
        )
        assert obs["obs"].shape == (4, 12, 84, 84)
        assert rewards.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)
        assert len(infos) == 4

        obs, rewards, terminated, truncated, infos = env.step(
            np.asarray([[0], [1], [0], [1]], dtype=np.int32)
        )
        assert obs["obs"].shape == (4, 12, 84, 84)
        assert rewards.shape == (4,)
        assert terminated.shape == (4,)
        assert truncated.shape == (4,)
        assert len(infos) == 4
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
    cfg["training"]["worker_num_splits"] = 1
    cfg["backend"].pop("actor_worker_gpus", None)

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--use-gpu-render") + 1] == "True"
    assert args[args.index("--gpu-render-device") + 1] == "cuda"
    assert args[args.index("--actor_worker_gpus") + 1] == "0"
    assert args[args.index("--env_gpu_observations") + 1] == "True"
    assert args[args.index("--env_gpu_actions") + 1] == "False"
    assert args[args.index("--batched_sampling") + 1] == "True"
    assert args[args.index("--gpu-render-batch-size") + 1] == str(cfg["training"]["num_envs_per_worker"])
    assert args[args.index("--num_envs_per_worker") + 1] == "1"


def test_flappy_gpu_render_config_cpu_does_not_force_actor_worker_gpu():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["use_gpu_render"] = True
    cfg["env"]["gpu_render_device"] = "cpu"
    cfg["training"]["worker_num_splits"] = 1
    cfg["backend"].pop("actor_worker_gpus", None)

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--use-gpu-render") + 1] == "True"
    assert args[args.index("--gpu-render-device") + 1] == "cpu"
    assert "--actor_worker_gpus" not in args
    assert args[args.index("--env_gpu_observations") + 1] == "False"
    assert args[args.index("--env_gpu_actions") + 1] == "False"
    assert args[args.index("--batched_sampling") + 1] == "True"
    assert args[args.index("--gpu-render-batch-size") + 1] == str(cfg["training"]["num_envs_per_worker"])
    assert args[args.index("--num_envs_per_worker") + 1] == "1"


def test_flappy_gpu_render_config_does_not_remap_cpu_render_path():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["use_gpu_render"] = False
    cfg["training"]["num_envs_per_worker"] = 7
    cfg["training"]["worker_num_splits"] = 2

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--use-gpu-render") + 1] == "False"
    assert args[args.index("--num_envs_per_worker") + 1] == "7"
    assert args[args.index("--worker_num_splits") + 1] == "2"
    assert "--gpu-render-batch-size" not in args


def test_flappy_gpu_render_config_maps_profile_args():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["use_gpu_render"] = True
    cfg["env"]["gpu_render_device"] = "cpu"
    cfg["env"]["gpu_render_profile"] = True
    cfg["env"]["gpu_render_profile_interval"] = 7
    cfg["training"]["worker_num_splits"] = 1

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--gpu-render-profile") + 1] == "True"
    assert args[args.index("--gpu-render-profile-interval") + 1] == "7"


def test_flappy_gpu_render_config_maps_timelimit_diagnostics_arg():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["debug_timelimit_diagnostics"] = True
    cfg["training"]["worker_num_splits"] = 1

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--debug-timelimit-diagnostics") + 1] == "True"


def test_flappy_gpu_render_validation_requires_square_obs_resize():
    cfg = load_training_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["use_gpu_render"] = True
    cfg["env"]["gpu_render_device"] = "cpu"
    cfg["env"]["obs_resize"] = [84, 96]
    cfg["training"]["worker_num_splits"] = 1

    with pytest.raises(ValueError, match="square"):
        validate_training_config(cfg)


def test_flappy_gpu_render_validation_rejects_explicit_non_batched_sampling():
    cfg = load_training_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["use_gpu_render"] = True
    cfg["env"]["gpu_render_device"] = "cpu"
    cfg["training"]["worker_num_splits"] = 1
    cfg["training"]["batched_sampling"] = False

    with pytest.raises(ValueError, match="batched_sampling"):
        validate_training_config(cfg)


def test_flappy_gpu_render_validation_rejects_worker_splits_not_one():
    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["use_gpu_render"] = True
    cfg["env"]["gpu_render_device"] = "cpu"
    cfg["training"]["num_envs_per_worker"] = 4
    cfg["training"]["worker_num_splits"] = 2

    with pytest.raises(ValueError, match="worker_num_splits"):
        validate_training_config(cfg)


def test_flappy_gpu_render_cli_rejects_worker_splits_not_one_without_remap():
    from training.flappy_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["use_gpu_render"] = True
    cfg["env"]["gpu_render_device"] = "cpu"
    cfg["training"]["num_envs_per_worker"] = 4
    cfg["training"]["worker_num_splits"] = 2

    with pytest.raises(ValueError, match="worker_num_splits"):
        build_cli_args_from_config(cfg)


def test_flappy_gpu_render_validation_rejects_non_positive_profile_interval():
    cfg = load_training_config("configs/experiments/flappy/sf_zero_latency.yaml")
    cfg["env"]["gpu_render_profile_interval"] = 0

    with pytest.raises(ValueError, match="gpu_render_profile_interval"):
        validate_training_config(cfg)

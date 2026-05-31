from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _cuda_available() -> bool:
    torch = pytest.importorskip("torch")
    return bool(torch.cuda.is_available())


def _cfg(*, simulator: str, add_latency_info: bool, max_episode_steps: int = 25000, **overrides):
    from training.demon_attack_sf.integration import SF_ENV_NAME

    values = {
        "env": SF_ENV_NAME,
        "env_gpu_actions": False,
        "env_gpu_observations": False,
        "seed": 0,
        "simulator": simulator,
        "env_id": "ALE/DemonAttack-v5",
        "screen_size": 84,
        "env_fps": 60,
        "obs_fps": 15,
        "noop_max": 1,
        "frame_stack": 4,
        "terminal_on_life_loss": True,
        "clip_reward": True,
        "repeat_action_probability": 0.0,
        "full_action_space": False,
        "max_episode_steps": max_episode_steps,
        "latency_type": "fixed",
        "fixed_latency_ms": 1000.0 / 60.0,
        "mean_latency_ms": None,
        "std_latency_ms": None,
        "min_latency_ms": None,
        "max_latency_ms": None,
        "latency_seed": 123,
        "add_latency_info": add_latency_info,
        "hold_policy": "hold",
        "ordering_policy": "latest_ready",
        "max_pending_actions": None,
        "episode_metrics_path": "",
        "export_env_raw_rgb_frames": False,
        "gpu_env_device": "cuda",
        "gpu_env_batch_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_cpu_reference_env(cfg):
    from training.demon_attack_sf.integration import make_reference_env

    return make_reference_env(cfg)


def _make_gpu_env(cfg, *, batch_size: int = 1):
    from sample_factory.utils.attr_dict import AttrDict

    from training.demon_attack_sf.batched_gpu_env import DemonAttackBatchedGpuEnv

    return DemonAttackBatchedGpuEnv(
        cfg=cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        gpu_env_device="cuda",
        gpu_env_batch_size=batch_size,
    )


def _agent_obs_array(obs, agent_idx: int = 0) -> np.ndarray:
    tensor = obs["obs"][agent_idx]
    return tensor.detach().cpu().numpy()


def _scalar_list(value) -> list:
    if hasattr(value, "detach"):
        return value.detach().cpu().reshape(-1).tolist()
    return np.asarray(value).reshape(-1).tolist()


def _assert_info_equal(cpu_info: dict, gpu_info: dict) -> None:
    assert set(gpu_info) == set(cpu_info)
    for key, cpu_value in cpu_info.items():
        gpu_value = gpu_info[key]
        if isinstance(cpu_value, dict):
            assert isinstance(gpu_value, dict)
            _assert_info_equal(cpu_value, gpu_value)
        else:
            assert gpu_value == cpu_value


@pytest.mark.parametrize("add_latency_info", [False, True])
def test_demon_attack_gpu_reset_info_matches_sample_factory_cpu_reference(add_latency_info):
    if not _cuda_available():
        pytest.skip("CUDA is not available")

    cpu_cfg = _cfg(simulator="cpu", add_latency_info=add_latency_info)
    gpu_cfg = _cfg(simulator="gpu", add_latency_info=add_latency_info)
    cpu_env = _make_cpu_reference_env(cpu_cfg)
    gpu_env = _make_gpu_env(gpu_cfg)

    try:
        cpu_obs, cpu_infos = cpu_env.reset()
        gpu_obs, gpu_infos = gpu_env.reset()

        assert tuple(cpu_obs["obs"].shape) == tuple(gpu_obs["obs"].shape)
        assert cpu_obs["obs"].dtype == gpu_obs["obs"].dtype
        _assert_info_equal(cpu_infos[0], gpu_infos[0])

        gpu_stack = _agent_obs_array(gpu_obs)
        assert np.array_equal(gpu_stack[0], gpu_stack[1])
        assert np.array_equal(gpu_stack[1], gpu_stack[2])
        assert np.array_equal(gpu_stack[2], gpu_stack[3])
    finally:
        cpu_env.close()
        gpu_env.close()


@pytest.mark.parametrize("add_latency_info", [False, True])
def test_demon_attack_gpu_step_info_matches_sample_factory_cpu_reference(add_latency_info):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    cpu_cfg = _cfg(simulator="cpu", add_latency_info=add_latency_info)
    gpu_cfg = _cfg(simulator="gpu", add_latency_info=add_latency_info)
    cpu_env = _make_cpu_reference_env(cpu_cfg)
    gpu_env = _make_gpu_env(gpu_cfg)

    try:
        cpu_env.reset()
        gpu_env.reset()

        cpu_obs, cpu_rewards, cpu_terminated, cpu_truncated, cpu_infos = cpu_env.step([0])
        gpu_obs, gpu_rewards, gpu_terminated, gpu_truncated, gpu_infos = gpu_env.step(
            torch.zeros(1, dtype=torch.int64, device="cuda")
        )

        assert tuple(cpu_obs["obs"].shape) == tuple(gpu_obs["obs"].shape)
        assert _scalar_list(gpu_rewards) == _scalar_list(cpu_rewards)
        assert _scalar_list(gpu_terminated) == _scalar_list(cpu_terminated)
        assert _scalar_list(gpu_truncated) == _scalar_list(cpu_truncated)
        _assert_info_equal(cpu_infos[0], gpu_infos[0])
    finally:
        cpu_env.close()
        gpu_env.close()


@pytest.mark.parametrize("batch_size", [2, 128])
def test_demon_attack_gpu_batched_agents_keep_cpu_reference_step_contract(batch_size):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    cpu_cfg = _cfg(simulator="cpu", add_latency_info=True)
    gpu_cfg = _cfg(simulator="gpu", add_latency_info=True, gpu_env_batch_size=batch_size)
    cpu_env = _make_cpu_reference_env(cpu_cfg)
    gpu_env = _make_gpu_env(gpu_cfg, batch_size=batch_size)

    try:
        cpu_env.reset()
        gpu_env.reset()

        _, cpu_rewards, cpu_terminated, cpu_truncated, cpu_infos = cpu_env.step([0])
        _, gpu_rewards, gpu_terminated, gpu_truncated, gpu_infos = gpu_env.step(
            torch.zeros(batch_size, dtype=torch.int64, device="cuda")
        )

        assert _scalar_list(gpu_rewards) == _scalar_list(cpu_rewards) * batch_size
        assert _scalar_list(gpu_terminated) == _scalar_list(cpu_terminated) * batch_size
        assert _scalar_list(gpu_truncated) == _scalar_list(cpu_truncated) * batch_size
        for gpu_info in gpu_infos:
            _assert_info_equal(cpu_infos[0], gpu_info)
    finally:
        cpu_env.close()
        gpu_env.close()


@pytest.mark.parametrize("add_latency_info", [False, True])
def test_demon_attack_gpu_time_limit_done_info_matches_sample_factory_cpu_reference(add_latency_info):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    cpu_cfg = _cfg(simulator="cpu", add_latency_info=add_latency_info, max_episode_steps=1)
    gpu_cfg = _cfg(simulator="gpu", add_latency_info=add_latency_info, max_episode_steps=1)
    cpu_env = _make_cpu_reference_env(cpu_cfg)
    gpu_env = _make_gpu_env(gpu_cfg)

    try:
        cpu_env.reset()
        gpu_env.reset()

        _, cpu_rewards, cpu_terminated, cpu_truncated, cpu_infos = cpu_env.step([0])
        _, gpu_rewards, gpu_terminated, gpu_truncated, gpu_infos = gpu_env.step(
            torch.zeros(1, dtype=torch.int64, device="cuda")
        )

        assert _scalar_list(gpu_rewards) == _scalar_list(cpu_rewards)
        assert _scalar_list(gpu_terminated) == _scalar_list(cpu_terminated)
        assert _scalar_list(gpu_truncated) == _scalar_list(cpu_truncated)
        _assert_info_equal(cpu_infos[0], gpu_infos[0])
    finally:
        cpu_env.close()
        gpu_env.close()


@pytest.mark.parametrize(
    "latency_overrides",
    [
        {
            "latency_type": "normal_distribution",
            "fixed_latency_ms": None,
            "mean_latency_ms": 50.0,
            "std_latency_ms": 20.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 100.0,
        },
        {
            "latency_type": "uniform_distribution",
            "fixed_latency_ms": None,
            "min_latency_ms": 0.0,
            "max_latency_ms": 100.0,
        },
    ],
)
def test_demon_attack_gpu_sampled_latency_trace_matches_sample_factory_cpu_reference(latency_overrides):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    cpu_cfg = _cfg(simulator="cpu", add_latency_info=True, **latency_overrides)
    gpu_cfg = _cfg(simulator="gpu", add_latency_info=True, **latency_overrides)
    cpu_env = _make_cpu_reference_env(cpu_cfg)
    gpu_env = _make_gpu_env(gpu_cfg)

    try:
        cpu_env.reset()
        gpu_env.reset()
        for decision_idx, action in enumerate([0, 1, 2, 3]):
            _, _, _, _, cpu_infos = cpu_env.step([action])
            _, _, _, _, gpu_infos = gpu_env.step(
                torch.tensor([action], dtype=torch.int64, device="cuda")
            )
            cpu_info = cpu_infos[0]
            gpu_info = gpu_infos[0]
            assert set(gpu_info) == set(cpu_info)
            for info in (cpu_info, gpu_info):
                assert info["episode_idx"] == 1
                assert info["decision_step_idx"] == decision_idx
                assert info["env_step_idx"] == (decision_idx + 1) * 4
                assert info["raw_frame"] == info["env_step_idx"]
                assert info["queued_action"] == action
                assert info["raw_frames_advanced"] == 4
                assert info["obs_stride_raw_frames"] == 4
                assert info["hold_policy"] == "hold"
                assert info["ordering_policy"] == "latest_ready"
                assert info["latency_distribution"] == cpu_info["latency_distribution"]
                assert info["latency_distribution_label"] == cpu_info["latency_distribution_label"]
                assert 0 <= info["latency_raw_frames"] <= 6
                assert info["latency_ms"] == pytest.approx(info["latency_raw_frames"] * (1000.0 / 60.0))
                assert info["ready_raw_frame"] >= decision_idx * 4
                assert info["pending_action_count"] >= 0
    finally:
        cpu_env.close()
        gpu_env.close()


def test_demon_attack_gpu_life_loss_done_is_not_true_episode_end():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    gpu_cfg = _cfg(
        simulator="gpu",
        add_latency_info=True,
        fixed_latency_ms=4 * 1000.0 / 60.0,
    )
    gpu_env = _make_gpu_env(gpu_cfg)
    actions = np.random.default_rng(2026).integers(0, 6, size=700, dtype=np.int64)

    try:
        gpu_env.reset()
        for action in actions:
            _, _, terminated, truncated, infos = gpu_env.step(
                torch.tensor([action], dtype=torch.int64, device="cuda")
            )
            if bool(terminated.item()) or bool(truncated.item()):
                info = infos[0]
                if info["lives"] > 0:
                    assert info["is_true_episode_end"] is False
                    assert "episode_extra_stats" not in info
                    return
        raise AssertionError("expected a life-loss terminal before true game over")
    finally:
        gpu_env.close()


def test_demon_attack_latency_bench_eval_factory_keeps_cpu_adapter_for_gpu_training_config(monkeypatch, tmp_path):
    from latency_bench.core.types import Action
    from latency_bench.envs.factory import build_env_adapter
    from training.common.latency_bench_periodic_eval import build_latency_bench_config_from_training_config

    captured = {}

    class FakeDemonAttackEnvAdapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import latency_bench.envs.demon_attack as demon_attack

    monkeypatch.setattr(demon_attack, "DemonAttackEnvAdapter", FakeDemonAttackEnvAdapter)
    config = build_latency_bench_config_from_training_config(
        {
            "experiment": {"name": "demon_attack_gpu_train", "seed": 0},
            "backend": {"type": "sample_factory", "device": "gpu", "run_mode": "train"},
            "env": {
                "name": "demon_attack",
                "simulator": "gpu",
                "env_fps": 60,
                "obs_fps": 15,
                "frame_stack": 4,
                "terminal_on_life_loss": True,
                "clip_reward": True,
                "screen_size": 84,
                "export_env_raw_rgb_frames": True,
            },
            "latency": {"type": "zero", "add_latency_info": False},
            "scheduler": {"hold_policy": "hold", "ordering_policy": "latest_ready"},
            "training": {"train_for_env_steps": 1000},
            "evaluation": {"eval_episodes": 1},
            "logging": {"output_dir": str(tmp_path / "train_outputs")},
        },
        checkpoint_path=tmp_path / "checkpoint.pth",
        output_dir=tmp_path / "eval_outputs",
    )

    env = build_env_adapter(config, Action(value=0, name="noop", is_noop=True))

    assert isinstance(env, FakeDemonAttackEnvAdapter)
    assert config["backend"]["run_mode"] == "eval"
    assert captured["env_id"] == "ALE/DemonAttack-v5"
    assert captured["screen_size"] == 84
    assert captured["export_env_raw_rgb_frames"] is True

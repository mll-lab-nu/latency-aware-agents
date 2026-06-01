from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from latency_bench.core.config import load_config


ACTION_SEQUENCE = [
    (0, 0, 0, 0),
    (0, 1, 0, 0),
    (1, 0, 0, 0),
    (0, 0, 0, 1),
    (2, 0, 0, 0),
    (0, 2, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 2, 0),
] * 8


def _training_config(
    *,
    add_latency_info=False,
    eval_raw_reward=True,
    train_dir=None,
    experiment_name=None,
):
    from training.common.config_adapter import load_training_config

    config = load_training_config("configs/base/deadly_corridor.yaml")
    config["logging"] = dict(config["logging"])
    config["logging"]["wandb_project"] = ""
    config["training"] = dict(config["training"])
    config["training"]["num_workers"] = 1
    config["training"]["num_envs_per_worker"] = 1
    config["training"]["worker_num_splits"] = 1
    config["env"] = dict(config["env"])
    config["latency"] = dict(config["latency"])
    config["latency"]["add_latency_info"] = bool(add_latency_info)
    config["evaluation"] = dict(config["evaluation"])
    config["evaluation"]["eval_raw_reward"] = bool(eval_raw_reward)
    if train_dir is not None:
        config["backend"] = dict(config["backend"])
        config["backend"]["train_dir"] = str(train_dir)
    if experiment_name is not None:
        config["experiment"] = dict(config["experiment"])
        config["experiment"]["name"] = str(experiment_name)
    return config


def _cfg_from_training_config(config, *, simulator, envpool_batch_size=1):
    from training.deadly_corridor_sf import integration

    cfg = dict(config)
    cfg["env"] = dict(config["env"])
    cfg["env"]["simulator"] = simulator
    if simulator == "envpool":
        cfg["training"] = dict(config["training"])
        cfg["training"]["num_envs_per_worker"] = int(envpool_batch_size)
    return integration.parse_cfg(integration.build_cli_args_from_config(cfg))


def _deadly_spec():
    from sf_examples.vizdoom.doom.doom_utils import DOOM_ENVS

    return next(item for item in DOOM_ENVS if item.name == "doom_deadly_corridor")


def _make_env(*, config, simulator, env_config, envpool_batch_size=1):
    from training.deadly_corridor_sf import integration

    cfg = _cfg_from_training_config(config, simulator=simulator, envpool_batch_size=envpool_batch_size)
    return integration.make_latency_doom_env_from_spec(_deadly_spec(), "doom_deadly_corridor", cfg, env_config)


def _assert_value_equal(actual, expected):
    if isinstance(expected, dict):
        assert set(actual.keys()) == set(expected.keys())
        for key in expected:
            _assert_value_equal(actual[key], expected[key])
        return
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, float) and math.isnan(expected):
        assert isinstance(actual, float)
        assert math.isnan(actual)
        return
    assert actual == expected


def _assert_info_matches_cpu(envpool_info, cpu_info):
    info = dict(envpool_info)
    reset_info = info.pop("reset_info", None)
    assert set(info.keys()) == set(cpu_info.keys())
    for key in cpu_info:
        _assert_value_equal(info[key], cpu_info[key])
    return reset_info


def _read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_deadly_corridor_envpool_config_uses_cpu_batched_sampling():
    from training.deadly_corridor_sf.integration import build_cli_args_from_config

    cfg = load_config("configs/base/deadly_corridor.yaml")
    cfg["env"]["simulator"] = "envpool"
    cfg["env"]["envpool_batch_size"] = 8
    cfg["training"]["num_envs_per_worker"] = 8

    args = build_cli_args_from_config(cfg)

    assert args[args.index("--simulator") + 1] == "envpool"
    assert args[args.index("--envpool-batch-size") + 1] == "8"
    assert args[args.index("--num_envs_per_worker") + 1] == "1"
    assert args[args.index("--worker_num_splits") + 1] == "1"
    assert args[args.index("--batched_sampling") + 1] == "True"
    assert "--env_gpu_observations" not in args
    assert "--env_gpu_actions" not in args


def test_deadly_corridor_envpool_constructs_envpool_without_cpu_fallback(monkeypatch):
    from training.deadly_corridor_sf import integration

    class FakeEnvPoolEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = ModuleType("training.deadly_corridor_sf.envpool_env")
    fake_module.DeadlyCorridorEnvPoolEnv = FakeEnvPoolEnv
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    spec = SimpleNamespace(name="doom_deadly_corridor", env_spec_file="deadly_corridor.cfg", reward_scaling=0.01)
    env_config = SimpleNamespace(worker_index=2, vector_index=3, env_id=5)
    cfg = SimpleNamespace(
        simulator="envpool",
        envpool_batch_size=8,
        env_fps=35,
        obs_fps=8.75,
        frame_stack=4,
        seed=42,
        episode_metrics_path="",
    )

    env = integration.make_latency_doom_env_from_spec(
        spec,
        "doom_deadly_corridor",
        cfg,
        env_config,
    )

    assert isinstance(env, FakeEnvPoolEnv)
    assert env.kwargs["cfg"] is cfg
    assert env.kwargs["env_config"] is env_config
    assert env.kwargs["spec"].name == spec.name
    assert env.kwargs["spec"].env_spec_file == spec.env_spec_file
    assert env.kwargs["spec"].reward_scaling == spec.reward_scaling
    assert env.kwargs["envpool_batch_size"] == 8


def test_deadly_corridor_envpool_integration_applies_eval_raw_reward_scaling(monkeypatch):
    from training.deadly_corridor_sf import integration

    class FakeEnvPoolEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_module = ModuleType("training.deadly_corridor_sf.envpool_env")
    fake_module.DeadlyCorridorEnvPoolEnv = FakeEnvPoolEnv
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    spec = SimpleNamespace(name="doom_deadly_corridor", env_spec_file="deadly_corridor.cfg", reward_scaling=0.01)
    env_config = SimpleNamespace(worker_index=2, vector_index=3, env_id=5)
    cfg = SimpleNamespace(
        simulator="envpool",
        envpool_batch_size=8,
        env_fps=35,
        obs_fps=8.75,
        frame_stack=4,
        seed=42,
        episode_metrics_path="",
        disable_reward_scaling=True,
    )

    env = integration.make_latency_doom_env_from_spec(
        spec,
        "doom_deadly_corridor",
        cfg,
        env_config,
    )

    assert env.kwargs["spec"].reward_scaling == 1.0
    assert spec.reward_scaling == 0.01


def test_deadly_corridor_envpool_tuple_actions_match_sample_factory_button_order():
    from training.deadly_corridor_sf.envpool_env import tuple_action_to_envpool_buttons

    assert tuple_action_to_envpool_buttons((0, 0, 0, 0)).tolist() == [0, 0, 0, 0, 0, 0, 0]
    assert tuple_action_to_envpool_buttons((1, 0, 0, 0)).tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert tuple_action_to_envpool_buttons((2, 0, 0, 0)).tolist() == [0, 1, 0, 0, 0, 0, 0]
    assert tuple_action_to_envpool_buttons((0, 1, 0, 0)).tolist() == [0, 0, 1, 0, 0, 0, 0]
    assert tuple_action_to_envpool_buttons((0, 2, 0, 0)).tolist() == [0, 0, 0, 1, 0, 0, 0]
    assert tuple_action_to_envpool_buttons((0, 0, 1, 0)).tolist() == [0, 0, 0, 0, 1, 0, 0]
    assert tuple_action_to_envpool_buttons((0, 0, 2, 0)).tolist() == [0, 0, 0, 0, 0, 1, 0]
    assert tuple_action_to_envpool_buttons((0, 0, 0, 1)).tolist() == [0, 0, 0, 0, 0, 0, 1]
    assert tuple_action_to_envpool_buttons((1, 1, 2, 1)).tolist() == [1, 0, 1, 0, 0, 1, 1]


def test_deadly_corridor_envpool_passes_cpu_reference_vizdoom_paths_and_native_reward(monkeypatch):
    import vizdoom

    from training.deadly_corridor_sf.envpool_env import DeadlyCorridorEnvPoolEnv

    captured = {}

    fake_envpool = ModuleType("envpool")
    fake_envpool.make_gymnasium = lambda *args, **kwargs: captured.update(task_id=args[0], **kwargs) or object()
    monkeypatch.setitem(sys.modules, "envpool", fake_envpool)

    env = object.__new__(DeadlyCorridorEnvPoolEnv)
    env.num_agents = 8
    env.raw_height = 120
    env.raw_width = 160
    env.spec = SimpleNamespace(env_spec_file="deadly_corridor.cfg")

    env._make_env(seed=123)

    vizdoom_dir = Path(vizdoom.__file__).resolve().parent
    assert captured["task_id"] == "VizdoomCustom-v1"
    assert captured["vzd_path"] == str(vizdoom_dir / "vizdoom")
    assert captured["iwad_path"] == str(vizdoom_dir / "freedoom2.wad")
    assert captured["use_native_reward"] is True
    assert captured["preserve_config_screen_resolution"] is True
    assert captured["reward_config"] == {}
    assert captured["selected_weapon_reward_config"] == {}
    assert captured["use_inter_area_resize"] is False
    assert captured["frame_skip"] == 1
    assert captured["stack_num"] == 1


def test_deadly_corridor_envpool_matches_cpu_reference_for_seed_actions_pixels_and_rewards():
    pytest.importorskip("envpool")

    from sample_factory.utils.attr_dict import AttrDict

    config = _training_config()
    env_config = AttrDict(worker_index=0, vector_index=0, env_id=0)

    cpu_env = _make_env(config=config, simulator="cpu", env_config=env_config)
    envpool_env = _make_env(config=config, simulator="envpool", env_config=env_config)
    try:
        cpu_obs, _cpu_info = cpu_env.reset(seed=123)
        envpool_obs, _envpool_info = envpool_env.reset(seed=123)
        np.testing.assert_array_equal(envpool_obs["obs"][0], cpu_obs)

        for action in ACTION_SEQUENCE:
            cpu_obs, cpu_reward, cpu_terminated, cpu_truncated, cpu_info = cpu_env.step(action)
            envpool_obs, envpool_reward, envpool_terminated, envpool_truncated, envpool_info = envpool_env.step(action)

            np.testing.assert_array_equal(envpool_obs["obs"][0], cpu_obs)
            assert float(envpool_reward[0]) == float(cpu_reward)
            assert float(envpool_info[0]["raw_reward"]) == float(cpu_info["raw_reward"])
            assert bool(envpool_terminated[0]) is bool(cpu_terminated)
            assert bool(envpool_truncated[0]) is bool(cpu_truncated)
            if cpu_terminated or cpu_truncated:
                break
    finally:
        cpu_env.close()
        envpool_env.close()


def test_deadly_corridor_envpool_matches_cpu_reference_with_latency_info_and_reset_info():
    pytest.importorskip("envpool")

    from sample_factory.utils.attr_dict import AttrDict

    config = _training_config(add_latency_info=True)
    env_config = AttrDict(worker_index=0, vector_index=0, env_id=0)

    cpu_env = _make_env(config=config, simulator="cpu", env_config=env_config)
    envpool_env = _make_env(config=config, simulator="envpool", env_config=env_config)
    try:
        cpu_obs, cpu_info = cpu_env.reset(seed=123)
        envpool_obs, envpool_info = envpool_env.reset(seed=123)
        np.testing.assert_array_equal(envpool_obs["obs"][0], cpu_obs)
        _assert_info_matches_cpu(envpool_info[0], cpu_info)

        for action in ACTION_SEQUENCE:
            cpu_obs, cpu_reward, cpu_terminated, cpu_truncated, cpu_info = cpu_env.step(action)
            envpool_obs, envpool_reward, envpool_terminated, envpool_truncated, envpool_info = envpool_env.step(action)

            np.testing.assert_array_equal(envpool_obs["obs"][0], cpu_obs)
            assert float(envpool_reward[0]) == float(cpu_reward)
            assert bool(envpool_terminated[0]) is bool(cpu_terminated)
            assert bool(envpool_truncated[0]) is bool(cpu_truncated)
            reset_info = _assert_info_matches_cpu(envpool_info[0], cpu_info)
            if cpu_terminated or cpu_truncated:
                cpu_reset_obs, cpu_reset_info = cpu_env.reset()
                assert reset_info is not None
                _assert_info_matches_cpu(reset_info, cpu_reset_info)
                for next_action in ACTION_SEQUENCE[:8]:
                    cpu_obs, cpu_reward, cpu_terminated, cpu_truncated, cpu_info = cpu_env.step(next_action)
                    (
                        envpool_obs,
                        envpool_reward,
                        envpool_terminated,
                        envpool_truncated,
                        envpool_info,
                    ) = envpool_env.step(next_action)
                    np.testing.assert_array_equal(envpool_obs["obs"][0], cpu_obs)
                    assert float(envpool_reward[0]) == float(cpu_reward)
                    assert bool(envpool_terminated[0]) is bool(cpu_terminated)
                    assert bool(envpool_truncated[0]) is bool(cpu_truncated)
                    _assert_info_matches_cpu(envpool_info[0], cpu_info)
                break
    finally:
        cpu_env.close()
        envpool_env.close()


def test_deadly_corridor_envpool_matches_cpu_reference_when_rewards_are_scaled():
    pytest.importorskip("envpool")

    from sample_factory.utils.attr_dict import AttrDict

    config = _training_config(eval_raw_reward=False)
    env_config = AttrDict(worker_index=0, vector_index=0, env_id=0)

    cpu_env = _make_env(config=config, simulator="cpu", env_config=env_config)
    envpool_env = _make_env(config=config, simulator="envpool", env_config=env_config)
    try:
        cpu_obs, _cpu_info = cpu_env.reset(seed=123)
        envpool_obs, _envpool_info = envpool_env.reset(seed=123)
        np.testing.assert_array_equal(envpool_obs["obs"][0], cpu_obs)

        for action in ACTION_SEQUENCE:
            cpu_obs, cpu_reward, cpu_terminated, cpu_truncated, cpu_info = cpu_env.step(action)
            envpool_obs, envpool_reward, envpool_terminated, envpool_truncated, envpool_info = envpool_env.step(action)

            np.testing.assert_array_equal(envpool_obs["obs"][0], cpu_obs)
            assert float(envpool_reward[0]) == float(cpu_reward)
            assert float(envpool_info[0]["raw_reward"]) == float(cpu_info["raw_reward"])
            assert float(envpool_reward[0]) == pytest.approx(float(envpool_info[0]["raw_reward"]) * 0.01)
            assert bool(envpool_terminated[0]) is bool(cpu_terminated)
            assert bool(envpool_truncated[0]) is bool(cpu_truncated)
            if cpu_terminated or cpu_truncated:
                break
    finally:
        cpu_env.close()
        envpool_env.close()


def test_deadly_corridor_envpool_batch_agents_match_independent_cpu_references():
    pytest.importorskip("envpool")

    from sample_factory.utils.attr_dict import AttrDict

    batch_size = 3
    config = _training_config()
    envpool_config = AttrDict(worker_index=0, vector_index=0, env_id=0)
    cpu_envs = [
        _make_env(
            config=config,
            simulator="cpu",
            env_config=AttrDict(worker_index=0, vector_index=0, env_id=agent_idx),
        )
        for agent_idx in range(batch_size)
    ]
    envpool_env = _make_env(
        config=config,
        simulator="envpool",
        env_config=envpool_config,
        envpool_batch_size=batch_size,
    )
    try:
        envpool_obs, _envpool_info = envpool_env.reset(seed=321)
        for agent_idx, cpu_env in enumerate(cpu_envs):
            cpu_obs, _cpu_info = cpu_env.reset(seed=321)
            np.testing.assert_array_equal(envpool_obs["obs"][agent_idx], cpu_obs)

        for step_idx in range(12):
            actions = [ACTION_SEQUENCE[(step_idx + agent_idx) % len(ACTION_SEQUENCE)] for agent_idx in range(batch_size)]
            envpool_obs, envpool_reward, envpool_terminated, envpool_truncated, envpool_info = envpool_env.step(actions)
            for agent_idx, cpu_env in enumerate(cpu_envs):
                cpu_obs, cpu_reward, cpu_terminated, cpu_truncated, cpu_info = cpu_env.step(actions[agent_idx])
                np.testing.assert_array_equal(envpool_obs["obs"][agent_idx], cpu_obs)
                assert float(envpool_reward[agent_idx]) == float(cpu_reward)
                assert float(envpool_info[agent_idx]["raw_reward"]) == float(cpu_info["raw_reward"])
                assert bool(envpool_terminated[agent_idx]) is bool(cpu_terminated)
                assert bool(envpool_truncated[agent_idx]) is bool(cpu_truncated)
    finally:
        for cpu_env in cpu_envs:
            cpu_env.close()
        envpool_env.close()


def test_deadly_corridor_envpool_action_input_forms_match_tuple_batch():
    from training.deadly_corridor_sf.envpool_env import DeadlyCorridorEnvPoolEnv

    env = object.__new__(DeadlyCorridorEnvPoolEnv)
    env.num_agents = 1

    assert env._normalize_actions(np.asarray([1, 2, 0, 1], dtype=np.int64)) == [(1, 2, 0, 1)]

    env.num_agents = 2
    assert env._normalize_actions(np.asarray([[1, 2, 0, 1], [0, 1, 2, 0]], dtype=np.int64)) == [
        (1, 2, 0, 1),
        (0, 1, 2, 0),
    ]
    assert env._normalize_actions(
        (
            np.asarray([1, 0], dtype=np.int64),
            np.asarray([2, 1], dtype=np.int64),
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([1, 0], dtype=np.int64),
        )
    ) == [(1, 2, 0, 1), (0, 1, 2, 0)]


def test_deadly_corridor_envpool_episode_metrics_match_cpu_reference(tmp_path):
    pytest.importorskip("envpool")

    from sample_factory.utils.attr_dict import AttrDict

    cpu_config = _training_config(train_dir=tmp_path, experiment_name="cpu")
    envpool_config = _training_config(train_dir=tmp_path, experiment_name="envpool")
    cpu_metrics_path = tmp_path / "cpu" / "episode_metrics.jsonl"
    envpool_metrics_path = tmp_path / "envpool" / "episode_metrics.jsonl"
    env_config = AttrDict(worker_index=0, vector_index=0, env_id=0)

    cpu_env = _make_env(config=cpu_config, simulator="cpu", env_config=env_config)
    envpool_env = _make_env(config=envpool_config, simulator="envpool", env_config=env_config)
    try:
        cpu_env.reset(seed=123)
        envpool_env.reset(seed=123)
        for action in ACTION_SEQUENCE:
            _cpu_obs, _cpu_reward, cpu_terminated, cpu_truncated, _cpu_info = cpu_env.step(action)
            (
                _envpool_obs,
                _envpool_reward,
                envpool_terminated,
                envpool_truncated,
                _envpool_info,
            ) = envpool_env.step(action)
            assert bool(envpool_terminated[0]) is bool(cpu_terminated)
            assert bool(envpool_truncated[0]) is bool(cpu_truncated)
            if cpu_terminated or cpu_truncated:
                break
    finally:
        cpu_env.close()
        envpool_env.close()

    cpu_events = _read_jsonl(cpu_metrics_path)
    envpool_events = _read_jsonl(envpool_metrics_path)
    assert envpool_events == cpu_events

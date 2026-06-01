from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


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

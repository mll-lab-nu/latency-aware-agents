from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from latency_bench.core.config import load_config
from latency_bench.core.types import Action
from latency_bench.envs.gymnasium import GymnasiumEnvAdapter
from scripts.scaffold_humanoidbench import (
    ENV_ID,
    NATIVE_ENV_ID,
    actuator_labels,
    build_env_config,
    build_probe_report,
    nominal_stance_action,
)
from training.gymnasium_sf.integration import build_train_cfg_from_config, make_reference_env


CONFIG_DIR = Path("configs/examples/gymnasium/h1hand_balance_hard")


def test_humanoidbench_runtime_contract_and_nominal_stance():
    import latency_bench.envs.humanoidbench  # noqa: F401

    env = gym.make(ENV_ID, render_mode="rgb_array", width=256, height=256)
    try:
        native_env = env.unwrapped
        observation, reset_info = env.reset(seed=0)
        action = nominal_stance_action(env)
        next_observation, _, terminated, truncated, _ = env.step(action)
        rgb = env.render()

        assert env.spec.max_episode_steps == 1000
        assert env.action_space == gym.spaces.Box(-1.0, 1.0, shape=(61,), dtype=np.float32)
        assert env.observation_space == gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(177,),
            dtype=np.float64,
        )
        assert native_env.model.opt.timestep == 0.002
        assert native_env.frame_skip == 10
        assert native_env.dt == 0.02
        assert len(actuator_labels(env)) == 61
        assert actuator_labels(env)[0] == "left_hip_yaw"
        assert actuator_labels(env)[-1] == "rh_A_LFJ0"
        assert env.action_space.contains(action)
        assert not np.array_equal(action, np.zeros(61, dtype=np.float32))
        assert observation.shape == (177,)
        assert observation.dtype == np.float64
        assert next_observation.shape == (177,)
        assert next_observation.dtype == np.float64
        assert reset_info == {}
        assert terminated is False
        assert truncated is False
        assert rgb.shape == (256, 256, 3)
        assert rgb.dtype == np.uint8
    finally:
        env.close()


def test_humanoidbench_wrapper_converts_policy_list_to_native_array():
    import latency_bench.envs.humanoidbench  # noqa: F401

    env = gym.make(ENV_ID, render_mode="rgb_array", width=256, height=256)
    try:
        env.reset(seed=0)
        action = nominal_stance_action(env).tolist()
        observation, _, terminated, truncated, _ = env.step(action)

        assert observation.shape == (177,)
        assert terminated is False
        assert truncated is False
    finally:
        env.close()


def test_humanoidbench_probe_emits_runtime_contract():
    report = build_probe_report()
    env_config = build_env_config(report)

    assert report["action_space"]["shape"] == [61]
    assert report["observation_space"] == {
        "type": "Box",
        "shape": [177],
        "dtype": "float64",
    }
    assert report["render"] == {"shape": [256, 256, 3], "dtype": "uint8"}
    assert report["timing"] == {
        "dt_seconds": 0.02,
        "frame_skip": 10,
        "model_timestep_seconds": 0.002,
        "native_env_fps": 50.0,
    }
    assert len(env_config["action_space"]["labels"]) == 61
    assert len(env_config["noop_action"]) == 61
    assert env_config["registration_imports"] == ["latency_bench.envs.humanoidbench"]
    assert load_config(CONFIG_DIR / "small_model_train.yaml")["env"] == env_config


def test_humanoidbench_seeded_reset_and_zero_latency_adapter_parity():
    config = load_config(CONFIG_DIR / "small_model_train.yaml")
    env_config = config["env"]
    import latency_bench.envs.humanoidbench  # noqa: F401

    native_env = gym.make(NATIVE_ENV_ID, **env_config["make_kwargs"])
    adapter = GymnasiumEnvAdapter(
        env_id=env_config["env_id"],
        registration_imports=env_config["registration_imports"],
        make_kwargs=env_config["make_kwargs"],
        env_fps=env_config["env_fps"],
        noop_action=Action(value=env_config["noop_action"], name="noop", is_noop=True),
        frame_stack=env_config["frame_stack"],
        export_env_raw_rgb_frames=False,
    )
    try:
        native_observation, _ = native_env.reset(seed=7)
        adapter_observation = adapter.reset(seed=7)
        np.testing.assert_array_equal(adapter_observation.data, native_observation)

        action = np.asarray(env_config["noop_action"], dtype=np.float32)
        native_next_observation, native_reward, native_terminated, native_truncated, _ = (
            native_env.step(action)
        )
        adapter_result = adapter.step(Action(value=action.tolist(), name="nominal_stance"))
        np.testing.assert_array_equal(adapter_result.observation.data, native_next_observation)
        assert adapter_result.reward == native_reward
        assert adapter_result.done is native_terminated
        assert adapter_result.truncated is native_truncated
    finally:
        native_env.close()
        adapter.close()


def test_humanoidbench_configs_and_sample_factory_keep_box_61_contract():
    config_paths = sorted(CONFIG_DIR.glob("*.yaml"))
    configs = [load_config(path) for path in config_paths]
    env_config = configs[0]["env"]

    assert len(config_paths) == 4
    assert all(config["env"] == env_config for config in configs[1:])
    assert env_config["env_fps"] == 50.0
    assert env_config["obs_fps"] == 50.0
    assert env_config["frame_stack"] == 1
    assert env_config["action_space"]["type"] == "box"
    assert len(env_config["action_space"]["labels"]) == 61
    assert len(env_config["noop_action"]) == 61

    train_config = load_config(CONFIG_DIR / "small_model_train.yaml")
    cfg = build_train_cfg_from_config(train_config)
    reference_env = make_reference_env(cfg)
    try:
        assert reference_env.action_space == gym.spaces.Box(
            -1.0,
            1.0,
            shape=(61,),
            dtype=np.float32,
        )
        assert reference_env.observation_space["obs"].shape == (177,)
    finally:
        reference_env.close()

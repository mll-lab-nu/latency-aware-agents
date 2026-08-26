from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from latency_bench.core.config import load_config
from latency_bench.core.types import Action
from latency_bench.envs.gymnasium import GymnasiumEnvAdapter
from latency_bench.envs.humanoidbench import HUMANOIDBENCH_ENV_SPECS
from scripts.scaffold_humanoidbench import actuator_labels, build_noop_action
from training.gymnasium_sf.integration import build_train_cfg_from_config, make_reference_env


CONFIG_DIR = Path("configs/examples/gymnasium/h1hand_balance_hard")
EXPECTED_ACTUATOR_LABELS = load_config(CONFIG_DIR / "small_model_train.yaml")["env"][
    "action_space"
]["labels"]
TASK_CONTRACTS = (
    (
        "h1hand_balance_hard",
        "LatencyBench/H1HandBalanceHard-v0",
        "h1hand-balance_hard-v0",
        (177,),
        1000,
    ),
    (
        "h1hand_hurdle",
        "LatencyBench/H1HandHurdle-v0",
        "h1hand-hurdle-v0",
        (151,),
        1000,
    ),
    (
        "h1hand_run",
        "LatencyBench/H1HandRun-v0",
        "h1hand-run-v0",
        (151,),
        1000,
    ),
    (
        "h1hand_basketball",
        "LatencyBench/H1HandBasketball-v0",
        "h1hand-basketball-v0",
        (164,),
        500,
    ),
    (
        "h1strong_highbar_hard",
        "LatencyBench/H1StrongHighBarHard-v0",
        "h1strong-highbar_hard-v0",
        (151,),
        1000,
    ),
)


@pytest.mark.parametrize(
    ("task_name", "env_id", "native_env_id", "observation_shape", "horizon"),
    TASK_CONTRACTS,
    ids=[contract[0] for contract in TASK_CONTRACTS],
)
def test_humanoidbench_family_runtime_contract(
    task_name,
    env_id,
    native_env_id,
    observation_shape,
    horizon,
):
    native_spec = gym.spec(native_env_id)
    registered_spec = gym.spec(env_id)
    env_spec = HUMANOIDBENCH_ENV_SPECS[task_name]
    env = gym.make(env_id, render_mode="rgb_array", width=256, height=256)
    try:
        native_env = env.unwrapped
        observation, _ = env.reset(seed=0)
        action = build_noop_action(task_name, env)
        next_observation, _, _, _, _ = env.step(action.tolist())
        rgb = env.render()

        assert env_spec == {
            "env_id": env_id,
            "native_env_id": native_env_id,
            "observation_shape": observation_shape,
            "max_episode_steps": native_spec.max_episode_steps,
        }
        assert native_spec.max_episode_steps == horizon
        assert registered_spec.max_episode_steps == horizon
        assert registered_spec.kwargs == {"native_env_id": native_env_id}
        assert env.action_space == gym.spaces.Box(
            -1.0,
            1.0,
            shape=(61,),
            dtype=np.float32,
        )
        assert actuator_labels(env) == EXPECTED_ACTUATOR_LABELS
        assert env.observation_space == gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=observation_shape,
            dtype=np.float64,
        )
        assert native_env.model.opt.timestep == 0.002
        assert native_env.frame_skip == 10
        assert native_env.dt == 0.02
        assert 1.0 / native_env.dt == 50.0
        assert env.action_space.contains(action)
        assert observation.shape == observation_shape
        assert observation.dtype == np.float64
        assert next_observation.shape == observation_shape
        assert next_observation.dtype == np.float64
        assert rgb.shape == (256, 256, 3)
        assert rgb.dtype == np.uint8
    finally:
        env.close()


def test_humanoidbench_basketball_repeated_seeded_reset_is_exact():
    env_spec = HUMANOIDBENCH_ENV_SPECS["h1hand_basketball"]
    env = gym.make(env_spec["env_id"], render_mode="rgb_array", width=256, height=256)
    try:
        first_observation, _ = env.reset(seed=17)
        second_observation, _ = env.reset(seed=17)

        np.testing.assert_array_equal(second_observation, first_observation)
    finally:
        env.close()


def test_humanoidbench_highbar_constrains_only_thumb_keyframe_controls():
    task_name = "h1strong_highbar_hard"
    env_spec = HUMANOIDBENCH_ENV_SPECS[task_name]
    env = gym.make(env_spec["env_id"], render_mode="rgb_array", width=256, height=256)
    try:
        native_env = env.unwrapped
        labels = actuator_labels(env)
        thumb_ids = [labels.index(name) for name in ("lh_A_THJ5", "rh_A_THJ5")]
        keyframe_control = native_env.model.key_ctrl[native_env.keyframe].copy()
        constrained_control = keyframe_control.copy()
        constrained_control[thumb_ids] = np.float32(1.0472)

        action = build_noop_action(task_name, env)
        expected_action = (
            2.0
            * (
                constrained_control.astype(native_env.action_low.dtype)
                - native_env.action_low
            )
            / (native_env.action_high - native_env.action_low)
            - 1.0
        ).astype(env.action_space.dtype)
        changed_ids = np.flatnonzero(constrained_control != keyframe_control).tolist()

        assert keyframe_control[thumb_ids].tolist() == [1.05, 1.05]
        np.testing.assert_array_equal(
            native_env.action_high[thumb_ids],
            np.full(2, 1.0472, dtype=np.float32),
        )
        assert changed_ids == thumb_ids
        np.testing.assert_array_equal(action, expected_action)
        assert env.action_space.contains(action)

        env.reset(seed=0)
        env.step(action)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["g1-reach-v0", "g1-push-v0"])
def test_humanoidbench_mujoco_33_task_markers_render(env_id):
    env = gym.make(env_id, render_mode="rgb_array", width=32, height=32)
    try:
        env.reset(seed=0)
        assert env.render().shape == (32, 32, 3)
    finally:
        env.close()


def test_humanoidbench_seeded_reset_and_zero_latency_adapter_parity():
    config = load_config(CONFIG_DIR / "small_model_train.yaml")
    env_config = config["env"]
    native_env_id = HUMANOIDBENCH_ENV_SPECS["h1hand_balance_hard"]["native_env_id"]

    native_env = gym.make(native_env_id, **env_config["make_kwargs"])
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
    env_config = load_config(CONFIG_DIR / "small_model_train.yaml")["env"]

    for name in ("small_model_eval.yaml", "latency_bench_eval.yaml"):
        assert load_config(CONFIG_DIR / name)["env"] == env_config
    for name in (
        "small_model_train_25hz.yaml",
        "small_model_eval_25hz.yaml",
        "starvla_eval.yaml",
    ):
        assert load_config(CONFIG_DIR / name)["env"] == {**env_config, "obs_fps": 25.0}
    assert env_config["env_fps"] == 50.0
    assert env_config["obs_fps"] == 50.0
    assert env_config["frame_stack"] == 1
    assert env_config["state_space"]["labels"] == [
        f"state_{index:03d}" for index in range(177)
    ]
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

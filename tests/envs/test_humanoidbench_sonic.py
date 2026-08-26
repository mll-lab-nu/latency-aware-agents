from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import mujoco
import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from latency_bench.envs import humanoidbench_sonic as sonic


class _DecoderSession:
    def __init__(self, *args, **kwargs):
        self.input = None

    def get_inputs(self):
        return [SimpleNamespace(name="obs_dict")]

    def get_outputs(self):
        return [SimpleNamespace(name="action")]

    def run(self, output_names, inputs):
        self.input = inputs["obs_dict"]
        return [np.zeros((1, 29), dtype=np.float32)]


@pytest.fixture(scope="module")
def scene_root():
    if "LATENCY_BENCH_SONIC_SCENE_ROOT" not in os.environ:
        pytest.skip(
            "set LATENCY_BENCH_SONIC_SCENE_ROOT to the corrected SONIC scene directory"
        )
    return Path(os.environ["LATENCY_BENCH_SONIC_SCENE_ROOT"])


@pytest.fixture(scope="module")
def scene_path(scene_root):
    path = scene_root / sonic.G1_SONIC_ENV_SPECS["g1_sonic_run"][
        "scene_filename"
    ].removeprefix("sonic_g1/")
    mujoco.MjModel.from_xml_path(str(path))
    return path


@pytest.fixture(scope="module")
def hurdle_scene_path(scene_root):
    path = scene_root / sonic.G1_SONIC_ENV_SPECS["g1_sonic_hurdle"][
        "scene_filename"
    ].removeprefix("sonic_g1/")
    mujoco.MjModel.from_xml_path(str(path))
    return path


@pytest.fixture(scope="module")
def scene_paths(scene_root):
    paths = {
        task_name: scene_root / spec["scene_filename"].removeprefix("sonic_g1/")
        for task_name, spec in sonic.G1_SONIC_ENV_SPECS.items()
    }
    return paths


@pytest.fixture
def env(monkeypatch, scene_path):
    monkeypatch.setattr(sonic.ort, "InferenceSession", _DecoderSession)
    instance = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_run"]["env_id"],
        scene_path=str(scene_path),
        sonic_decoder_path="decoder.onnx",
        sonic_decoder_mode=sonic.SONIC_DECODER_MODE_ENVIRONMENT,
        render_mode="rgb_array",
        width=64,
        height=64,
    )
    yield instance
    instance.close()


def test_sonic_run_environment_contract(env):
    first_observation, _ = env.reset(seed=17)
    second_observation, _ = env.reset(seed=17)
    np.testing.assert_array_equal(second_observation, first_observation)

    native = env.unwrapped
    assert env.action_space == gym.spaces.Box(-1, 1, shape=(78,), dtype=np.float32)
    assert env.observation_space == gym.spaces.Box(
        -np.inf, np.inf, shape=(99,), dtype=np.float64
    )
    assert native.model.nq == 50
    assert native.model.nv == 49
    assert native.model.nu == 43
    assert native.model.opt.timestep == 0.002
    assert native.frame_skip == 10
    assert native.dt == 0.02
    assert (
        gym.spec(sonic.G1_SONIC_ENV_SPECS["g1_sonic_run"]["env_id"])
        .max_episode_steps
        == 1000
    )
    assert native.robot.dof == 50

    next_observation, reward, terminated, truncated, info = env.step(
        np.zeros(78, dtype=np.float32)
    )
    assert next_observation.shape == (99,)
    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert info["per_timestep_reward"] == reward
    assert env.render().shape == (64, 64, 3)
    assert native.camera_name == "cam_default"

    qpos = native.data.qpos.copy()
    qvel = native.data.qvel.copy()
    qvel[0] = 5.0
    native.set_state(qpos, qvel)
    *_, moving_info = env.step(np.zeros(78, dtype=np.float32))
    assert moving_info["move"] > 0.99


def test_sonic_decoder_history_and_joint_mapping(env):
    env.reset(seed=0)
    native = env.unwrapped
    mujoco.mj_resetDataKeyframe(native.model, native.data, 0)
    keyframe_body_q = native.data.qpos[native._body_qpos_addresses].copy()
    np.testing.assert_array_equal(
        native.data.qpos[native._hand_qpos_addresses], np.zeros(14)
    )
    native._history.clear()
    native._history.append(native._frame())

    body_q, body_dq = native._body_state()
    np.testing.assert_array_equal(
        body_q,
        (keyframe_body_q - sonic._DEFAULT_ANGLES)[sonic._MUJOCO_INDEX_FOR_POLICY],
    )
    np.testing.assert_array_equal(body_dq, np.zeros(29))

    token = np.arange(64, dtype=np.float32)
    decoder_input = native._decoder_input(token)
    assert decoder_input.shape == (994,)
    np.testing.assert_array_equal(decoder_input[:64], token)
    np.testing.assert_array_equal(decoder_input[64 : 64 + 9 * 3], np.zeros(27))
    np.testing.assert_array_equal(
        decoder_input[64 + 30 : 64 + 30 + 9 * 29], np.zeros(261)
    )

    policy_action = np.arange(29, dtype=np.float64) / 100
    hand_action = np.zeros(14, dtype=np.float64)
    control = native._controller(policy_action, hand_action)
    expected_target = sonic._DEFAULT_ANGLES + policy_action * sonic._ACTION_SCALE
    expected_body_control = sonic._KP * (expected_target - keyframe_body_q)
    np.testing.assert_allclose(
        control[native._body_actuator_ids],
        expected_body_control,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        control[native._hand_actuator_ids],
        sonic._HAND_KP
        * (
            native._hand_joint_ranges[:, 0]
            + (hand_action + 1.0)
            * 0.5
            * (native._hand_joint_ranges[:, 1] - native._hand_joint_ranges[:, 0])
        ),
    )

    raw_action = np.linspace(-1.0, 1.0, 29)
    raw_control = native._controller(raw_action, hand_action, raw_body=True)
    raw_target = native._body_joint_ranges[:, 0] + (raw_action + 1.0) * 0.5 * (
        native._body_joint_ranges[:, 1] - native._body_joint_ranges[:, 0]
    )
    np.testing.assert_allclose(
        raw_control[native._body_actuator_ids],
        np.clip(
            sonic._KP * (raw_target - native.data.qpos[native._body_qpos_addresses]),
            native.model.actuator_ctrlrange[native._body_actuator_ids, 0],
            native.model.actuator_ctrlrange[native._body_actuator_ids, 1],
        ),
    )


def test_inference_worker_mode_exposes_decoder_state_and_consumes_body_action(
    scene_path,
):
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_run"]["env_id"],
        scene_path=str(scene_path),
        sonic_decoder_path="unused.onnx",
        sonic_decoder_mode=sonic.SONIC_DECODER_MODE_INFERENCE_WORKER,
        render_mode="rgb_array",
        width=64,
        height=64,
    )
    try:
        observation, _ = env.reset(seed=0)
        assert observation["obs"].shape == (99,)
        assert observation["sonic_state"].shape == (930,)
        np.testing.assert_array_equal(
            observation["sonic_state"], env.unwrapped._decoder_state()
        )

        body_action = np.arange(29, dtype=np.float32) / 100
        env.unwrapped.set_decoded_action(body_action)
        next_observation, *_ = env.step(np.zeros(78, dtype=np.float32))
        np.testing.assert_array_equal(env.unwrapped._last_action, body_action)
        assert next_observation["sonic_state"].shape == (930,)
    finally:
        env.close()


def test_hurdle_residual_uses_reference_tokens_and_sonic_reset(
    monkeypatch, tmp_path, hurdle_scene_path
):
    tokens = np.arange(3 * 64, dtype=np.float32).reshape(3, 64) / 64
    token_path = tmp_path / "run_tokens.npy"
    np.save(token_path, tokens)
    monkeypatch.setattr(sonic.ort, "InferenceSession", _DecoderSession)
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_hurdle"]["env_id"],
        scene_path=str(hurdle_scene_path),
        control_mode=sonic.G1_CONTROL_MODE_SONIC_RESIDUAL,
        sonic_decoder_path="decoder.onnx",
        sonic_decoder_mode=sonic.SONIC_DECODER_MODE_ENVIRONMENT,
        sonic_base_token_path=str(token_path),
        sonic_residual_scale=1.0 / 16.0,
        render_mode="rgb_array",
        width=32,
        height=32,
    )
    try:
        observation, _ = env.reset(seed=7)
        native = env.unwrapped
        assert env.action_space.shape == (64,)
        assert observation["obs"].shape == (99 + 930 + 64,)
        np.testing.assert_array_equal(observation["base_token"], tokens[0])
        np.testing.assert_array_equal(observation["obs"][-64:], tokens[0])
        np.testing.assert_array_equal(
            native.data.qpos[native._body_qpos_addresses], sonic._DEFAULT_ANGLES
        )
        np.testing.assert_array_equal(
            native.data.qpos[native._hand_qpos_addresses], np.zeros(14)
        )
        assert native.data.qpos[2] == 0.76

        residual = np.ones(64, dtype=np.float32)
        next_observation, *_ = env.step(residual)
        np.testing.assert_allclose(
            native._decoder.input[0, :64], tokens[0] + residual / 16.0
        )
        np.testing.assert_array_equal(next_observation["base_token"], tokens[1])
        assert native._base_token_cursor == 1

        native.data.qpos[native._hand_qpos_addresses] = 0.1
        native.data.qvel[native._hand_qvel_addresses] = 0.0
        control = native._controller(np.zeros(29), None)
        np.testing.assert_allclose(
            control[native._hand_actuator_ids], -sonic._HAND_KP * 0.1
        )

        reset_observation, _ = env.reset(seed=11)
        np.testing.assert_array_equal(reset_observation["base_token"], tokens[0])
        assert native._base_token_cursor == 0
        np.testing.assert_array_equal(native._last_action, np.zeros(29))
    finally:
        env.close()


def test_hurdle_residual_inference_worker_contract(tmp_path, hurdle_scene_path):
    tokens = np.zeros((2, 64), dtype=np.float32)
    token_path = tmp_path / "run_tokens.npy"
    np.save(token_path, tokens)
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_hurdle"]["env_id"],
        scene_path=str(hurdle_scene_path),
        control_mode=sonic.G1_CONTROL_MODE_SONIC_RESIDUAL,
        sonic_decoder_path="unused.onnx",
        sonic_decoder_mode=sonic.SONIC_DECODER_MODE_INFERENCE_WORKER,
        sonic_base_token_path=str(token_path),
        sonic_residual_scale=1.0 / 16.0,
        render_mode="rgb_array",
        width=32,
        height=32,
    )
    try:
        observation, _ = env.reset(seed=0)
        assert observation["obs"].shape == (99 + 930 + 64,)
        assert observation["sonic_state"].shape == (930,)
        assert observation["base_token"].shape == (64,)
        env.unwrapped.set_decoded_action(np.zeros(29, dtype=np.float32))
        next_observation, *_ = env.step(np.zeros(64, dtype=np.float32))
        np.testing.assert_array_equal(next_observation["base_token"], tokens[1])
        assert env.unwrapped._base_token_cursor == 1
    finally:
        env.close()


@pytest.mark.parametrize(
    ("task_name", "expected_obs_dim", "horizon"),
    [
        ("g1_sonic_balance_hard", 125, 1000),
        ("g1_sonic_hurdle", 99, 1000),
        ("g1_sonic_run", 99, 1000),
        ("g1_sonic_basketball", 112, 500),
    ],
)
def test_g1_task_family_raw_and_sonic_contract(
    scene_paths, task_name, expected_obs_dim, horizon
):
    spec = sonic.G1_SONIC_ENV_SPECS[task_name]
    raw = gym.make(
        spec["env_id"],
        task_name=task_name,
        scene_path=str(scene_paths[task_name]),
        control_mode=sonic.G1_CONTROL_MODE_RAW,
        sonic_decoder_path="unused.onnx",
        sonic_decoder_mode=sonic.SONIC_DECODER_MODE_ENVIRONMENT,
        render_mode="rgb_array",
        width=32,
        height=32,
    )
    try:
        observation, _ = raw.reset(seed=7)
        if task_name == "g1_sonic_basketball":
            first, _ = raw.reset(seed=23)
            second, _ = raw.reset(seed=23)
            np.testing.assert_array_equal(first, second)
            observation, _ = raw.reset(seed=7)
        assert observation.shape == (expected_obs_dim,)
        assert raw.action_space.shape == (43,)
        assert raw.unwrapped.model.opt.timestep == 0.002
        assert raw.unwrapped.frame_skip == 10
        assert raw.unwrapped.dt == 0.02
        task_class = sonic.TASKS[task_name.removeprefix("g1_sonic_")]
        assert isinstance(raw.unwrapped._task, task_class)
        assert raw.unwrapped.camera_name == task_class.camera_name
        assert horizon == task_class.max_episode_steps
        assert gym.spec(spec["env_id"]).max_episode_steps == horizon
        next_observation, reward, _terminated, truncated, _ = raw.step(
            np.zeros(43, dtype=np.float32)
        )
        assert next_observation.shape == (expected_obs_dim,)
        assert np.isfinite(reward)
        assert truncated is False
        assert raw.render().shape == (32, 32, 3)
    finally:
        raw.close()

    sonic_env = gym.make(
        spec["env_id"],
        task_name=task_name,
        scene_path=str(scene_paths[task_name]),
        control_mode=sonic.G1_CONTROL_MODE_SONIC,
        sonic_decoder_path="unused.onnx",
        sonic_decoder_mode=sonic.SONIC_DECODER_MODE_INFERENCE_WORKER,
        render_mode="rgb_array",
        width=32,
        height=32,
    )
    try:
        observation, _ = sonic_env.reset(seed=7)
        assert observation["obs"].shape == (expected_obs_dim,)
        assert observation["sonic_state"].shape == (930,)
        assert sonic_env.action_space.shape == (78,)
        sonic_env.unwrapped.set_decoded_action(np.zeros(29, dtype=np.float32))
        next_observation, reward, _terminated, truncated, _ = sonic_env.step(
            np.zeros(78, dtype=np.float32)
        )
        assert next_observation["obs"].shape == (expected_obs_dim,)
        assert next_observation["sonic_state"].shape == (930,)
        assert np.isfinite(reward)
        assert truncated is False
        assert sonic_env.render().shape == (32, 32, 3)
    finally:
        sonic_env.close()


def test_g1_sonic_truck_package_density(scene_paths):
    model = mujoco.MjModel.from_xml_path(str(scene_paths["g1_sonic_truck"]))
    package_names = (
        "package_a",
        "package_b",
        "package_c",
        "package_d",
        "package_e",
    )
    masses = np.array(
        [model.body_mass[model.body(package).id] for package in package_names]
    )
    np.testing.assert_allclose(masses, [0.24, 0.40, 0.48, 0.60, 0.64])


def test_g1_sonic_registry_covers_the_supported_task_family():
    assert len(sonic.G1_SONIC_ENV_SPECS) == len(sonic.TASKS) - 2 == 30
    for unsupported in sonic.G1_SONIC_UNSUPPORTED_TASKS:
        assert f"g1_sonic_{unsupported}" not in sonic.G1_SONIC_ENV_SPECS
    for task_key, task_class in sonic.TASKS.items():
        if task_key in sonic.G1_SONIC_UNSUPPORTED_TASKS:
            continue
        spec = sonic.G1_SONIC_ENV_SPECS[f"g1_sonic_{task_key}"]
        assert spec["scene_filename"] == f"sonic_g1/envs/scene_{task_key}.xml"
        assert spec["horizon"] == task_class.max_episode_steps
        assert gym.spec(spec["env_id"]).max_episode_steps == task_class.max_episode_steps


_INITIAL_TRAINING_TASKS = {
    "g1_sonic_balance_hard",
    "g1_sonic_hurdle",
    "g1_sonic_run",
    "g1_sonic_basketball",
}


@pytest.mark.parametrize(
    "task_name",
    sorted(set(sonic.G1_SONIC_ENV_SPECS) - _INITIAL_TRAINING_TASKS),
)
@pytest.mark.parametrize(
    ("control_mode", "decoder_mode", "action_dim"),
    [
        (
            sonic.G1_CONTROL_MODE_RAW,
            sonic.SONIC_DECODER_MODE_ENVIRONMENT,
            43,
        ),
        (
            sonic.G1_CONTROL_MODE_SONIC,
            sonic.SONIC_DECODER_MODE_INFERENCE_WORKER,
            78,
        ),
    ],
)
def test_extended_g1_task_family_reset_step_and_render(
    scene_paths, task_name, control_mode, decoder_mode, action_dim
):
    spec = sonic.G1_SONIC_ENV_SPECS[task_name]
    env = gym.make(
        spec["env_id"],
        task_name=task_name,
        scene_path=str(scene_paths[task_name]),
        control_mode=control_mode,
        sonic_decoder_path="unused.onnx",
        sonic_decoder_mode=decoder_mode,
        render_mode="rgb_array",
        width=16,
        height=16,
    )
    try:
        first, _ = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        if decoder_mode == sonic.SONIC_DECODER_MODE_INFERENCE_WORKER:
            np.testing.assert_array_equal(first["obs"], second["obs"])
            env.unwrapped.set_decoded_action(np.zeros(29, dtype=np.float32))
        else:
            np.testing.assert_array_equal(first, second)
        observation, reward, _terminated, truncated, _info = env.step(
            np.zeros(action_dim, dtype=np.float32)
        )
        policy_observation = (
            observation["obs"]
            if decoder_mode == sonic.SONIC_DECODER_MODE_INFERENCE_WORKER
            else observation
        )
        assert policy_observation.shape == env.unwrapped._task.observation_space.shape
        assert np.isfinite(reward)
        assert truncated is False
        assert env.unwrapped.robot.dof == 50
        assert env.unwrapped.model.nq >= 50
        assert env.unwrapped.model.nv >= 49
        assert env.unwrapped.model.nu == 43
        assert env.render().shape == (16, 16, 3)
    finally:
        env.close()

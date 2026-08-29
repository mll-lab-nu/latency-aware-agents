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
from latency_bench.envs.humanoidbench_sonic_contract import (
    G1_DEFAULT_ANGLES,
    MUJOCO_INDEX_FOR_POLICY,
    POLICY_INDEX_FOR_MUJOCO,
)


class _DecoderSession:
    def __init__(self, *args, **kwargs):
        self.input = None
        self.path = args[0]

    def get_inputs(self):
        return [SimpleNamespace(name="obs_dict")]

    def get_outputs(self):
        return [
            SimpleNamespace(
                name="encoded_tokens" if "encoder" in self.path else "action"
            )
        ]

    def run(self, output_names, inputs):
        self.input = inputs["obs_dict"]
        width = 64 if "encoder" in self.path else 29
        return [np.zeros((1, width), dtype=np.float32)]


def _reference_bundle(path, run_tokens, jump_tokens):
    run_reference_qpos = np.zeros((len(run_tokens), 36), dtype=np.float32)
    run_reference_qpos[:, 2] = 0.75
    run_reference_qpos[:, 3] = 1.0
    run_reference_qpos[:, 7:] = G1_DEFAULT_ANGLES
    run_encoder_qpos = np.repeat(
        run_reference_qpos[-1:], len(run_tokens) + 45, axis=0
    )
    jump_encoder_qpos = np.repeat(
        run_reference_qpos[-1:], len(jump_tokens) + 45, axis=0
    )
    np.savez(
        path,
        run_encoder_qpos=run_encoder_qpos,
        jump_encoder_qpos=jump_encoder_qpos,
        run_reference_qpos=run_reference_qpos,
        jump_reference_qpos=jump_encoder_qpos[: len(jump_tokens)],
        run_reference_foot_contacts=np.zeros((len(run_tokens), 2), dtype=bool),
    )


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
    np.testing.assert_allclose(
        first_observation[-86:], env.unwrapped._command_observation()
    )
    np.testing.assert_array_equal(first_observation[-43:], np.zeros(43))

    native = env.unwrapped
    assert env.action_space == gym.spaces.Box(-1, 1, shape=(78,), dtype=np.float32)
    assert env.observation_space == gym.spaces.Box(
        -np.inf, np.inf, shape=(185,), dtype=np.float64
    )
    assert native.model.nq == 50
    assert native.model.nv == 49
    assert native.model.nu == 43
    assert native.model.opt.timestep == 0.002
    assert native.frame_skip == 10
    assert native.dt == 0.02
    assert (
        gym.spec(sonic.G1_SONIC_ENV_SPECS["g1_sonic_run"]["env_id"]).max_episode_steps
        == 1000
    )
    assert native.robot.dof == 50

    next_observation, reward, terminated, truncated, info = env.step(
        np.zeros(78, dtype=np.float32)
    )
    assert next_observation.shape == (185,)
    np.testing.assert_allclose(
        next_observation[-86:], env.unwrapped._command_observation()
    )
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
    assert moving_info["move"] > 0.9


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
    native._q_cmd = np.concatenate([keyframe_body_q, np.zeros(14)])
    native._v_cmd.fill(0.0)

    body_q, body_dq = native._body_state()
    np.testing.assert_array_equal(
        body_q,
        (keyframe_body_q - G1_DEFAULT_ANGLES)[MUJOCO_INDEX_FOR_POLICY],
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
    expected_target, expected_hand_target = native._condition_targets(
        policy_action, hand_action, raw_body=False
    )
    control = native._controller(expected_target, expected_hand_target)
    expected_body_control = sonic._KP * (expected_target - keyframe_body_q)
    np.testing.assert_allclose(
        control[native._body_actuator_ids],
        expected_body_control,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        control[native._hand_actuator_ids],
        sonic._HAND_KP * expected_hand_target,
    )

    raw_action = np.linspace(-1.0, 1.0, 29)
    raw_target, raw_hand_target = native._condition_targets(
        raw_action, hand_action, raw_body=True
    )
    raw_control = native._controller(raw_target, raw_hand_target)
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
        assert observation["obs"].shape == (185,)
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


def test_hurdle_composed_reference_and_sonic_reset(
    monkeypatch, tmp_path, hurdle_scene_path
):
    run_tokens = np.linspace(-0.75, 0.75, 3 * 64, dtype=np.float32).reshape(3, 64)
    jump_tokens = np.full((2, 64), 0.5, dtype=np.float32)
    bundle_path = tmp_path / "reference.npz"
    _reference_bundle(bundle_path, run_tokens, jump_tokens)
    monkeypatch.setattr(sonic.ort, "InferenceSession", _DecoderSession)
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_hurdle"]["env_id"],
        scene_path=str(hurdle_scene_path),
        control_mode=sonic.G1_CONTROL_MODE_SONIC_COMPOSED,
        sonic_decoder_path="decoder.onnx",
        sonic_encoder_path="encoder.onnx",
        sonic_decoder_mode=sonic.SONIC_DECODER_MODE_ENVIRONMENT,
        sonic_reference_bundle_path=str(bundle_path),
        render_mode="rgb_array",
        width=32,
        height=32,
    )
    try:
        observation, _ = env.reset(seed=7)
        native = env.unwrapped
        assert env.action_space.shape == (29,)
        assert observation["obs"].shape == (
            native._task.observation_space.shape[0] + 930 + 64 + 2,
        )
        np.testing.assert_array_equal(observation["base_token"], np.zeros(64))
        np.testing.assert_array_equal(observation["obs"][-66:-2], np.zeros(64))
        np.testing.assert_array_equal(observation["obs"][-2:], [0.0, 0.0])
        np.testing.assert_allclose(
            native.data.qpos[native._body_qpos_addresses],
            G1_DEFAULT_ANGLES,
            atol=native.randomness,
        )
        np.testing.assert_allclose(
            native.data.qpos[native._hand_qpos_addresses],
            np.zeros(14),
            atol=native.randomness,
        )

        residual_action = np.linspace(-1.0, 1.0, 29, dtype=np.float32)
        next_observation, _, _, _, info = env.step(residual_action)
        np.testing.assert_array_equal(native._decoder.input[0, :64], np.zeros(64))
        np.testing.assert_allclose(
            native._last_action,
            (
                residual_action
                * sonic.SONIC_COMPOSED_PHYSICAL_RESIDUAL_SCALE
                / sonic._ACTION_SCALE
            )[MUJOCO_INDEX_FOR_POLICY],
        )
        assert info["residual_abs_mean"] == pytest.approx(
            np.abs(residual_action).mean()
        )
        assert info["residual_abs_max"] == 1.0
        assert info["residual_saturation_fraction"] == 2 / 29
        np.testing.assert_array_equal(next_observation["base_token"], np.zeros(64))
        assert native._run_phase == 1
        assert native._reference_mode == sonic.RUN_STATE

        native.data.qpos[native._hand_qpos_addresses] = 0.1
        native.data.qvel[native._hand_qvel_addresses] = 0.0
        _, hand_target = native._condition_targets(
            np.zeros(29), None, raw_body=False
        )
        control = native._controller(np.zeros(29), hand_target)
        np.testing.assert_allclose(
            control[native._hand_actuator_ids], -sonic._HAND_KP * 0.1
        )

        reset_observation, _ = env.reset(seed=11)
        np.testing.assert_array_equal(reset_observation["base_token"], np.zeros(64))
        assert native._run_phase == 0
        np.testing.assert_array_equal(native._last_action, np.zeros(29))
    finally:
        env.close()


def test_sonic_reset_keeps_board_keyframe_height(scene_paths):
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_balance_hard"]["env_id"],
        scene_path=str(scene_paths["g1_sonic_balance_hard"]),
        control_mode=sonic.G1_CONTROL_MODE_SONIC,
        sonic_decoder_path="unused.onnx",
        sonic_decoder_mode=sonic.SONIC_DECODER_MODE_INFERENCE_WORKER,
        render_mode="rgb_array",
        width=32,
        height=32,
    )
    try:
        env.reset(seed=7)
        native = env.unwrapped
        assert native.data.qpos[2] > 1.0
        assert abs(native.data.qpos[2] - native.model.key_qpos[0, 2]) <= native.randomness
    finally:
        env.close()


def test_direct_sonic_can_use_strict_64d_action_with_fixed_fingers(
    monkeypatch, scene_path
):
    monkeypatch.setattr(sonic.ort, "InferenceSession", _DecoderSession)
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_run"]["env_id"],
        scene_path=str(scene_path),
        control_mode=sonic.G1_CONTROL_MODE_SONIC,
        sonic_decoder_path="decoder.onnx",
        sonic_learn_hands=False,
        sonic_decoder_mode=sonic.SONIC_DECODER_MODE_ENVIRONMENT,
        render_mode="rgb_array",
        width=32,
        height=32,
    )
    try:
        observation, _ = env.reset(seed=0)
        native = env.unwrapped
        assert env.action_space.shape == (64,)
        assert observation.shape == (185,)
        _, hand_target = native._condition_targets(
            np.zeros(29), None, raw_body=False
        )
        np.testing.assert_allclose(hand_target, np.zeros(14), atol=1e-12)
        env.step(np.zeros(64, dtype=np.float32))
        np.testing.assert_allclose(native._q_cmd[29:], np.zeros(14), atol=1e-12)
    finally:
        env.close()


def test_direct_sonic_worker_strict_64d_keeps_fingers_fixed(scene_path):
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_run"]["env_id"],
        scene_path=str(scene_path),
        control_mode=sonic.G1_CONTROL_MODE_SONIC,
        sonic_learn_hands=False,
        sonic_decoder_mode=sonic.SONIC_DECODER_MODE_INFERENCE_WORKER,
        render_mode="rgb_array",
        width=32,
        height=32,
    )
    try:
        observation, _ = env.reset(seed=0)
        native = env.unwrapped
        assert env.action_space.shape == (64,)
        native.set_decoded_action(np.zeros(29, dtype=np.float32))
        env.step(np.zeros(64, dtype=np.float32))
        np.testing.assert_allclose(native._q_cmd[29:], np.zeros(14), atol=1e-12)
    finally:
        env.close()


def test_hurdle_composed_inference_worker_contract(
    monkeypatch, tmp_path, hurdle_scene_path
):
    run_tokens = np.zeros((2, 64), dtype=np.float32)
    jump_tokens = np.ones((2, 64), dtype=np.float32)
    bundle_path = tmp_path / "reference.npz"
    _reference_bundle(bundle_path, run_tokens, jump_tokens)
    monkeypatch.setattr(sonic.ort, "InferenceSession", _DecoderSession)
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_hurdle"]["env_id"],
        scene_path=str(hurdle_scene_path),
        control_mode=sonic.G1_CONTROL_MODE_SONIC_COMPOSED,
        sonic_decoder_path="unused.onnx",
        sonic_encoder_path="encoder.onnx",
        sonic_decoder_mode=sonic.SONIC_DECODER_MODE_INFERENCE_WORKER,
        sonic_reference_bundle_path=str(bundle_path),
        render_mode="rgb_array",
        width=32,
        height=32,
    )
    try:
        observation, _ = env.reset(seed=0)
        assert observation["obs"].shape == (
            env.unwrapped._task.observation_space.shape[0] + 930 + 64 + 2,
        )
        assert observation["sonic_state"].shape == (930,)
        assert observation["base_token"].shape == (64,)
        env.unwrapped.set_decoded_action(np.zeros(29, dtype=np.float32))
        next_observation, *_ = env.step(np.zeros(29, dtype=np.float32))
        np.testing.assert_array_equal(next_observation["base_token"], np.zeros(64))
        assert env.unwrapped._run_phase == 1
    finally:
        env.close()


def test_run_composed_reference_advances_without_hurdle_state(
    monkeypatch, tmp_path, scene_path
):
    run_tokens = np.zeros((1_001, 64), dtype=np.float32)
    run_reference_qpos = np.zeros((len(run_tokens), 36), dtype=np.float32)
    run_reference_qpos[:, 2] = 0.75
    run_reference_qpos[:, 3] = 1.0
    run_reference_qpos[:, 7:] = G1_DEFAULT_ANGLES
    run_encoder_qpos = np.repeat(
        run_reference_qpos[-1:], len(run_tokens) + 45, axis=0
    )
    bundle_path = tmp_path / "run_reference.npz"
    np.savez(
        bundle_path,
        run_encoder_qpos=run_encoder_qpos,
        run_reference_qpos=run_reference_qpos,
        run_reference_foot_contacts=np.zeros((len(run_tokens), 2), dtype=bool),
    )
    monkeypatch.setattr(sonic.ort, "InferenceSession", _DecoderSession)
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_run"]["env_id"],
        scene_path=str(scene_path),
        control_mode=sonic.G1_CONTROL_MODE_SONIC_COMPOSED,
        sonic_decoder_path="decoder.onnx",
        sonic_encoder_path="encoder.onnx",
        sonic_decoder_mode=sonic.SONIC_DECODER_MODE_ENVIRONMENT,
        sonic_reference_bundle_path=str(bundle_path),
        render_mode="rgb_array",
        width=32,
        height=32,
    )
    try:
        observation, _ = env.reset(seed=0)
        native = env.unwrapped
        monkeypatch.setattr(native._task, "get_terminated", lambda: (False, {}))
        assert not hasattr(native, "_hurdle_x_positions")
        assert native._reference_mode == sonic.RUN_STATE
        for phase in range(1_000):
            observation, _, terminated, truncated, _ = env.step(
                np.zeros(29, dtype=np.float32)
            )
            assert terminated is False
            assert truncated is (phase == 999)
            assert native._run_phase == phase + 1
        assert observation["base_token"].shape == (64,)
    finally:
        env.close()


def test_composed_control_latches_jump_and_applies_physical_residual(
    monkeypatch, tmp_path, hurdle_scene_path
):
    run_tokens = np.zeros((3, 64), dtype=np.float32)
    jump_tokens = np.full((32, 64), 0.25, dtype=np.float32)
    bundle_path = tmp_path / "reference.npz"
    _reference_bundle(bundle_path, run_tokens, jump_tokens)
    monkeypatch.setattr(sonic.ort, "InferenceSession", _DecoderSession)
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_hurdle"]["env_id"],
        scene_path=str(hurdle_scene_path),
        control_mode=sonic.G1_CONTROL_MODE_SONIC_COMPOSED,
        sonic_decoder_path="decoder.onnx",
        sonic_encoder_path="encoder.onnx",
        sonic_decoder_mode=sonic.SONIC_DECODER_MODE_ENVIRONMENT,
        sonic_reference_bundle_path=str(bundle_path),
        hurdle_count=1,
        render_mode="rgb_array",
        width=32,
        height=32,
    )
    try:
        env.reset(seed=0)
        native = env.unwrapped
        assert native._hurdle_x_positions == (7.0,)
        native.data.qpos[0] = 6.0
        native._maybe_trigger_jump()
        assert native._reference_mode == sonic.JUMP_STATE
        assert native._jump_phase == 30
        observation, _, _, _, first_info = env.step(np.ones(29, dtype=np.float32))
        np.testing.assert_array_equal(native._decoder.input[0, :64], np.zeros(64))
        np.testing.assert_array_equal(observation["base_token"], np.zeros(64))
        np.testing.assert_array_equal(observation["obs"][-2:], [1.0, 31 / 32])
        assert first_info["target_velocity_max_ratio"] <= 1.0 + 1e-12
        assert first_info["target_acceleration_max_ratio_per_s"] > 10.0

        residual_action = np.ones(29, dtype=np.float32)
        _, _, _, _, info = env.step(residual_action)
        np.testing.assert_array_equal(native._decoder.input[0, :64], np.zeros(64))
        np.testing.assert_allclose(
            native._last_action,
            (sonic.SONIC_COMPOSED_PHYSICAL_RESIDUAL_SCALE / sonic._ACTION_SCALE)[
                MUJOCO_INDEX_FOR_POLICY
            ],
        )
        assert native._jump_phase == 0
        assert info["residual_abs_mean"] == 1.0
        assert info["residual_abs_max"] == 1.0
        assert info["residual_saturation_fraction"] == 1.0

        native._reference_mode = sonic.JUMP_STATE
        native._jump_phase = 30
        monkeypatch.setattr(
            native._task,
            "get_reward",
            lambda: (
                -200.0,
                {
                    "failure_penalty": 0.0,
                    "quality_failure_penalty": -200.0,
                    "timeout_penalty": 0.0,
                },
            ),
        )
        monkeypatch.setattr(native._task, "get_terminated", lambda: (True, {}))
        _, reward, terminated, _, info = env.step(residual_action)
        assert reward == -200.0
        assert terminated is True
        assert info["residual_smoothness_penalty"] == 0.0
    finally:
        env.close()


def test_composed_hurdle_height_updates_all_collision_geoms(
    tmp_path, hurdle_scene_path, monkeypatch
):
    run_tokens = np.zeros((3, 64), dtype=np.float32)
    jump_tokens = np.zeros((32, 64), dtype=np.float32)
    bundle_path = tmp_path / "reference.npz"
    _reference_bundle(bundle_path, run_tokens, jump_tokens)
    monkeypatch.setattr(sonic.ort, "InferenceSession", _DecoderSession)
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_hurdle"]["env_id"],
        scene_path=str(hurdle_scene_path),
        control_mode=sonic.G1_CONTROL_MODE_SONIC_COMPOSED,
        sonic_decoder_path="decoder.onnx",
        sonic_encoder_path="encoder.onnx",
        sonic_reference_bundle_path=str(bundle_path),
        hurdle_height=0.2,
        render_mode="rgb_array",
        width=32,
        height=32,
    )
    try:
        model = env.unwrapped.model
        for geom_id in env.unwrapped._task._hurdle_geom_ids:
            assert model.geom_size[geom_id, 2] == pytest.approx(0.1)
            assert model.geom_pos[geom_id, 2] == pytest.approx(0.1)
        body_id = model.body("starting_hurdle_1_0").id
        visual_geom_ids = np.flatnonzero(model.geom_bodyid == body_id)
        np.testing.assert_allclose(model.geom_size[visual_geom_ids[2:4], 2], 0.1)
        np.testing.assert_allclose(model.geom_pos[visual_geom_ids[2:4], 2], 0.1)
        assert model.geom_pos[visual_geom_ids[4], 2] == pytest.approx(0.17)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("task_name", "expected_obs_dim", "horizon"),
    [
        ("g1_sonic_balance_hard", 211, 1000),
        ("g1_sonic_hurdle", 186, 1000),
        ("g1_sonic_run", 185, 1000),
        ("g1_sonic_basketball", 198, 500),
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


def test_raw_reset_action_matches_default_pose_and_can_drop_hands(scene_path):
    raw = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_run"]["env_id"],
        task_name="g1_sonic_run",
        scene_path=str(scene_path),
        control_mode=sonic.G1_CONTROL_MODE_RAW,
        raw_learn_hands=False,
        render_mode="rgb_array",
        width=32,
        height=32,
    )
    try:
        action = np.asarray(
            sonic.g1_raw_reset_action(scene_path, learn_hands=False),
            dtype=np.float64,
        )
        assert action.shape == (29,)
        assert raw.action_space.shape == (29,)
        native = raw.unwrapped
        body_target = native._body_joint_ranges[:, 0] + (action + 1.0) * 0.5 * (
            native._body_joint_ranges[:, 1] - native._body_joint_ranges[:, 0]
        )
        np.testing.assert_allclose(body_target, G1_DEFAULT_ANGLES)
        raw.reset(seed=0)
        raw.step(action.astype(np.float32))
    finally:
        raw.close()


def test_direct_absolute_mode_exposes_policy_order_targets_and_93d_proprio(scene_path):
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_run"]["env_id"],
        task_name="g1_sonic_run",
        scene_path=str(scene_path),
        control_mode=sonic.G1_CONTROL_MODE_DIRECT_ABSOLUTE,
        target_velocity_ramp_time_s=None,
        camera_name="head_camera",
        render_mode=None,
        width=32,
        height=32,
    )
    try:
        _, reset_info = env.reset(seed=0)
        native = env.unwrapped
        expected_low = native._body_joint_ranges[:, 0][MUJOCO_INDEX_FOR_POLICY]
        expected_high = native._body_joint_ranges[:, 1][MUJOCO_INDEX_FOR_POLICY]
        np.testing.assert_allclose(env.action_space.low, expected_low)
        np.testing.assert_allclose(env.action_space.high, expected_high)
        assert env.action_space.shape == (29,)
        assert native.camera_name == "head_camera"
        assert reset_info[sonic.GR00T_PROPRIO_INFO].shape == (93,)
        assert reset_info[sonic.GR00T_PROPRIO_INFO].dtype == np.float32

        action = native.data.qpos[native._body_qpos_addresses][
            MUJOCO_INDEX_FOR_POLICY
        ].copy()
        action[0] += 0.01
        _, _, _, _, info = env.step(action.astype(np.float32))
        proprio = info[sonic.GR00T_PROPRIO_INFO]
        target = info[sonic.CONDITIONED_BODY_TARGET_INFO]
        np.testing.assert_allclose(info[sonic.ACTION_TARGET_INFO], target)
        np.testing.assert_allclose(target, action, atol=1e-6)
        np.testing.assert_allclose(
            proprio[:3], native._base_angular_velocity(), atol=1e-12
        )
        np.testing.assert_allclose(
            proprio[3:32],
            native.data.qpos[native._body_qpos_addresses][MUJOCO_INDEX_FOR_POLICY],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            proprio[32:61],
            native.data.qvel[native._body_qvel_addresses][MUJOCO_INDEX_FOR_POLICY],
            atol=1e-12,
        )
        np.testing.assert_allclose(proprio[61:90], target, atol=1e-12)
        np.testing.assert_allclose(proprio[90:], native._projected_gravity(), atol=1e-12)
    finally:
        env.close()


def test_raw_command_limiter_applies_velocity_acceleration_and_joint_bounds(scene_paths):
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_balance_hard"]["env_id"],
        task_name="g1_sonic_balance_hard",
        scene_path=str(scene_paths["g1_sonic_balance_hard"]),
        control_mode=sonic.G1_CONTROL_MODE_RAW,
        render_mode="rgb_array",
        width=16,
        height=16,
    )
    try:
        observation, _ = env.reset(seed=0)
        native = env.unwrapped
        np.testing.assert_allclose(observation[-86:], native._command_observation())
        np.testing.assert_array_equal(observation[-43:], np.zeros(43))
        previous_q_cmd = native._q_cmd.copy()
        previous_v_cmd = native._v_cmd.copy()
        _, _, _, _, info = env.step(np.ones(43, dtype=np.float32))
        assert info["command_tracking_error"] > 0.0
        assert info["target_velocity_max_ratio"] <= 1.0 + 1e-12
        assert info["target_acceleration_max_ratio_per_s"] <= 10.0 + 1e-9
        target_velocity_limits = native._target_velocity_limits
        target_acceleration_step = target_velocity_limits * native.dt / 0.1
        np.testing.assert_array_less(
            np.abs(native._q_cmd - previous_q_cmd) / native.dt,
            target_velocity_limits + 1e-12,
        )
        np.testing.assert_array_less(
            np.abs(native._v_cmd - previous_v_cmd),
            target_acceleration_step + 1e-12,
        )
        np.testing.assert_array_less(
            native._q_cmd,
            native._joint_ranges[:, 1] + 1e-12,
        )
        np.testing.assert_array_less(
            native._joint_ranges[:, 0] - 1e-12,
            native._q_cmd,
        )

        for action in (np.full(43, -1.0), np.full(43, 1.0)):
            previous_v_cmd = native._v_cmd.copy()
            _, _, _, _, info = env.step(action.astype(np.float32))
            np.testing.assert_array_less(
                np.abs(native._v_cmd - previous_v_cmd),
                target_acceleration_step + 1e-12,
            )
            assert info["target_velocity_max_ratio"] <= 1.0 + 1e-12

        q_previous = native._joint_ranges[:, 1] - 1e-12
        _, velocity = native._condition_velocity(
            native._joint_ranges[:, 1],
            q_previous,
            np.zeros(43),
            native._joint_ranges,
            target_velocity_limits,
        )
        np.testing.assert_array_less(
            q_previous + velocity * native.dt,
            native._joint_ranges[:, 1] + 1e-12,
        )

        rng = np.random.default_rng(0)
        q_cmd = native._q_cmd.copy()
        v_cmd = native._v_cmd.copy()
        for _ in range(1_000):
            q_request = rng.uniform(
                native._joint_ranges[:, 0], native._joint_ranges[:, 1]
            )
            next_q_cmd, next_v_cmd = native._condition_velocity(
                q_request,
                q_cmd,
                v_cmd,
                native._joint_ranges,
                target_velocity_limits,
            )
            np.testing.assert_allclose(
                next_q_cmd - q_cmd, next_v_cmd * native.dt, atol=1e-12
            )
            np.testing.assert_array_less(
                np.abs(next_v_cmd - v_cmd), target_acceleration_step + 1e-12
            )
            np.testing.assert_array_less(
                next_q_cmd, native._joint_ranges[:, 1] + 1e-12
            )
            np.testing.assert_array_less(
                native._joint_ranges[:, 0] - 1e-12, next_q_cmd
            )
            q_cmd, v_cmd = next_q_cmd, next_v_cmd

        native._target_acceleration_limits = None
        q_previous = native._joint_ranges.mean(axis=1)
        q_request = native._joint_ranges[:, 1]
        q_cmd, v_cmd = native._condition_velocity(
            q_request,
            q_previous,
            np.zeros(43),
            native._joint_ranges,
            target_velocity_limits,
        )
        expected_q_cmd = np.minimum(
            q_previous + target_velocity_limits * native.dt,
            native._joint_ranges[:, 1],
        )
        np.testing.assert_allclose(q_cmd, expected_q_cmd)
        np.testing.assert_allclose(v_cmd, (q_cmd - q_previous) / native.dt)
    finally:
        env.close()


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


def test_g1_sonic_joint_mapping_is_bidirectional():
    np.testing.assert_array_equal(
        POLICY_INDEX_FOR_MUJOCO[MUJOCO_INDEX_FOR_POLICY],
        np.arange(len(MUJOCO_INDEX_FOR_POLICY)),
    )
    for task_key, task_class in sonic.TASKS.items():
        if task_key in sonic.G1_SONIC_UNSUPPORTED_TASKS:
            continue
        spec = sonic.G1_SONIC_ENV_SPECS[f"g1_sonic_{task_key}"]
        assert spec["scene_filename"] == f"sonic_g1/envs/scene_{task_key}.xml"
        assert spec["horizon"] == task_class.max_episode_steps
        assert (
            gym.spec(spec["env_id"]).max_episode_steps == task_class.max_episode_steps
        )


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
        assert policy_observation.shape == (
            env.unwrapped._task.observation_space.shape[0] + 86,
        )
        assert np.isfinite(reward)
        assert truncated is False
        assert env.unwrapped.robot.dof == 50
        assert env.unwrapped.model.nq >= 50
        assert env.unwrapped.model.nv >= 49
        assert env.unwrapped.model.nu == 43
        assert env.render().shape == (16, 16, 3)
    finally:
        env.close()

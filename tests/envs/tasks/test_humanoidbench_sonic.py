from __future__ import annotations

import os
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from latency_bench.envs.tasks.humanoidbench import sonic as sonic
from latency_bench.envs.tasks.humanoidbench.contract import (
    G1_DEFAULT_ANGLES,
    G1_HAND_JOINT_NAMES,
    MUJOCO_INDEX_FOR_POLICY,
    POLICY_INDEX_FOR_MUJOCO,
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
    path = scene_root / sonic.G1_SONIC_ENV_SPECS["g1_sonic_balance_simple"][
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
def env(scene_path):
    instance = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_balance_simple"]["env_id"],
        scene_path=str(scene_path),
        control_mode=sonic.G1_CONTROL_MODE_RAW,
        render_mode="rgb_array",
        width=64,
        height=64,
    )
    yield instance
    instance.close()


def test_g1_balance_environment_contract(env):
    first_observation, _ = env.reset(seed=17)
    second_observation, _ = env.reset(seed=17)
    np.testing.assert_array_equal(second_observation, first_observation)
    np.testing.assert_allclose(
        first_observation[-86:], env.unwrapped._command_observation()
    )
    np.testing.assert_array_equal(first_observation[-43:], np.zeros(43))

    native = env.unwrapped
    assert env.action_space == gym.spaces.Box(-1, 1, shape=(43,), dtype=np.float32)
    assert env.observation_space == gym.spaces.Box(
        -np.inf, np.inf, shape=(198,), dtype=np.float64
    )
    assert native.model.nq == 57
    assert native.model.nv == 55
    assert native.model.nu == 43
    assert native.model.opt.timestep == 0.002
    assert native.frame_skip == 10
    assert native.dt == 0.02
    assert (
        gym.spec(sonic.G1_SONIC_ENV_SPECS["g1_sonic_balance_simple"]["env_id"]).max_episode_steps
        == 1000
    )
    assert native.robot.dof == 50

    next_observation, reward, terminated, truncated, info = env.step(
        np.zeros(43, dtype=np.float32)
    )
    assert next_observation.shape == (198,)
    np.testing.assert_allclose(
        next_observation[-86:], env.unwrapped._command_observation()
    )
    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert info["per_timestep_reward"] == reward
    assert env.render().shape == (64, 64, 3)
    assert native.camera_name == "cam_default"


@pytest.mark.parametrize(
    ("task_name", "expected_obs_dim", "horizon"),
    [
        ("g1_sonic_balance_hard", 211, 1000),
        ("g1_sonic_balance_simple", 198, 1000),
        ("g1_sonic_basketball", 198, 500),
    ],
)
def test_g1_task_family_raw_contract(
    scene_paths, task_name, expected_obs_dim, horizon
):
    spec = sonic.G1_SONIC_ENV_SPECS[task_name]
    raw = gym.make(
        spec["env_id"],
        task_name=task_name,
        scene_path=str(scene_paths[task_name]),
        control_mode=sonic.G1_CONTROL_MODE_RAW,
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


def test_raw_reset_action_matches_default_pose_and_can_drop_hands(scene_path):
    raw = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_balance_simple"]["env_id"],
        task_name="g1_sonic_balance_simple",
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


def test_direct_absolute_mode_exposes_policy_order_targets(scene_path):
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_balance_simple"]["env_id"],
        task_name="g1_sonic_balance_simple",
        scene_path=str(scene_path),
        control_mode=sonic.G1_CONTROL_MODE_DIRECT_ABSOLUTE,
        camera_name="head_camera",
        render_mode=None,
        width=32,
        height=32,
    )
    try:
        env.reset(seed=0)
        native = env.unwrapped
        expected_low = np.concatenate(
            [
                native._body_joint_ranges[:, 0][MUJOCO_INDEX_FOR_POLICY],
                native._hand_joint_ranges[:, 0],
            ]
        )
        expected_high = np.concatenate(
            [
                native._body_joint_ranges[:, 1][MUJOCO_INDEX_FOR_POLICY],
                native._hand_joint_ranges[:, 1],
            ]
        )
        np.testing.assert_allclose(env.action_space.low, expected_low)
        np.testing.assert_allclose(env.action_space.high, expected_high)
        assert env.action_space.shape == (43,)
        assert native.camera_name == "head_camera"
        assert len(G1_HAND_JOINT_NAMES) == 14

        action = np.concatenate(
            [
                native.data.qpos[native._body_qpos_addresses][MUJOCO_INDEX_FOR_POLICY],
                native.data.qpos[native._hand_qpos_addresses],
            ]
        )
        action[0] += 0.01
        action[29] += 0.05
        env.step(action.astype(np.float32))
        np.testing.assert_allclose(
            native._q_cmd,
            np.concatenate([action[:29][POLICY_INDEX_FOR_MUJOCO], action[29:]]),
            atol=1e-6,
        )
    finally:
        env.close()


def test_raw_and_absolute_requests_reach_pd_with_only_xml_physical_bounds(scene_paths):
    env = gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_balance_hard"]["env_id"],
        task_name="g1_sonic_balance_hard",
        scene_path=str(scene_paths["g1_sonic_balance_hard"]),
        control_mode=sonic.G1_CONTROL_MODE_RAW,
        render_mode=None,
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
        issued = native.g1_absolute_request_from_raw_action(np.ones(43, dtype=np.float32))
        chunk = np.random.default_rng(42).uniform(-1, 1, (16, 43)).astype(np.float32)
        np.testing.assert_array_equal(
            native.g1_absolute_request_from_raw_action(chunk),
            np.stack([native.g1_absolute_request_from_raw_action(head) for head in chunk]),
        )
        np.testing.assert_allclose(
            issued,
            np.concatenate([
                native._body_joint_ranges[:, 1][MUJOCO_INDEX_FOR_POLICY],
                native._hand_joint_ranges[:, 1],
            ]),
        )
        np.testing.assert_array_equal(native._q_cmd, previous_q_cmd)
        np.testing.assert_array_equal(native._v_cmd, previous_v_cmd)
        env.step(np.ones(43, dtype=np.float32))
        np.testing.assert_allclose(
            issued,
            np.concatenate([native._q_cmd[:29][MUJOCO_INDEX_FOR_POLICY], native._q_cmd[29:]]),
        )
        direct = gym.make(
            env.spec.id,
            **(env.spec.kwargs | {
                "control_mode": sonic.G1_CONTROL_MODE_DIRECT_ABSOLUTE,
                "render_mode": None,
            }),
        )
        try:
            direct.reset(seed=0)
            direct.step(issued)
            np.testing.assert_allclose(
                direct.unwrapped._q_cmd, native._q_cmd, atol=1e-6
            )
            # The absolute codec rounds requests to float32; contact dynamics amplify
            # that sub-microradian rounding to about 2e-5 in this saturated step.
            np.testing.assert_allclose(direct.unwrapped.data.qpos, native.data.qpos, atol=1e-4)
        finally:
            direct.close()
        np.testing.assert_allclose(native._v_cmd, (native._q_cmd - previous_q_cmd) / native.dt)
        # Absolute requests are not clipped to joint ranges, and Python returns raw PD.
        request = native._joint_ranges[:, 1] + 10.0
        body, hands = native._condition_targets(request[:29], request[29:], raw_body=False, absolute_position=True)
        np.testing.assert_array_equal(native._q_cmd, request)
        native._controller(body, hands)
        ctrl = native.data.ctrl.copy()
        assert np.any(ctrl > native.model.actuator_ctrlrange[:, 1])
        assert native.model.actuator_ctrllimited.all()
        mujoco.mj_step(native.model, native.data)
        np.testing.assert_array_less(native.data.actuator_force, native.model.actuator_ctrlrange[:, 1] + 1e-6)
        np.testing.assert_array_less(native.model.actuator_ctrlrange[:, 0] - 1e-6, native.data.actuator_force)
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
    assert len(sonic.G1_SONIC_ENV_SPECS) == len(sonic.TASKS) - 4 == 28
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
    "g1_sonic_balance_simple",
    "g1_sonic_basketball",
}


@pytest.mark.parametrize(
    "task_name",
    sorted(set(sonic.G1_SONIC_ENV_SPECS) - _INITIAL_TRAINING_TASKS),
)
def test_extended_g1_task_family_reset_step_and_render(
    scene_paths, task_name
):
    spec = sonic.G1_SONIC_ENV_SPECS[task_name]
    env = gym.make(
        spec["env_id"],
        task_name=task_name,
        scene_path=str(scene_paths[task_name]),
        control_mode=sonic.G1_CONTROL_MODE_RAW,
        render_mode="rgb_array",
        width=16,
        height=16,
    )
    try:
        first, _ = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        observation, reward, _terminated, truncated, _info = env.step(
            np.zeros(43, dtype=np.float32)
        )
        assert observation.shape == (
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


def test_balance_branch_restores_physics_controller_and_rng(scene_paths):
    """Long-lived DAgger contract: independent native branches reproduce the next transition."""
    envs = [gym.make(
        sonic.G1_SONIC_ENV_SPECS["g1_sonic_balance_simple"]["env_id"],
        scene_path=str(scene_paths["g1_sonic_balance_simple"]),
        control_mode=sonic.G1_CONTROL_MODE_DIRECT_ABSOLUTE,
        render_mode=None,
    ) for _ in range(3)]
    try:
        envs[0].reset(seed=13)
        source, reference, branch = [env.unwrapped for env in envs]
        action = source.g1_absolute_request_from_raw_action(np.zeros(43, dtype=np.float32))
        for _ in range(4):
            source.step(action)
        reference.copy_balance_state_from(source)
        branch.copy_balance_state_from(source)
        state = source._policy_observation().copy()
        branch.step(action + 0.1)
        branch.np_random.standard_normal()
        np.testing.assert_array_equal(source._policy_observation(), state)
        actual, expected = source.step(action), reference.step(action)
        np.testing.assert_array_equal(source.data.qpos, reference.data.qpos)
        np.testing.assert_array_equal(source._policy_observation(), reference._policy_observation())
        assert actual[1:4] == expected[1:4]
        assert source.np_random.standard_normal() == reference.np_random.standard_normal()
    finally:
        for env in envs:
            env.close()

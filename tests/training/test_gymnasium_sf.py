from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.latency.sample import LatencySample
from latency_bench.envs.gymnasium_task import (
    gymnasium_discrete_action_labels,
    gymnasium_discrete_action_values,
    gymnasium_noop_action_id,
)
from latency_bench.policy.sample_factory import _GymnasiumPolicyActionResolver
from training.common.latency_bench_periodic_eval import (
    build_latency_bench_config_from_training_config,
)
from training.gymnasium_sf.integration import (
    ChannelFirstFrameStackWrapper,
    G1RawRequestWrapper,
    GymnasiumActionCatalogWrapper,
    GymnasiumActionLatencyWrapper,
    GymnasiumStateInfoWrapper,
    SAMPLE_FACTORY_CONFIG_PARSER,
    build_cli_args_from_config,
    make_reference_env,
)


class _NativeActionTraceEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(31)

    def __init__(self):
        self.native_actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.native_actions.append(action)
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


class _ChannelFirstImageEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 2, 2), dtype=np.uint8)
    action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros((1, 2, 2), dtype=np.uint8), {}

    def step(self, action):
        return np.ones((1, 2, 2), dtype=np.uint8), 0.0, False, False, {}


class _StateTraceEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(31)

    def __init__(self):
        self.state = 0
        self.native_actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return np.array([self.state], dtype=np.float32), {}

    def step(self, action):
        self.native_actions.append(action)
        self.state += 1
        return np.array([self.state], dtype=np.float32), 0.0, False, False, {}


def _vector_gymnasium_config():
    config = load_config(
        "configs/examples/gymnasium/inverted_pendulum/small_model_train.yaml"
    )
    config["backend"]["device"] = "cpu"
    config["env"] = {
        "name": "gymnasium",
        "task_name": "mountain_car",
        "env_id": "MountainCar-v0",
        "registration_imports": [],
        "make_kwargs": {"render_mode": "rgb_array"},
        "env_fps": 30,
        "obs_fps": 10,
        "frame_stack": 1,
        "noop_action": "coast",
        "action_map": {"push_left": 0, "coast": 1, "push_right": 2},
        "action_order": ["push_left", "coast", "push_right"],
        "base_prompt": (
            "Drive the car up the right hill. Choose exactly one action from: "
            "push_left, coast, push_right."
        ),
    }
    return config


def test_task_contract_uses_explicit_action_order_and_noop_position():
    env_cfg = {
        "action_map": {"coast": 1, "push_right": 2, "push_left": 0},
        "action_order": ["push_left", "coast", "push_right"],
        "noop_action": "coast",
    }

    assert gymnasium_discrete_action_labels(env_cfg) == (
        "push_left",
        "coast",
        "push_right",
    )
    assert gymnasium_discrete_action_values(env_cfg) == (0, 1, 2)
    assert gymnasium_noop_action_id(env_cfg) == 1


def test_inverted_pendulum_rgb_state_adaptation_declares_state_contract():
    config = load_config(
        "configs/examples/gymnasium/inverted_pendulum_rgb_state/small_model_train.yaml"
    )
    args = build_cli_args_from_config(config)

    assert config["env"]["task_name"] == "inverted_pendulum_rgb_state"
    assert config["env"]["state_space"]["labels"] == [
        "cart_position",
        "pole_angle",
        "cart_velocity",
        "pole_angular_velocity",
    ]
    state_arg = args[args.index("--gym-state-labels-json") + 1]
    assert state_arg == (
        '["cart_position", "pole_angle", "cart_velocity", "pole_angular_velocity"]'
    )


def test_walker2d_rgb_state_adaptation_declares_seventeen_state_values():
    config = load_config(
        "configs/examples/gymnasium/walker2d_rgb_state/small_model_train.yaml"
    )

    assert config["env"]["task_name"] == "walker2d_rgb_state"
    assert len(config["env"]["state_space"]["labels"]) == 17
    assert config["env"]["state_space"]["labels"][:2] == [
        "torso_height",
        "torso_angle",
    ]
    assert config["env"]["action_space"]["type"] == "box"
    assert len(config["env"]["action_space"]["labels"]) == 6


def test_action_catalog_and_latency_decode_nonzero_noop_to_native_action():
    native_env = _NativeActionTraceEnv()
    env = GymnasiumActionCatalogWrapper(native_env, [10, 20, 30])
    env = GymnasiumActionLatencyWrapper(
        env,
        fixed_latency_raw_frames=2,
        env_fps=10,
        obs_fps=10,
        hold_policy="hold",
        ordering_policy="issue_order_fifo",
        noop_action=1,
    )
    env.reset(seed=3)

    env.step(2)
    env.step(2)
    env.step(2)

    assert native_env.native_actions == [20, 20, 30]


def test_action_latency_latest_ready_drops_late_older_result():
    from latency_bench.latency.sample import LatencySample

    native_env = _NativeActionTraceEnv()
    env = GymnasiumActionLatencyWrapper(
        native_env,
        fixed_latency_raw_frames=0,
        env_fps=10,
        obs_fps=10,
        hold_policy="one_frame_then_noop",
        ordering_policy="latest_ready",
        noop_action=0,
    )
    env.reset(seed=3)
    samples = iter(
        [
            LatencySample.from_latency_ms(500.0, clock=env.clock),
            LatencySample.from_latency_ms(100.0, clock=env.clock),
            LatencySample.from_latency_ms(1000.0, clock=env.clock),
            LatencySample.from_latency_ms(1000.0, clock=env.clock),
            LatencySample.from_latency_ms(1000.0, clock=env.clock),
            LatencySample.from_latency_ms(1000.0, clock=env.clock),
        ]
    )
    env._next_latency_sample = lambda **_: next(samples)

    for action in (1, 2, 3, 4, 5, 6):
        env.step(action)

    assert native_env.native_actions == [0, 0, 2, 0, 0, 0]


def test_channel_first_frame_stack_repeats_reset_and_appends_new_frame():
    env = ChannelFirstFrameStackWrapper(_ChannelFirstImageEnv(), 4)

    reset_observation, _ = env.reset(seed=3)
    next_observation, *_ = env.step(0)

    assert env.observation_space.shape == (4, 2, 2)
    np.testing.assert_array_equal(reset_observation, np.zeros((4, 2, 2), dtype=np.uint8))
    np.testing.assert_array_equal(next_observation[:3], np.zeros((3, 2, 2), dtype=np.uint8))
    np.testing.assert_array_equal(next_observation[3], np.ones((2, 2), dtype=np.uint8))


def test_gymnasium_state_info_wrapper_preserves_numeric_observation():
    env = GymnasiumStateInfoWrapper(_NativeActionTraceEnv())

    reset_observation, reset_info = env.reset(seed=3)
    next_observation, *_, step_info = env.step(0)

    np.testing.assert_array_equal(reset_info["gymnasium_state"], reset_observation)
    np.testing.assert_array_equal(step_info["gymnasium_state"], next_observation)


def test_gymnasium_state_info_wrapper_uses_policy_state_from_dict_observation():
    observation = {
        "obs": np.array([1.0, 2.0]),
        "sonic_state": np.zeros(930),
    }
    np.testing.assert_array_equal(
        GymnasiumStateInfoWrapper._state(observation), observation["obs"]
    )


def test_gymnasium_state_is_unstacked_and_aligned_with_action():
    native_env = _StateTraceEnv()
    env = GymnasiumActionCatalogWrapper(
        GymnasiumStateInfoWrapper(native_env),
        [10, 20, 30],
    )
    env = ChannelFirstFrameStackWrapper(env, 2)

    reset_observation, reset_info = env.reset(seed=3)
    next_observation, _, _, _, step_info = env.step(1)

    np.testing.assert_array_equal(reset_observation, [0.0, 0.0])
    np.testing.assert_array_equal(reset_info["gymnasium_state"], [0.0])
    np.testing.assert_array_equal(next_observation, [0.0, 1.0])
    np.testing.assert_array_equal(step_info["gymnasium_state"], [1.0])
    assert native_env.native_actions == [20]


def test_sample_factory_config_round_trips_complete_task_contract():
    config = _vector_gymnasium_config()
    cfg = SAMPLE_FACTORY_CONFIG_PARSER.parse_train(build_cli_args_from_config(config))

    assert cfg.gym_task_name == "mountain_car"
    assert cfg.gym_env_id == "MountainCar-v0"
    assert cfg.gym_make_kwargs_json == '{"render_mode": "rgb_array"}'
    assert cfg.gym_registration_imports_json == "[]"
    assert cfg.gym_action_labels_json == '["push_left", "coast", "push_right"]'
    assert cfg.gym_action_values_json == "[0, 1, 2]"
    assert cfg.gym_noop_action_id == 1
    assert cfg.gym_base_prompt.startswith("Drive the car")
    assert cfg.use_env_info_cache is False

    env = make_reference_env(cfg)
    try:
        assert env.action_space == gym.spaces.Discrete(3)
        assert env.observation_space["obs"].shape == (2,)
    finally:
        env.close()


def test_gymnasium_sf_policy_decodes_catalog_ids_with_action_order():
    config = _vector_gymnasium_config()
    resolver = _GymnasiumPolicyActionResolver(
        ActionResolver(config["env"]),
        config["env"]["action_order"],
    )

    action = resolver.resolve(2)

    assert action.name == "push_right"
    assert action.value == 2


def test_periodic_eval_preserves_generic_task_contract(tmp_path):
    training_config = _vector_gymnasium_config()

    config = build_latency_bench_config_from_training_config(
        training_config,
        checkpoint_path=tmp_path / "checkpoint.pth",
        output_dir=tmp_path / "eval",
    )

    assert config["env"] == training_config["env"]
    assert config["policy"]["type"] == "gymnasium_sf"
    assert config["policy"]["checkpoint_path"] == str(tmp_path / "checkpoint.pth")
    assert config["backend"]["run_mode"] == "eval"


class _VectorActionTraceEnv(gym.Env):
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(43,), dtype=np.float32)

    def __init__(self):
        self.native_actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.native_actions.clear()
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.native_actions.append(np.asarray(action).copy())
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


class _ChunkDelaySampler:
    """Deterministic, cloneable timing stream for the wrapper scheduling contract."""
    def __init__(self, delays, clock):
        self.delays = delays
        self.clock = clock
        self.draws = 0

    def sample(self, context):
        ready, service = self.delays[self.draws % len(self.delays)]
        self.draws += 1
        return LatencySample.from_action_ready(
            action_ready_raw_frames=ready, worker_service_raw_frames=service,
            worker_service_ms=service * self.clock.frame_ms, clock=self.clock,
        )


@pytest.mark.parametrize("delays,expected,windows", [([(0, 0)],
  [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],
  [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]),
 ([(2, 2)],
  [(-1, -1), (-1, -1), (0, 2), (0, 3), (2, 2), (2, 3), (4, 2), (4, 3)],
  [(-1, -1), (-1, -1), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4)]),
 ([(2, 2), (3, 3), (2, 2)],
  [(-1, -1), (-1, -1), (0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (5, 2)],
  [(-1, -1), (-1, -1), (2, 5), (2, 5), (2, 5), (3, 5), (3, 5), (2, 4)]),
 ([(2, 3)],
  [(-1, -1), (-1, -1), (0, 2), (0, 3), (0, 4), (3, 2), (3, 3), (3, 4)],
  [(-1, -1), (-1, -1), (2, 5), (2, 5), (2, 5), (2, 5), (2, 5), (2, 5)]),
 ([(3, 2)],
  [(-1, -1), (-1, -1), (-1, -1), (0, 3), (0, 4), (2, 3), (2, 4), (4, 3)],
  [(-1, -1), (-1, -1), (-1, -1), (3, 5), (3, 5), (3, 5), (3, 5), (3, 5)]),
 ([(3, 2), (2, 2)],
  [(-1, -1), (-1, -1), (-1, -1), (0, 3), (2, 2), (2, 3), (2, 4), (4, 3)],
  [(-1, -1), (-1, -1), (-1, -1), (3, 4), (2, 5), (2, 5), (2, 5), (3, 4)]),
 ([(2, 10)],
  [(-1, -1), (-1, -1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 7), (0, 7)],
  [(-1, -1), (-1, -1), (2, 12), (2, 12), (2, 12), (2, 12), (2, 12), (2, 12), (2, 12), (2, 12)])])
@pytest.mark.parametrize("terminal_frame", [None, 4])
def test_g1_action_chunks_follow_execution_sources_and_planned_windows(delays, expected, windows, terminal_frame):
    native = _VectorActionTraceEnv()
    native._raw_learn_hands = True
    from latency_bench.envs.humanoidbench_sonic import MUJOCO_INDEX_FOR_POLICY

    def absolute_action(action):
        joint = 1.0 + 3.0 * action
        return np.concatenate((joint[:29][MUJOCO_INDEX_FOR_POLICY], joint[29:]))

    native.g1_absolute_request_from_raw_action = absolute_action
    latency = GymnasiumActionLatencyWrapper(
        native,
        fixed_latency_raw_frames=2,
        env_fps=10,
        obs_fps=10,
        hold_policy="hold",
        ordering_policy="latest_ready",
        simulated_worker_capacity=1,
        add_latency_info=True,
        hold_last_chunk_action=True,
        noop_action=np.zeros(43, dtype=np.float32),
    )
    latency.profile_sampler = _ChunkDelaySampler(delays, latency.clock)
    env = G1RawRequestWrapper(latency, action_chunk_horizon=8, latency_env=latency)
    observation, _ = env.reset(seed=1)
    assert env.action_space.shape == (344,)
    assert observation["replay_clock"].shape == (7,)
    np.testing.assert_array_equal(observation["obs"], np.zeros(1))

    original_step = native.step
    def step(action):
        obs, reward, _, truncated, info = original_step(action)
        return obs, reward, len(native.native_actions) == terminal_frame, truncated, info
    native.step = step
    traces = []
    chunks = []
    admitted = 0
    for frame in range(len(expected) if terminal_frame is None else terminal_frame):
        chunk = np.stack(
            [np.full(43, frame * 0.1 + index * 0.01, dtype=np.float32) for index in range(8)]
        )
        chunks.append(chunk)
        observation, _, _, _, info = env.step(chunk.reshape(-1))
        admitted += info["command_admitted"]
        assert latency.profile_sampler.draws == admitted
        traces.append(tuple(observation["replay_clock"][2:4]))
        assert tuple(observation["replay_clock"][5:7]) == windows[frame]
        source, head = expected[frame]
        if source >= 0:
            np.testing.assert_allclose(native.native_actions[frame], absolute_action(chunks[source][head]))
    assert traces == expected[:len(traces)]

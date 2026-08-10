from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from latency_bench.core.actions import ActionResolver
from latency_bench.core.config import load_config
from latency_bench.envs.gymnasium_task import (
    gymnasium_discrete_action_labels,
    gymnasium_discrete_action_values,
    gymnasium_noop_action_id,
)
from latency_bench.policy.sample_factory_common import SampleFactoryPolicyRunner
from training.common.latency_bench_periodic_eval import (
    build_latency_bench_config_from_training_config,
)
from training.common.latency_wrapper import ActionLatencyWrapper
from training.gymnasium_sf.integration import (
    GymnasiumActionCatalogWrapper,
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


def _mountain_car_config():
    return load_config("configs/examples/gymnasium_mountain_car_small_model_train.yaml")


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


def test_action_catalog_and_latency_decode_nonzero_noop_to_native_action():
    native_env = _NativeActionTraceEnv()
    env = GymnasiumActionCatalogWrapper(native_env, [10, 20, 30])
    env = ActionLatencyWrapper(
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


def test_sample_factory_config_round_trips_complete_task_contract():
    config = _mountain_car_config()
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
    config = _mountain_car_config()
    runner = SampleFactoryPolicyRunner.__new__(SampleFactoryPolicyRunner)
    runner.action_resolver = ActionResolver(config["env"])
    runner.policy_action_refs = config["env"]["action_order"]

    action = runner._resolve_action(2)

    assert action.name == "push_right"
    assert action.value == 2


def test_periodic_eval_preserves_generic_task_contract(tmp_path):
    training_config = _mountain_car_config()

    config = build_latency_bench_config_from_training_config(
        training_config,
        checkpoint_path=tmp_path / "checkpoint.pth",
        output_dir=tmp_path / "eval",
    )

    assert config["env"] == training_config["env"]
    assert config["policy"]["type"] == "gymnasium_sf"
    assert config["policy"]["checkpoint_path"] == str(tmp_path / "checkpoint.pth")
    assert config["backend"]["run_mode"] == "eval"

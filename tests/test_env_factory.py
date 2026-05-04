from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from latency_bench.core.config import load_config
from latency_bench.core.types import Action, Observation, PolicyOutput, StepResult
from latency_bench.envs.base import DummyEnvAdapter
from latency_bench.envs import deadly_corridor
from latency_bench.envs.deadly_corridor import _noop_action_from_space
from latency_bench.envs.flappy import FlappyBirdEnvAdapter
from latency_bench.envs.factory import build_env_adapter, get_registered_env_names
from latency_bench.experiments.run_simulated_latency import build_executor
from latency_bench.observation.builder import ObservationBuilder
from latency_bench.policy.base import ConstantPolicyRunner


def _config(env_name: str = "dummy", output_dir: str = "outputs/factory_test") -> dict:
    return {
        "experiment": {
            "name": "factory_test",
            "seed": 0,
            "num_episodes": 1,
        },
        "executor": {
            "mode": "simulated",
        },
        "env": {
            "name": env_name,
            "fps_env": 30,
            "noop_action": "noop",
            "action_map": {
                "noop": "noop",
                "jump": "jump",
            },
        },
        "control": {
            "frequency_hz": 30,
            "observe_every_n_frames": 1,
        },
        "latency": {
            "type": "zero",
        },
        "scheduler": {
            "persistence_mode": "noop_after_oneshot",
            "stale_action_mode": "latest_ready",
        },
        "policy": {
            "type": "constant",
            "action": "noop",
        },
        "evaluation": {
            "eval_max_steps": 5,
        },
        "logging": {
            "output_dir": output_dir,
        },
    }


def _flappy_config(output_dir: str = "outputs/factory_test") -> dict:
    config = _config("flappy", output_dir=output_dir)
    config["env"].update(
        {
            "gym_id": "FlappyBird-v0",
            "render_mode": "rgb_array",
            "observation_mode": "image",
            "use_lidar": False,
            "normalize_obs": True,
            "audio_on": False,
            "action_map": {
                "noop": 0,
                "flap": 1,
            },
            "oneshot_actions": ["flap"],
        }
    )
    config["policy"] = {
        "type": "constant",
        "action": "flap",
        "actions": ["noop", "flap"],
    }
    return config


def test_dummy_env_adapter_is_registered():
    assert "dummy" in get_registered_env_names()
    assert "flappy" in get_registered_env_names()


def test_build_env_adapter_returns_dummy_adapter_for_dummy_config():
    noop = Action(value="noop", name="noop", is_noop=True)

    env = build_env_adapter(_config("dummy"), noop)

    assert isinstance(env, DummyEnvAdapter)
    assert env.fps_env == 30.0
    assert env.max_steps == 5
    assert env.noop_action is noop


def test_build_env_adapter_rejects_unsupported_env_name():
    noop = Action(value="noop", name="noop", is_noop=True)

    with pytest.raises(ValueError, match="Unsupported env.name: unknown"):
        build_env_adapter(_config("unknown"), noop)


def test_demon_attack_base_config_declares_action_repeat():
    config = load_config("configs/base/demon_attack.yaml")

    assert config["env"]["action_repeat"] == 4


def test_deadly_corridor_noop_action_supports_tuple_action_space():
    from gymnasium.spaces import Discrete, Tuple

    action_space = Tuple((Discrete(3), Discrete(3), Discrete(3), Discrete(2)))

    assert _noop_action_from_space(action_space) == (0, 0, 0, 0)


def test_deadly_corridor_eval_raw_reward_disables_reward_scaling(monkeypatch):
    captured = {}

    class FakeDeadlyCorridorEnvAdapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        deadly_corridor,
        "DeadlyCorridorEnvAdapter",
        FakeDeadlyCorridorEnvAdapter,
    )
    config = _config("deadly_corridor")
    config["evaluation"] = {"eval_raw_reward": True}
    noop = Action(value=(0, 0, 0, 0), name="noop", is_noop=True)

    env = build_env_adapter(config, noop)

    assert isinstance(env, FakeDeadlyCorridorEnvAdapter)
    assert captured["config"] is config


def test_deadly_corridor_default_keeps_reward_scaling(monkeypatch):
    captured = {}

    class FakeDeadlyCorridorEnvAdapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        deadly_corridor,
        "DeadlyCorridorEnvAdapter",
        FakeDeadlyCorridorEnvAdapter,
    )
    config = _config("deadly_corridor")
    noop = Action(value=(0, 0, 0, 0), name="noop", is_noop=True)

    env = build_env_adapter(config, noop)

    assert isinstance(env, FakeDeadlyCorridorEnvAdapter)
    assert captured["config"] is config


def test_deadly_corridor_spec_copy_disables_reward_scaling_without_mutating_original():
    from types import SimpleNamespace

    spec = SimpleNamespace(reward_scaling=0.01)

    spec_to_use = deadly_corridor._spec_with_reward_scaling(
        spec,
        disable_reward_scaling=True,
    )

    assert spec_to_use is not spec
    assert spec.reward_scaling == 0.01
    assert spec_to_use.reward_scaling == 1.0


def test_deadly_corridor_spec_copy_keeps_reward_scaling_by_default():
    from types import SimpleNamespace

    spec = SimpleNamespace(reward_scaling=0.01)

    spec_to_use = deadly_corridor._spec_with_reward_scaling(
        spec,
        disable_reward_scaling=False,
    )

    assert spec_to_use is spec
    assert spec_to_use.reward_scaling == 0.01


def test_deadly_corridor_adapter_builds_prepared_eval_cfg_from_full_config(monkeypatch):
    from gymnasium.spaces import Discrete
    from training.deadly_corridor_sf import integration

    captured = {}

    def fake_build_cli_args_from_config(config):
        captured["build_config"] = config
        return ["--from-full-config", "1"]

    def fake_parse_cfg(argv):
        captured["parse_argv"] = list(argv)
        return SimpleNamespace(
            mode="eval",
            fps=35,
            env_frameskip=4,
            frame_stack=4,
            eval_raw_reward=True,
            disable_reward_scaling=False,
            max_episode_steps=2100,
        )

    def fake_prepare_eval_cfg(base_cfg, latency_value):
        captured["prepared_base_cfg"] = base_cfg
        captured["prepared_latency"] = latency_value
        return SimpleNamespace(
            mode=base_cfg.mode,
            fps=0,
            env_frameskip=base_cfg.env_frameskip,
            eval_env_frameskip=base_cfg.env_frameskip,
            frame_stack=base_cfg.frame_stack,
            eval_raw_reward=base_cfg.eval_raw_reward,
            disable_reward_scaling=True,
            max_episode_steps=base_cfg.max_episode_steps,
            queue_latency_values="0",
            latency_mode="episode_fixed",
            latency_sampling_mode="uniform_all",
            no_render=True,
        )

    def fake_load_from_checkpoint(cfg):
        cfg.fps = 0
        cfg.frame_stack = 1
        cfg.max_episode_steps = 0
        cfg.disable_reward_scaling = False
        cfg.loaded_from_checkpoint = True
        return cfg

    class FakeGymEnv:
        action_space = Discrete(3)

        def close(self):
            pass

    original_spec = SimpleNamespace(name="doom_deadly_corridor", reward_scaling=0.01)

    def fake_make_doom_env_from_spec(spec, env_name, cfg, env_config, render_mode=None, **kwargs):
        captured["doom_spec"] = spec
        captured["doom_env_name"] = env_name
        captured["doom_cfg"] = cfg
        captured["doom_env_config"] = env_config
        captured["doom_render_mode"] = render_mode
        return FakeGymEnv()

    monkeypatch.setattr(integration, "register_deadly_corridor_components", lambda: None)
    monkeypatch.setattr(integration, "build_cli_args_from_config", fake_build_cli_args_from_config)
    monkeypatch.setattr(integration, "parse_cfg", fake_parse_cfg)
    monkeypatch.setattr(integration, "_prepare_eval_cfg", fake_prepare_eval_cfg)
    monkeypatch.setitem(
        sys.modules,
        "sample_factory.cfg.arguments",
        SimpleNamespace(load_from_checkpoint=fake_load_from_checkpoint),
    )
    monkeypatch.setitem(
        sys.modules,
        "sample_factory.utils.attr_dict",
        SimpleNamespace(AttrDict=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setitem(
        sys.modules,
        "sf_examples.vizdoom.doom.doom_utils",
        SimpleNamespace(
            DOOM_ENVS=[original_spec],
            make_doom_env_from_spec=fake_make_doom_env_from_spec,
            make_latency_doom_env_from_spec=lambda *args, **kwargs: pytest.fail(
                "latency wrapper env factory must not be used"
            ),
        ),
    )

    config = _config("deadly_corridor")
    config["backend"] = {"run_mode": "eval"}
    config["training"] = {}
    config["evaluation"] = {"eval_raw_reward": True}
    config["env"].update(
        {
            "env_id": "doom_deadly_corridor",
            "fps_env": 35,
            "frame_stack": 4,
            "max_episode_steps": 2100,
        }
    )
    noop = Action(value=0, name="noop", is_noop=True)

    env = deadly_corridor.DeadlyCorridorEnvAdapter(config=config, noop_action=noop)

    assert captured["build_config"] is config
    assert captured["parse_argv"] == ["--from-full-config", "1"]
    assert captured["prepared_latency"] == 0
    assert captured["doom_env_name"] == "doom_deadly_corridor"
    assert captured["doom_render_mode"] is None
    assert env.cfg.loaded_from_checkpoint is True
    assert env.cfg.mode == "eval"
    assert env.cfg.fps == 35
    assert captured["doom_cfg"].fps > 0
    assert env.cfg.disable_reward_scaling is True
    assert env.cfg.max_episode_steps == 2100
    assert env.cfg.frame_stack == 4
    assert captured["doom_cfg"] is env.cfg
    assert captured["doom_spec"] is not original_spec
    assert captured["doom_spec"].reward_scaling == 1.0
    assert original_spec.reward_scaling == 0.01


def test_build_env_adapter_passes_demon_attack_action_repeat(monkeypatch):
    from latency_bench.envs import demon_attack

    captured = {}

    class FakeDemonAttackEnvAdapter:
        def __init__(self, *, action_repeat: int, **kwargs):
            captured["action_repeat"] = action_repeat
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        demon_attack,
        "DemonAttackEnvAdapter",
        FakeDemonAttackEnvAdapter,
    )
    config = _config("demon_attack")
    config["env"].update(
        {
            "env_id": "ALE/DemonAttack-v5",
            "action_repeat": 4,
            "screen_size": 84,
            "noop_max": 30,
            "terminal_on_life_loss": False,
            "clip_reward": True,
            "repeat_action_probability": 0.0,
            "full_action_space": False,
        }
    )
    noop = Action(value=0, name="noop", is_noop=True)

    env = build_env_adapter(config, noop)

    assert isinstance(env, FakeDemonAttackEnvAdapter)
    assert captured["action_repeat"] == 4


def test_demon_attack_latency_bench_eval_raw_reward_disables_clipping_and_keeps_life_loss_terminal(monkeypatch):
    from latency_bench.envs import demon_attack

    captured = {}

    class FakeDemonAttackEnvAdapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        demon_attack,
        "DemonAttackEnvAdapter",
        FakeDemonAttackEnvAdapter,
    )
    config = _config("demon_attack")
    config["backend"] = {"run_mode": "eval"}
    config["evaluation"] = {"eval_raw_reward": True}
    config["policy"] = {"type": "demon_attack_sf"}
    config["env"].update(
        {
            "env_id": "ALE/DemonAttack-v5",
            "action_repeat": 4,
            "screen_size": 84,
            "noop_max": 30,
            "terminal_on_life_loss": True,
            "clip_reward": True,
            "repeat_action_probability": 0.0,
            "full_action_space": False,
        }
    )
    noop = Action(value=0, name="noop", is_noop=True)

    build_env_adapter(config, noop)

    assert captured["terminal_on_life_loss"] is True
    assert captured["clip_reward"] is False
    assert captured["continue_life_loss_terminal"] is True


def test_build_executor_uses_registered_env_factory(tmp_path):
    executor = build_executor(_config("dummy", output_dir=str(tmp_path)))

    assert isinstance(executor.env, DummyEnvAdapter)


def test_build_env_adapter_returns_flappy_adapter_for_flappy_config():
    noop = Action(value=0, name="noop", is_noop=True)
    env = build_env_adapter(_flappy_config(), noop)

    try:
        assert isinstance(env, FlappyBirdEnvAdapter)
        assert env.fps_env == 30.0
        assert env.noop_action is noop
    finally:
        env.close()


def test_build_env_adapter_returns_flappy_sf_gpu_adapter_when_enabled(monkeypatch):
    from latency_bench.envs import factory as factory_module

    captured = {}

    class FakeFlappyGpuAdapter:
        def __init__(self, *, config, noop_action):
            captured["config"] = config
            captured["noop_action"] = noop_action

        def close(self):
            return None

    monkeypatch.setattr(factory_module, "FlappySampleFactoryGpuEnvAdapter", FakeFlappyGpuAdapter)
    config = _flappy_config()
    config["policy"] = {"type": "flappy_sf"}
    config["env"].update(
        {
            "use_gpu_render": True,
            "obs_resize": [84, 84],
            "obs_mode": "image_only",
            "frame_stack": 4,
        }
    )
    noop = Action(value=0, name="noop", is_noop=True)

    env = factory_module.build_env_adapter(config, noop)

    assert isinstance(env, FakeFlappyGpuAdapter)
    assert captured["config"] is config
    assert captured["noop_action"] is noop


def test_build_executor_maps_flappy_policy_actions(tmp_path):
    executor = build_executor(_flappy_config(output_dir=str(tmp_path)))

    try:
        assert isinstance(executor.env, FlappyBirdEnvAdapter)
        assert executor.policy.action.value == 1
        assert executor.policy.action.name == "flap"
        assert executor.policy.action.is_oneshot
        assert executor.action_scheduler.noop_action.value == 0
    finally:
        executor.env.close()


def test_sample_factory_policy_uses_frame_stack_observation_builder(monkeypatch, tmp_path):
    from latency_bench.executors import factory
    from latency_bench.observation.sample_factory import SampleFactoryFrameStackObservationBuilder

    noop = Action(value=0, name="noop", is_noop=True)

    def fake_build_env_adapter(config, noop_action):
        return DummyEnvAdapter(fps_env=30, max_steps=1, noop_action=noop)

    def fake_build_policy(config, action_resolver):
        return ConstantPolicyRunner(noop)

    monkeypatch.setattr(factory, "build_env_adapter", fake_build_env_adapter)
    monkeypatch.setattr(factory, "build_policy", fake_build_policy)
    config = _config("deadly_corridor", output_dir=str(tmp_path))
    config["env"].update({"frame_stack": 4})
    config["policy"]["type"] = "deadly_corridor_sf"

    executor = build_executor(config)

    assert isinstance(executor.observation_builder, SampleFactoryFrameStackObservationBuilder)
    assert executor.observation_builder.frame_stack == 4
    assert executor.observation_builder.expected_channels == 12


def test_latency_bench_flappy_sf_eval_uses_same_obs_resize(monkeypatch, tmp_path):
    from latency_bench.executors import factory
    from latency_bench.envs.base import EnvAdapter
    from latency_bench.policy.base import PolicyRunner

    captured = {}
    noop = Action(value=0, name="noop", is_noop=True)

    class SingleFrameEnv(EnvAdapter):
        fps_env = 30.0
        noop_action = noop

        def __init__(self):
            self.env_step = 0

        @property
        def frame_ms(self):
            return 1000.0 / self.fps_env

        def reset(self, seed=None):
            self.env_step = 0
            return Observation(
                data=np.zeros((3, 84, 84), dtype=np.uint8),
                env_step=0,
                sim_time_ms=0.0,
                metadata={"frame_skip": 0},
            )

        def step(self, action):
            self.env_step += 1
            return StepResult(
                observation=Observation(
                    data=np.ones((3, 84, 84), dtype=np.uint8),
                    env_step=self.env_step,
                    sim_time_ms=self.env_step * self.frame_ms,
                    metadata={"frame_skip": 0},
                ),
                reward=0.0,
                done=True,
            )

        def render_frame(self):
            return np.zeros((3, 84, 84), dtype=np.uint8)

        def get_state_info(self):
            return {"env_step": self.env_step, "sim_time_ms": self.env_step * self.frame_ms}

        def close(self):
            pass

    class CapturingPolicy(PolicyRunner):
        def predict(self, observation):
            captured["shape"] = observation.data.shape
            captured["dtype"] = observation.data.dtype
            return PolicyOutput(action=noop)

    monkeypatch.setattr(factory, "build_env_adapter", lambda config, noop_action: SingleFrameEnv())
    monkeypatch.setattr(factory, "build_policy", lambda config, action_resolver: CapturingPolicy())
    config = _flappy_config(output_dir=str(tmp_path))
    config["env"].update({"frame_stack": 4, "obs_resize": [84, 84], "obs_mode": "image_only"})
    config["policy"]["type"] = "flappy_sf"

    executor = build_executor(config)
    executor.run_episode(episode_id=0, seed=0, eval_max_steps=1)

    assert captured["shape"] == (12, 84, 84)
    assert captured["dtype"] == np.uint8


def test_latency_bench_flappy_sf_latency_channel_shape(monkeypatch, tmp_path):
    from modeling.flappy_ppo.preprocessing import latency_channel_value
    from latency_bench.executors import factory
    from latency_bench.envs.base import EnvAdapter
    from latency_bench.policy.base import PolicyRunner

    captured = {}
    noop = Action(value=0, name="noop", is_noop=True)

    class SingleFrameEnv(EnvAdapter):
        fps_env = 30.0
        noop_action = noop

        def reset(self, seed=None):
            return Observation(
                data=np.zeros((3, 84, 84), dtype=np.uint8),
                env_step=0,
                sim_time_ms=0.0,
                metadata={"frame_skip": 3},
            )

        def step(self, action):
            return StepResult(observation=None, reward=0.0, done=True)

        def render_frame(self):
            return np.zeros((3, 84, 84), dtype=np.uint8)

        def get_state_info(self):
            return {"env_step": 0, "sim_time_ms": 0.0}

        def close(self):
            pass

    class CapturingPolicy(PolicyRunner):
        def predict(self, observation):
            captured["shape"] = observation.data.shape
            captured["latency_value"] = int(observation.data[-1, 0, 0])
            return PolicyOutput(action=noop)

    monkeypatch.setattr(factory, "build_env_adapter", lambda config, noop_action: SingleFrameEnv())
    monkeypatch.setattr(factory, "build_policy", lambda config, action_resolver: CapturingPolicy())
    config = _flappy_config(output_dir=str(tmp_path))
    config["env"].update({"frame_stack": 4, "obs_resize": [84, 84], "obs_mode": "latency_channel"})
    config["latency"].update({"type": "fixed", "fixed_latency_ms": 100.0, "train_values": [0, 3]})
    config["policy"]["type"] = "flappy_sf"

    executor = build_executor(config)
    executor.run_episode(episode_id=0, seed=0, eval_max_steps=1)

    assert captured["shape"] == (13, 84, 84)
    assert captured["latency_value"] == int(latency_channel_value(3, 3))


def test_non_sample_factory_policy_uses_default_observation_builder(monkeypatch, tmp_path):
    from latency_bench.executors import factory

    noop = Action(value="noop", name="noop", is_noop=True)

    def fake_build_env_adapter(config, noop_action):
        return DummyEnvAdapter(fps_env=30, max_steps=1, noop_action=noop)

    def fake_build_policy(config, action_resolver):
        return ConstantPolicyRunner(noop)

    monkeypatch.setattr(factory, "build_env_adapter", fake_build_env_adapter)
    monkeypatch.setattr(factory, "build_policy", fake_build_policy)
    config = _config("dummy", output_dir=str(tmp_path))

    executor = build_executor(config)

    assert type(executor.observation_builder) is ObservationBuilder


def test_history_len_expands_to_frame_stack_for_sample_factory_policy(monkeypatch, tmp_path):
    from latency_bench.executors import factory

    noop = Action(value=0, name="noop", is_noop=True)

    def fake_build_env_adapter(config, noop_action):
        return DummyEnvAdapter(fps_env=30, max_steps=1, noop_action=noop)

    def fake_build_policy(config, action_resolver):
        return ConstantPolicyRunner(noop)

    monkeypatch.setattr(factory, "build_env_adapter", fake_build_env_adapter)
    monkeypatch.setattr(factory, "build_policy", fake_build_policy)
    config = _config("deadly_corridor", output_dir=str(tmp_path))
    config["control"]["history_len"] = 1
    config["env"]["frame_stack"] = 4
    config["policy"]["type"] = "deadly_corridor_sf"

    executor = build_executor(config)

    assert executor.history.max_len == 4


def test_history_len_not_expanded_for_non_sample_factory_policy(monkeypatch, tmp_path):
    from latency_bench.executors import factory

    noop = Action(value="noop", name="noop", is_noop=True)

    def fake_build_env_adapter(config, noop_action):
        return DummyEnvAdapter(fps_env=30, max_steps=1, noop_action=noop)

    def fake_build_policy(config, action_resolver):
        return ConstantPolicyRunner(noop)

    monkeypatch.setattr(factory, "build_env_adapter", fake_build_env_adapter)
    monkeypatch.setattr(factory, "build_policy", fake_build_policy)
    config = _config("dummy", output_dir=str(tmp_path))
    config["control"]["history_len"] = 1

    executor = build_executor(config)

    assert executor.history.max_len == 1

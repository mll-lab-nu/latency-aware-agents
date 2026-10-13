from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
import pytest

import latency_bench.envs.adapters.gymnasium as gymnasium_adapter_module
from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action, OrderingPolicy
from latency_bench.envs.factory import build_env_adapter
from latency_bench.envs.adapters.gymnasium import GymnasiumEnvAdapter
from latency_bench.envs.base import ENV_RAW_RGB_FRAME_STACK_INFO_KEY
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.latency.samplers import FixedSampler
from latency_bench.policy.base import ConstantPolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import ObservationCadenceDecisionScheduler
from latency_bench.scheduler.persistence import HoldPolicy


class FakeGymEnv:
    def __init__(
        self,
        *,
        step_results: list[tuple[Any, float, bool, bool, dict[str, Any]]] | None = None,
        render_values: list[int] | None = None,
    ) -> None:
        self.action_space = gymnasium.spaces.Dict(
            {"motor": gymnasium.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)}
        )
        self.observation_space = gymnasium.spaces.Dict(
            {"state": gymnasium.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)}
        )
        self.reset_observation = {
            "state": np.asarray([0.25, -0.5], dtype=np.float32),
            "unchanged": object(),
        }
        self.reset_info = {"episode": "reset", "source": object()}
        self.step_results = list(step_results or [])
        self.render_values = list(render_values or [10])
        self.reset_seeds: list[int | None] = []
        self.step_actions: list[Any] = []
        self.render_calls = 0
        self.close_calls = 0
        self._step_index = 0
        self._render_value = self.render_values[0]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        self.reset_seeds.append(seed)
        self._step_index = 0
        self._render_value = self.render_values[0]
        return self.reset_observation, self.reset_info

    def step(self, action: Any):
        self.step_actions.append(action)
        result = self.step_results[self._step_index]
        self._step_index += 1
        self._render_value = self.render_values[self._step_index]
        return result

    def render(self) -> np.ndarray:
        self.render_calls += 1
        return np.full((3, 4, 3), self._render_value, dtype=np.uint8)

    def close(self) -> None:
        self.close_calls += 1


class FakeChannelFirstImageEnv:
    action_space = gymnasium.spaces.Discrete(1)
    observation_space = gymnasium.spaces.Box(
        low=0,
        high=255,
        shape=(1, 2, 2),
        dtype=np.uint8,
    )

    def reset(self, *, seed=None, options=None):
        return np.zeros((1, 2, 2), dtype=np.uint8), {}

    def step(self, action):
        return np.ones((1, 2, 2), dtype=np.uint8), 0.0, False, False, {}

    def render(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def close(self):
        return None


def _install_fake_make(
    monkeypatch: pytest.MonkeyPatch,
    fake_env: FakeGymEnv,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_make(env_id: str, **kwargs: Any) -> FakeGymEnv:
        captured["env_id"] = env_id
        captured["kwargs"] = kwargs
        return fake_env

    monkeypatch.setattr(gymnasium, "make", fake_make)
    return captured


def _make_adapter(
    monkeypatch: pytest.MonkeyPatch,
    fake_env: FakeGymEnv,
    *,
    env_id: str = "Acme/IdentityEnv-v7",
    registration_imports: list[str] | None = None,
    make_kwargs: dict[str, Any] | None = None,
    env_fps: float = 20.0,
    noop_action: Action | None = None,
    frame_stack: int = 1,
    export_env_raw_rgb_frames: bool = False,
) -> tuple[GymnasiumEnvAdapter, dict[str, Any]]:
    captured = _install_fake_make(monkeypatch, fake_env)
    adapter = GymnasiumEnvAdapter(
        env_id=env_id,
        registration_imports=[] if registration_imports is None else registration_imports,
        make_kwargs={} if make_kwargs is None else make_kwargs,
        env_fps=env_fps,
        noop_action=noop_action or Action(value={"motor": np.zeros(2)}, name="noop", is_noop=True),
        frame_stack=frame_stack,
        export_env_raw_rgb_frames=export_env_raw_rgb_frames,
    )
    return adapter, captured


def test_gymnasium_adapter_imports_registration_modules_before_make(monkeypatch):
    fake_env = FakeGymEnv()
    events: list[tuple[str, Any]] = []

    def fake_import_module(module_name: str):
        events.append(("import", module_name))

    def fake_make(env_id: str, **kwargs: Any) -> FakeGymEnv:
        events.append(("make", env_id))
        return fake_env

    monkeypatch.setattr(gymnasium_adapter_module, "import_module", fake_import_module)
    monkeypatch.setattr(gymnasium, "make", fake_make)

    adapter = GymnasiumEnvAdapter(
        env_id="Acme/RegisteredEnv-v1",
        registration_imports=["acme_gym.registration", "project.custom_envs"],
        make_kwargs={"render_mode": "rgb_array"},
        env_fps=20.0,
        noop_action=Action(value=0, name="noop", is_noop=True),
        frame_stack=1,
        export_env_raw_rgb_frames=False,
    )
    try:
        assert events == [
            ("import", "acme_gym.registration"),
            ("import", "project.custom_envs"),
            ("make", "Acme/RegisteredEnv-v1"),
        ]
    finally:
        adapter.close()


def test_gymnasium_adapter_uses_obs_from_dict_state():
    observation = {
        "obs": np.asarray([1.0, 2.0], dtype=np.float32),
        "aux_state": np.zeros(3, dtype=np.float32),
    }

    np.testing.assert_array_equal(
        GymnasiumEnvAdapter._state(observation), observation["obs"]
    )


def test_gymnasium_adapter_passes_env_id_and_make_kwargs_unchanged(monkeypatch):
    fake_env = FakeGymEnv()
    opaque_setting = object()
    make_kwargs = {
        "render_mode": "rgb_array",
        "frameskip": 1,
        "domain_options": {"opaque_setting": opaque_setting},
    }

    adapter, captured = _make_adapter(
        monkeypatch,
        fake_env,
        env_id="Acme/ConfiguredEnv-v3",
        make_kwargs=make_kwargs,
    )
    try:
        assert captured["env_id"] == "Acme/ConfiguredEnv-v3"
        assert captured["kwargs"] == make_kwargs
        assert captured["kwargs"]["domain_options"]["opaque_setting"] is opaque_setting
        assert make_kwargs == {
            "render_mode": "rgb_array",
            "frameskip": 1,
            "domain_options": {"opaque_setting": opaque_setting},
        }
    finally:
        adapter.close()


def test_gymnasium_adapter_reset_preserves_observation_and_seed(monkeypatch):
    fake_env = FakeGymEnv()
    adapter, _ = _make_adapter(monkeypatch, fake_env, env_fps=20.0)
    try:
        observation = adapter.reset(seed=731)

        assert fake_env.reset_seeds == [731]
        assert observation.data is fake_env.reset_observation
        assert observation.env_step == 0
        assert observation.sim_time_ms == 0.0
        assert observation.metadata["episode"] == "reset"
        assert observation.metadata["source"] is fake_env.reset_info["source"]
        assert observation.metadata["env_step"] == 0
        assert observation.metadata["sim_time_ms"] == 0.0
        assert adapter.observe().data is fake_env.reset_observation
    finally:
        adapter.close()


def test_gymnasium_adapter_stacks_channel_first_policy_observations(monkeypatch):
    fake_env = FakeChannelFirstImageEnv()
    monkeypatch.setattr(gymnasium, "make", lambda env_id, **kwargs: fake_env)
    adapter = GymnasiumEnvAdapter(
        env_id="Acme/Image-v0",
        registration_imports=[],
        make_kwargs={},
        env_fps=20.0,
        noop_action=Action(value=0, name="noop", is_noop=True),
        frame_stack=4,
        export_env_raw_rgb_frames=False,
    )

    reset_observation = adapter.reset(seed=3)
    next_observation = adapter.step(Action(value=0, name="noop")).observation

    assert adapter.observation_space.shape == (4, 2, 2)
    np.testing.assert_array_equal(reset_observation.data, np.zeros((4, 2, 2), dtype=np.uint8))
    np.testing.assert_array_equal(next_observation.data[:3], np.zeros((3, 2, 2), dtype=np.uint8))
    np.testing.assert_array_equal(next_observation.data[3], np.ones((2, 2), dtype=np.uint8))


@pytest.mark.parametrize(
    ("terminated", "truncated"),
    [(True, False), (False, True)],
)
def test_gymnasium_adapter_step_preserves_gym_semantics(
    monkeypatch,
    terminated: bool,
    truncated: bool,
):
    step_observation = {
        "state": np.asarray([0.75, 0.5], dtype=np.float32),
        "unchanged": object(),
    }
    info_marker = object()
    fake_env = FakeGymEnv(
        step_results=[
            (
                step_observation,
                2.5,
                terminated,
                truncated,
                {"phase": "step", "marker": info_marker},
            )
        ],
        render_values=[10, 20],
    )
    adapter, _ = _make_adapter(monkeypatch, fake_env, env_fps=20.0)
    action_value = {"motor": np.asarray([0.3, -0.4], dtype=np.float32)}
    action = Action(value=action_value, name="steer")
    try:
        adapter.reset(seed=5)
        result = adapter.step(action)

        assert len(fake_env.step_actions) == 1
        assert fake_env.step_actions[0] is action_value
        assert result.observation is not None
        assert result.observation.data is step_observation
        assert result.reward == 2.5
        assert result.done is terminated
        assert result.truncated is truncated
        assert result.info["phase"] == "step"
        assert result.info["marker"] is info_marker
        assert result.info["env_step"] == 1
        assert result.info["sim_time_ms"] == 50.0
        assert result.observation.env_step == 1
        assert result.observation.sim_time_ms == 50.0
        assert result.observation.metadata["env_step"] == 1
        assert result.observation.metadata["sim_time_ms"] == 50.0
        assert adapter.observe().data is step_observation
    finally:
        adapter.close()


def test_gymnasium_adapter_render_game_frame_uses_gym_render(monkeypatch):
    step_observation = {"state": np.asarray([1.0, 1.0], dtype=np.float32)}
    fake_env = FakeGymEnv(
        step_results=[(step_observation, 0.0, False, False, {})],
        render_values=[10, 42],
    )
    adapter, _ = _make_adapter(monkeypatch, fake_env)
    try:
        adapter.reset(seed=0)
        adapter.step(Action(value={"motor": np.zeros(2)}, name="noop", is_noop=True))

        frame = adapter.render_game_frame()

        assert fake_env.render_calls == 1
        assert frame.shape == (3, 4, 3)
        assert np.all(frame == 42)
    finally:
        adapter.close()


def test_gymnasium_adapter_raw_rgb_stack_is_oldest_to_newest(monkeypatch):
    first_observation = {"state": np.asarray([1.0, 0.0], dtype=np.float32)}
    second_observation = {"state": np.asarray([2.0, 0.0], dtype=np.float32)}
    fake_env = FakeGymEnv(
        step_results=[
            (first_observation, 0.0, False, False, {"phase": 1}),
            (second_observation, 0.0, False, False, {"phase": 2}),
        ],
        render_values=[10, 20, 30],
    )
    adapter, _ = _make_adapter(
        monkeypatch,
        fake_env,
        frame_stack=3,
        export_env_raw_rgb_frames=True,
    )
    try:
        reset_observation = adapter.reset(seed=0)
        first_result = adapter.step(Action(value={"motor": np.zeros(2)}, name="noop"))
        second_result = adapter.step(Action(value={"motor": np.ones(2)}, name="move"))

        reset_stack = reset_observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY]
        assert first_result.observation is not None
        first_stack = first_result.observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY]
        assert second_result.observation is not None
        second_stack = second_result.observation.metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY]
        observed_stack = adapter.observe().metadata[ENV_RAW_RGB_FRAME_STACK_INFO_KEY]

        assert reset_stack.shape == (3, 3, 4, 3)
        assert reset_stack[:, 0, 0, 0].tolist() == [10, 10, 10]
        assert first_stack[:, 0, 0, 0].tolist() == [10, 10, 20]
        assert second_stack[:, 0, 0, 0].tolist() == [10, 20, 30]
        assert observed_stack[:, 0, 0, 0].tolist() == [10, 20, 30]
        assert second_result.observation.data is second_observation
    finally:
        adapter.close()


def test_gymnasium_adapter_close_delegates_to_gym_env(monkeypatch):
    fake_env = FakeGymEnv()
    adapter, _ = _make_adapter(monkeypatch, fake_env)

    adapter.close()

    assert fake_env.close_calls == 1


def test_factory_builds_gymnasium_adapter_without_rewriting_config(monkeypatch):
    from latency_bench.envs import factory
    monkeypatch.setattr(factory, "_ENV_ADAPTER_BUILDERS", dict(factory._ENV_ADAPTER_BUILDERS))
    factory.register_env_adapter("test_gymnasium", env_id="Acme/FactoryEnv-v2")(factory._build_gymnasium_env)
    fake_env = FakeGymEnv()
    captured = _install_fake_make(monkeypatch, fake_env)
    opaque_setting = object()
    make_kwargs = {
        "render_mode": "rgb_array",
        "frameskip": 1,
        "domain_options": {"opaque_setting": opaque_setting},
    }
    config = {
        'experiment': {
            'name': 'generic_gym_test',
            'seed': 17,
        },
        'executor': {
            'mode': 'simulated',
        },
        'env': {
            'name': 'test_gymnasium',
            'env_id': 'Acme/FactoryEnv-v2',
            'registration_imports': [],
            'make_kwargs': make_kwargs,
            'env_fps': 47.5,
            'obs_fps': 9.5,
            'frame_stack': 4,
            'export_env_raw_rgb_frames': True,
            'noop_action': 'NOOP',
            'action_map': {
                'NOOP': {
                    'motor': [0.0, 0.0],
                },
                'MOVE': {
                    'motor': [0.5, -0.5],
                },
            },
        },
        'latency': {
            'method': 'zero',
        },
        'scheduler': {
            'hold_policy': 'hold',
            'ordering_policy': 'issue_order_fifo',
        },
        'policy': {
            'type': 'constant',
            'action': 'NOOP',
        },
        'evaluation': {
            'episodes': 1,
            'max_env_steps': 3,
        },
        'logging': {
            'output_dir': 'outputs/generic_gym_test',
            'save_step_records': True,
            'save_action_records': True,
            'save_latency_records': True,
        },
    }
    noop_action = Action(
        value={"motor": np.asarray([0.0, 0.0], dtype=np.float32)},
        name="NOOP",
        is_noop=True,
    )

    adapter = build_env_adapter(config, noop_action)
    try:
        assert isinstance(adapter, GymnasiumEnvAdapter)
        assert captured["env_id"] == "Acme/FactoryEnv-v2"
        assert captured["kwargs"] == make_kwargs
        assert captured["kwargs"]["domain_options"]["opaque_setting"] is opaque_setting
        assert adapter.env_fps == 47.5
        assert adapter.frame_stack == 4
        assert adapter.noop_action is noop_action
        assert adapter.export_env_raw_rgb_frames is True
        assert config["env"]["make_kwargs"] is make_kwargs
        assert config["env"]["obs_fps"] == 9.5
        assert config["env"]["action_map"]["MOVE"] == {"motor": [0.5, -0.5]}
    finally:
        adapter.close()


def test_gymnasium_adapter_runs_on_existing_raw_frame_latency_timeline(monkeypatch):
    step_results = [
        ({"state": np.asarray([step], dtype=np.float32)}, 0.0, False, False, {})
        for step in range(1, 5)
    ]
    fake_env = FakeGymEnv(
        step_results=step_results,
        render_values=[10, 20, 30, 40, 50],
    )
    noop_action = Action(value=0, name="noop", is_noop=True)
    control_action = Action(value=1, name="control")
    adapter, _ = _make_adapter(
        monkeypatch,
        fake_env,
        env_fps=10.0,
        noop_action=noop_action,
    )
    clock = EnvClock(env_fps=10.0, obs_fps=10.0)
    executor = SimulatedLatencyExecutor(
        env=adapter,
        policy=ConstantPolicyRunner(control_action),
        decision_scheduler=ObservationCadenceDecisionScheduler(clock),
        latency_source=FixedSampler(fixed_latency_ms=200.0, clock=clock),
        action_scheduler=ActionScheduler(
            clock=clock,
            noop_action=noop_action,
            persistence_policy=HoldPolicy(),
            ordering_policy=OrderingPolicy.ISSUE_ORDER_FIFO,
        ),
        clock=clock,
    )
    try:
        executor.run_episode(episode_id=0, seed=19, max_env_steps=4)

        assert fake_env.step_actions == [0, 0, 1, 1]
    finally:
        adapter.close()


def test_gymnasium_adapter_zero_latency_matches_direct_gym_trajectory():
    actions = [0, 2, 2, 1, 0, 2]
    direct_env = gymnasium.make("MountainCar-v0")
    adapter = GymnasiumEnvAdapter(
        env_id="MountainCar-v0",
        registration_imports=[],
        make_kwargs={},
        env_fps=30.0,
        noop_action=Action(value=1, name="coast", is_noop=True),
        frame_stack=1,
        export_env_raw_rgb_frames=False,
        include_state=True,
    )
    try:
        direct_observation, direct_info = direct_env.reset(seed=91)
        adapted_observation = adapter.reset(seed=91)

        np.testing.assert_array_equal(adapted_observation.data, direct_observation)
        np.testing.assert_array_equal(
            adapted_observation.metadata["gymnasium_state"], direct_observation
        )
        assert adapted_observation.metadata.items() >= direct_info.items()

        for action_id in actions:
            direct_observation, direct_reward, direct_terminated, direct_truncated, direct_info = (
                direct_env.step(action_id)
            )
            adapted = adapter.step(Action(value=action_id, name=str(action_id)))

            assert adapted.observation is not None
            np.testing.assert_array_equal(adapted.observation.data, direct_observation)
            np.testing.assert_array_equal(
                adapted.observation.metadata["gymnasium_state"], direct_observation
            )
            assert adapted.reward == direct_reward
            assert adapted.done is direct_terminated
            assert adapted.truncated is direct_truncated
            for key, value in direct_info.items():
                assert adapted.info[key] == value
    finally:
        direct_env.close()
        adapter.close()

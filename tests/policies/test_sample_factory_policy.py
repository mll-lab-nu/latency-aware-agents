from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Tuple

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.envs.base import EnvAdapter
from latency_bench.envs.factory import register_env_adapter
from latency_bench.integrations.sample_factory import actions as sf_actions
from latency_bench.integrations.sample_factory import runtime as sf_runtime
from latency_bench.policy import sample_factory as sample_factory_policy
from latency_bench.policy.model_source import model_file


def test_sample_factory_integration_builds_policy_config_with_eval_parser():
    calls = []
    parsed_cfg = object()
    module = SimpleNamespace(
        build_cli_args_from_config=lambda config: ["--experiment", config["experiment"]],
        SAMPLE_FACTORY_CONFIG_PARSER=SimpleNamespace(
            parse_eval=lambda argv: calls.append(argv) or parsed_cfg
        ),
    )
    integration = sample_factory_policy.SampleFactoryIntegrationModule(module, "register_components")

    result = integration.build_sf_cfg_from_config({"experiment": "teacher"})

    assert result is parsed_cfg
    assert calls == [["--experiment", "teacher"]]


@pytest.mark.parametrize(
    ("env_name", "integration_module_name", "register_attr"),
    [
        ("flappy", "training.flappy_sf.integration", "register_flappy_envs"),
        ("demon_attack", "training.demon_attack_sf.integration", "register_demon_attack_envs"),
        (
            "deadly_corridor",
            "training.deadly_corridor_sf.integration",
            "register_deadly_corridor_components",
        ),
    ],
)
def test_sample_factory_policy_builder_selects_environment_integration(
    monkeypatch,
    env_name,
    integration_module_name,
    register_attr,
):
    integration = ModuleType(integration_module_name)
    registered = []
    setattr(integration, register_attr, lambda: registered.append(register_attr))
    imported = []
    monkeypatch.setattr(
        sample_factory_policy,
        "import_module",
        lambda module_name: imported.append(module_name) or integration,
    )
    monkeypatch.setattr(sample_factory_policy, "SampleFactoryPolicyRunner", lambda **kwargs: kwargs)

    config = {"env": {"name": env_name}, "policy": {"type": "sample_factory"}}
    action_resolver = object()
    policy = sample_factory_policy.build_sample_factory_policy(
        config,
        action_resolver,
    )
    policy["integration_module"].register_components()

    assert imported == [integration_module_name]
    assert registered == [register_attr]
    assert policy["config"] is config
    assert policy["action_resolver"] is action_resolver
    assert policy["integration_module"].module is integration


def test_resolve_torch_device_maps_legacy_gpu_to_cuda():
    assert sf_runtime.resolve_torch_device("gpu") == "cuda"
    assert sf_runtime.resolve_torch_device("cpu") == "cpu"


def _old_obs_batch_to_sf_inputs_reference(observations, device):
    if not observations:
        return {}, None

    batched_parts: dict[str, list[torch.Tensor]] = {}
    expected_keys: set[str] | None = None
    action_masks: list[torch.Tensor | None] = []
    for index, observation in enumerate(observations):
        obs_dict, action_mask = sample_factory_policy._obs_to_sf_inputs(observation, device)
        keys = set(obs_dict)
        if expected_keys is None:
            expected_keys = keys
            for key in keys:
                batched_parts[key] = []
        elif keys != expected_keys:
            raise ValueError(
                "Cannot batch Sample Factory observations with different keys: "
                f"expected {sorted(expected_keys)}, got {sorted(keys)} at batch index {index}"
            )
        for key, tensor in obs_dict.items():
            batched_parts[key].append(tensor)
        action_masks.append(action_mask)

    batched_obs = {
        key: torch.cat(tensors, dim=0)
        for key, tensors in batched_parts.items()
    }
    if all(mask is not None for mask in action_masks):
        return batched_obs, torch.cat([mask for mask in action_masks if mask is not None], dim=0)
    return batched_obs, None


def _assert_sf_input_batches_equal(actual, expected):
    actual_obs, actual_mask = actual
    expected_obs, expected_mask = expected
    assert set(actual_obs) == set(expected_obs)
    for key in expected_obs:
        assert actual_obs[key].shape == expected_obs[key].shape
        assert actual_obs[key].dtype == expected_obs[key].dtype
        assert actual_obs[key].device == expected_obs[key].device
        assert torch.equal(actual_obs[key], expected_obs[key])
    if expected_mask is None:
        assert actual_mask is None
    else:
        assert actual_mask is not None
        assert actual_mask.shape == expected_mask.shape
        assert actual_mask.dtype == expected_mask.dtype
        assert actual_mask.device == expected_mask.device
        assert torch.equal(actual_mask, expected_mask)


def _install_fake_sample_factory_modules(
    monkeypatch,
    *,
    load_from_checkpoint,
    actor_critic_factory=None,
    argmax_actions=None,
    preprocess_actions=None,
    get_rnn_size=None,
):
    def package(name: str) -> ModuleType:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    for name in (
        "sample_factory",
        "sample_factory.algo",
        "sample_factory.algo.sampling",
        "sample_factory.algo.utils",
        "sample_factory.model",
        "sample_factory.cfg",
    ):
        package(name)

    batched_sampling = ModuleType("sample_factory.algo.sampling.batched_sampling")
    batched_sampling.preprocess_actions = preprocess_actions or (lambda env_info, actions: actions)
    monkeypatch.setitem(sys.modules, batched_sampling.__name__, batched_sampling)

    action_distributions = ModuleType("sample_factory.algo.utils.action_distributions")
    action_distributions.argmax_actions = argmax_actions or (lambda action_distribution: 0)
    monkeypatch.setitem(sys.modules, action_distributions.__name__, action_distributions)

    env_info = ModuleType("sample_factory.algo.utils.env_info")
    env_info.extract_env_info = lambda reference_env, cfg: SimpleNamespace()
    monkeypatch.setitem(sys.modules, env_info.__name__, env_info)

    rl_utils = ModuleType("sample_factory.algo.utils.rl_utils")
    rl_utils.prepare_and_normalize_obs = lambda actor_critic, obs: obs
    monkeypatch.setitem(sys.modules, rl_utils.__name__, rl_utils)

    tensor_utils = ModuleType("sample_factory.algo.utils.tensor_utils")
    tensor_utils.unsqueeze_tensor = (
        lambda tensor, dim: tensor.unsqueeze(dim)
        if isinstance(tensor, torch.Tensor)
        else np.expand_dims(tensor, dim)
    )
    monkeypatch.setitem(sys.modules, tensor_utils.__name__, tensor_utils)

    class FakeActorCritic:
        def eval(self):
            pass

        def model_to_device(self, device):
            pass

    actor_critic = ModuleType("sample_factory.model.actor_critic")
    actor_critic.create_actor_critic = actor_critic_factory or (
        lambda cfg, observation_space, action_space: FakeActorCritic()
    )
    monkeypatch.setitem(sys.modules, actor_critic.__name__, actor_critic)

    model_utils = ModuleType("sample_factory.model.model_utils")
    model_utils.get_rnn_size = get_rnn_size or (lambda cfg: 1)
    monkeypatch.setitem(sys.modules, model_utils.__name__, model_utils)

    arguments = ModuleType("sample_factory.cfg.arguments")
    arguments.load_from_checkpoint = load_from_checkpoint
    monkeypatch.setitem(sys.modules, arguments.__name__, arguments)


def _config(checkpoint_path: str) -> dict:
    return {
        'env': {
            'name': 'dummy',
            'noop_action': 'noop',
            'action_map': {
                'noop': 0,
            },
        },
        'policy': {
            'type': 'sample_factory',
            'deterministic': True,
            'checkpoint': str(Path(checkpoint_path).absolute()).lstrip('/') if checkpoint_path else None,
            'source': 'local',
            'root': '/',
        },
        'executor': {
            'inference_devices': ['cpu'],
        },
    }


class _FakeReferenceEnv:
    def __init__(self, *, action_space=None, observation_space=object()):
        self.observation_space = observation_space
        self.action_space = action_space or Discrete(1)

    def close(self):
        pass


class _FakeIntegration:
    def __init__(
        self,
        *,
        action_space=None,
        observation_space=object(),
        use_rnn: bool = False,
        action_chunk_horizon: int = 1,
        algo: str = "APPO",
    ):
        self.action_space = action_space or Discrete(1)
        self.observation_space = observation_space
        self.use_rnn = bool(use_rnn)
        self.action_chunk_horizon = action_chunk_horizon
        self.algo = algo

    def register_components(self):
        pass

    def build_sf_cfg_from_config(self, config):
        return SimpleNamespace(
            algo=self.algo,
            policy_index=0,
            load_checkpoint_kind="latest",
            use_rnn=self.use_rnn,
            fasttd3_action_chunk_horizon=self.action_chunk_horizon,
        )

    def make_reference_env(self, cfg):
        return _FakeReferenceEnv(
            action_space=self.action_space,
            observation_space=self.observation_space,
        )


class _BatchDiscreteActorCritic:
    def __init__(self):
        self.forward_batch_sizes: list[int] = []
        self.last_argmax_actions = torch.empty(0, dtype=torch.long)

    def eval(self):
        pass

    def model_to_device(self, device):
        self.device = device

    def __call__(self, normalized_obs, rnn_states):
        obs = normalized_obs["obs"]
        batch_size = int(obs.shape[0])
        self.forward_batch_sizes.append(batch_size)
        first_values = obs.reshape(batch_size, -1)[:, 0].to(torch.long)
        self.last_argmax_actions = torch.remainder(first_values + 1, 5)
        return {
            "actions": torch.full((batch_size,), 99, device=obs.device),
            "new_rnn_states": rnn_states,
        }

    def action_distribution(self):
        return SimpleNamespace(argmax_actions=self.last_argmax_actions)


class _ChunkActorCritic:
    def __init__(self, action_dim: int):
        self.actions = torch.linspace(-0.8, 0.8, action_dim).reshape(1, action_dim)
        self.observations = []

    def eval(self):
        pass

    def model_to_device(self, device):
        self.actions = self.actions.to(device)

    def __call__(self, normalized_obs, rnn_states):
        self.observations.append(normalized_obs)
        return {
            "actions": self.actions.expand(normalized_obs["obs"].shape[0], -1),
            "new_rnn_states": rnn_states,
        }


class _StateEchoActorCritic:
    def __init__(self):
        self.seen_rnn_states: list[torch.Tensor] = []
        self.last_argmax_actions = torch.empty(0, dtype=torch.long)

    def eval(self):
        pass

    def model_to_device(self, device):
        self.device = device

    def __call__(self, normalized_obs, rnn_states):
        batch_size = int(normalized_obs["obs"].shape[0])
        self.seen_rnn_states.append(rnn_states.detach().cpu().clone())
        self.last_argmax_actions = torch.remainder(rnn_states[:, 0].to(torch.long) + 1, 5)
        return {
            "actions": torch.full((batch_size,), 99, device=normalized_obs["obs"].device),
            "new_rnn_states": rnn_states + 1.0,
        }

    def action_distribution(self):
        return SimpleNamespace(argmax_actions=self.last_argmax_actions)


class _BatchTupleActionDistribution:
    def __init__(self, component_actions: torch.Tensor):
        self.argmax_actions = component_actions
        self.component_actions = component_actions
        self.distributions = [
            SimpleNamespace(argmax_actions=component_actions[:, index])
            for index in range(component_actions.shape[1])
        ]


class _BatchTupleActorCritic:
    def __init__(self, component_actions: list[list[int]]):
        self.forward_batch_sizes: list[int] = []
        self.component_actions = torch.as_tensor(component_actions, dtype=torch.long)
        self.last_component_actions = torch.empty((0, self.component_actions.shape[1]), dtype=torch.long)

    def eval(self):
        pass

    def model_to_device(self, device):
        self.device = device

    def __call__(self, normalized_obs, rnn_states):
        obs = normalized_obs["obs"]
        batch_size = int(obs.shape[0])
        self.forward_batch_sizes.append(batch_size)
        actions = self.component_actions[:batch_size].to(obs.device)
        self.last_component_actions = actions
        return {
            "actions": actions,
            "new_rnn_states": rnn_states,
        }

    def action_distribution(self):
        return _BatchTupleActionDistribution(self.last_component_actions)


def _discrete_preprocess_actions(_env_info, actions):
    assert tuple(actions.shape) == (int(actions.shape[0]), 1)
    return actions.detach().cpu().numpy().squeeze(axis=1)


def _tuple_preprocess_actions(env_info, actions):
    if isinstance(env_info.action_space, Tuple):
        num_components = len(env_info.action_space.spaces)
    else:
        num_components = len(env_info.action_space.nvec)
    assert tuple(actions.shape) == (int(actions.shape[0]), num_components)
    return actions.detach().cpu().numpy()


def _build_discrete_runner(monkeypatch, *, action_space=None, use_rnn: bool = False):
    actor = _BatchDiscreteActorCritic()
    action_space = action_space or Discrete(5)
    _install_fake_sample_factory_modules(
        monkeypatch,
        load_from_checkpoint=lambda cfg: cfg,
        actor_critic_factory=lambda cfg, observation_space, action_space: actor,
        argmax_actions=lambda action_distribution: action_distribution.argmax_actions,
        preprocess_actions=_discrete_preprocess_actions,
    )
    monkeypatch.setattr(
        sf_runtime,
        "load_sample_factory_policy_state_dict",
        lambda cfg, actor_critic, device, checkpoint_path: ({"train_step": 123}, Path("/tmp/model.pth")),
    )
    config = _config("/tmp/model.pth")
    runner = sample_factory_policy.SampleFactoryPolicyRunner(
        config=config,
        action_resolver=ActionResolver(config["env"]),
        integration_module=_FakeIntegration(
            action_space=action_space,
            observation_space=Box(low=0, high=255, shape=(3, 2, 2), dtype=np.uint8),
            use_rnn=use_rnn,
        ),
    )
    return runner, actor


def _build_state_echo_discrete_runner(monkeypatch, *, use_rnn: bool):
    actor = _StateEchoActorCritic()
    _install_fake_sample_factory_modules(
        monkeypatch,
        load_from_checkpoint=lambda cfg: cfg,
        actor_critic_factory=lambda cfg, observation_space, action_space: actor,
        argmax_actions=lambda action_distribution: action_distribution.argmax_actions,
        preprocess_actions=_discrete_preprocess_actions,
        get_rnn_size=lambda cfg: 2,
    )
    monkeypatch.setattr(
        sf_runtime,
        "load_sample_factory_policy_state_dict",
        lambda cfg, actor_critic, device, checkpoint_path: ({"train_step": 123}, Path("/tmp/model.pth")),
    )
    config = _config("/tmp/model.pth")
    runner = sample_factory_policy.SampleFactoryPolicyRunner(
        config=config,
        action_resolver=ActionResolver(config["env"]),
        integration_module=_FakeIntegration(
            action_space=Discrete(5),
            observation_space=Box(low=0, high=255, shape=(3, 2, 2), dtype=np.uint8),
            use_rnn=use_rnn,
        ),
    )
    return runner, actor


def _build_tuple_runner(
    monkeypatch,
    *,
    action_space=None,
    component_actions: list[list[int]] | None = None,
    use_rnn: bool = False,
):
    action_space = action_space or Tuple((Discrete(3), Discrete(3), Discrete(3), Discrete(2)))
    component_actions = component_actions or [
        [0, 1, 2, 1],
        [1, 2, 0, 0],
        [2, 0, 1, 1],
        [1, 1, 1, 0],
    ]
    actor = _BatchTupleActorCritic(component_actions)

    def fake_extract_env_info(reference_env, cfg):
        space = reference_env.action_space
        if isinstance(space, Tuple):
            return SimpleNamespace(
                action_space=space,
                all_discrete=all(isinstance(item, Discrete) for item in space.spaces),
                action_splits=[1 for _item in space.spaces],
                gpu_actions=False,
            )
        if isinstance(space, MultiDiscrete):
            return SimpleNamespace(
                action_space=space,
                all_discrete=False,
                action_splits=None,
                gpu_actions=False,
            )
        return SimpleNamespace(
            action_space=space,
            all_discrete=False,
            action_splits=None,
            gpu_actions=False,
        )

    _install_fake_sample_factory_modules(
        monkeypatch,
        load_from_checkpoint=lambda cfg: cfg,
        actor_critic_factory=lambda cfg, observation_space, action_space: actor,
        argmax_actions=lambda action_distribution: action_distribution.argmax_actions,
        preprocess_actions=_tuple_preprocess_actions,
    )
    monkeypatch.setattr(sys.modules["sample_factory.algo.utils.env_info"], "extract_env_info", fake_extract_env_info)
    monkeypatch.setattr(
        sf_runtime,
        "load_sample_factory_policy_state_dict",
        lambda cfg, actor_critic, device, checkpoint_path: ({"train_step": 123}, Path("/tmp/model.pth")),
    )
    config = _config("/tmp/model.pth")
    config["policy"]["type"] = "sample_factory"
    config["env"]["noop_action"] = [0, 0, 0, 0]
    runner = sample_factory_policy.SampleFactoryPolicyRunner(
        config=config,
        action_resolver=ActionResolver(config["env"]),
        integration_module=_FakeIntegration(
            action_space=action_space,
            observation_space=Box(low=0, high=255, shape=(12, 2, 2), dtype=np.uint8),
            use_rnn=use_rnn,
        ),
    )
    return runner, actor


def test_deterministic_actions_for_tuple_space_stacks_component_argmaxes():
    action_space = Tuple((Discrete(3), Discrete(3), Discrete(3), Discrete(2)))
    component_actions = torch.tensor(
        [
            [0, 1, 2, 1],
            [1, 2, 0, 0],
        ],
        dtype=torch.long,
    )
    action_distribution = SimpleNamespace(
        argmax_actions=component_actions.reshape(1, -1),
        distributions=[
            SimpleNamespace(argmax_actions=component_actions[:, component_index])
            for component_index in range(component_actions.shape[1])
        ],
    )

    actions = sf_actions.deterministic_actions_for_space(
        action_distribution,
        action_space=action_space,
        argmax_actions=lambda distribution: distribution.argmax_actions,
        device=torch.device("cpu"),
    )

    assert tuple(actions.shape) == (2, 4)
    assert actions.tolist() == component_actions.tolist()


def test_deterministic_actions_for_discrete_space_keeps_argmax_output():
    action_distribution = SimpleNamespace(argmax_actions=torch.tensor([1, 0], dtype=torch.long))

    actions = sf_actions.deterministic_actions_for_space(
        action_distribution,
        action_space=Discrete(2),
        argmax_actions=lambda distribution: distribution.argmax_actions,
        device=torch.device("cpu"),
    )

    assert actions.tolist() == [1, 0]


class _SfDiscreteImageEnv(EnvAdapter):
    action_space = Discrete(5)

    def __init__(self, *, max_steps: int, noop_action: Action):
        self.env_fps = 30.0
        self.max_steps = int(max_steps)
        self.noop_action = noop_action
        self.gym_env = self
        self.env_step = 0

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        return self._observation(seed=seed)

    def step(self, action: Action) -> StepResult:
        self.env_step += 1
        return StepResult(
            observation=self._observation(),
            reward=1.0,
            done=self.env_step >= self.max_steps,
            truncated=False,
            info={"applied_action": action.value},
        )

    def render_frame(self):
        return self._frame()

    def get_state_info(self):
        return {"env_step": self.env_step, "sim_time_ms": self.env_step * (1000.0 / self.env_fps)}

    def _observation(self, *, seed: int | None = None) -> Observation:
        metadata = self.get_state_info()
        if seed is not None:
            metadata["seed"] = seed
        return Observation(data=self._frame(), env_step=self.env_step, sim_time_ms=metadata["sim_time_ms"], metadata=metadata)

    def _frame(self):
        return np.full((3, 2, 2), self.env_step, dtype=np.uint8)


class _SfTupleImageEnv(EnvAdapter):
    action_space = Tuple((Discrete(3), Discrete(3), Discrete(3), Discrete(2)))

    def __init__(self, *, max_steps: int, noop_action: Action):
        self.env_fps = 35.0
        self.max_steps = int(max_steps)
        self.noop_action = noop_action
        self.gym_env = self
        self.env_step = 0

    def reset(self, seed: int | None = None) -> Observation:
        self.env_step = 0
        return self._observation(seed=seed)

    def step(self, action: Action) -> StepResult:
        assert isinstance(action.value, tuple)
        assert len(action.value) == 4
        self.env_step += 1
        return StepResult(
            observation=self._observation(),
            reward=1.0,
            done=self.env_step >= self.max_steps,
            truncated=False,
            info={"applied_action": action.value},
        )

    def render_frame(self):
        return self._frame()

    def get_state_info(self):
        return {"env_step": self.env_step, "sim_time_ms": self.env_step * (1000.0 / self.env_fps)}

    def _observation(self, *, seed: int | None = None) -> Observation:
        metadata = self.get_state_info()
        if seed is not None:
            metadata["seed"] = seed
        return Observation(data=self._frame(), env_step=self.env_step, sim_time_ms=metadata["sim_time_ms"], metadata=metadata)

    def _frame(self):
        return np.full((12, 2, 2), self.env_step, dtype=np.uint8)


@register_env_adapter("sf_discrete_image_policy_test")
def _build_sf_discrete_image_policy_test_env(config: dict, noop_action: Action) -> EnvAdapter:
    return _SfDiscreteImageEnv(
        max_steps=int(config["evaluation"]["max_env_steps"]),
        noop_action=noop_action,
    )


@register_env_adapter("sf_tuple_image_policy_test")
def _build_sf_tuple_image_policy_test_env(config: dict, noop_action: Action) -> EnvAdapter:
    return _SfTupleImageEnv(
        max_steps=int(config["evaluation"]["max_env_steps"]),
        noop_action=noop_action,
    )


def test_sample_factory_policy_loads_config_from_checkpoint_when_checkpoint_path_is_empty(
    monkeypatch,
):
    load_calls = []
    state_loads = []

    def fake_load_from_checkpoint(cfg):
        load_calls.append(cfg)
        cfg.loaded_from_checkpoint = True
        return cfg

    def fake_load_state_dict(cfg, actor_critic, device, checkpoint_path):
        state_loads.append((getattr(cfg, "loaded_from_checkpoint", False), checkpoint_path))
        return {"train_step": 123}, Path("/tmp/best_123.pth")

    _install_fake_sample_factory_modules(
        monkeypatch,
        load_from_checkpoint=fake_load_from_checkpoint,
    )
    monkeypatch.setattr(
        sf_runtime,
        "load_sample_factory_policy_state_dict",
        fake_load_state_dict,
    )

    config = _config("")
    runner = sample_factory_policy.SampleFactoryPolicyRunner(
        config=config,
        action_resolver=ActionResolver(config["env"]),
        integration_module=_FakeIntegration(),
    )

    assert len(load_calls) == 1
    assert state_loads == [(True, None)]
    assert runner.checkpoint_global_step == 123
    assert runner.checkpoint_path == "/tmp/best_123.pth"
    assert model_file(config["policy"], "checkpoint") == "/tmp/best_123.pth"


def test_sample_factory_policy_skips_config_checkpoint_load_for_explicit_checkpoint_path(
    monkeypatch,
):
    load_calls = []
    state_loads = []

    def fake_load_from_checkpoint(cfg):
        load_calls.append(cfg)
        return cfg

    def fake_load_state_dict(cfg, actor_critic, device, checkpoint_path):
        state_loads.append((getattr(cfg, "loaded_from_checkpoint", False), checkpoint_path))
        return {"train_step": 123}, Path("/tmp/model.pth")

    _install_fake_sample_factory_modules(
        monkeypatch,
        load_from_checkpoint=fake_load_from_checkpoint,
    )
    monkeypatch.setattr(
        sf_runtime,
        "load_sample_factory_policy_state_dict",
        fake_load_state_dict,
    )

    runner = sample_factory_policy.SampleFactoryPolicyRunner(
        config=_config("/tmp/model.pth"),
        action_resolver=ActionResolver(_config("/tmp/model.pth")["env"]),
        integration_module=_FakeIntegration(),
    )

    assert load_calls == []
    assert state_loads == [(False, "/tmp/model.pth")]
    assert runner.checkpoint_global_step == 123


def test_sample_factory_policy_unsqueezes_discrete_argmax_actions_before_preprocess(
    monkeypatch,
):
    from latency_bench.core.types import Observation

    observed_shapes = []

    _install_fake_sample_factory_modules(
        monkeypatch,
        load_from_checkpoint=lambda cfg: cfg,
    )
    monkeypatch.setattr(
        sf_runtime,
        "load_sample_factory_policy_state_dict",
        lambda cfg, actor_critic, device, checkpoint_path: ({"train_step": 123}, Path("/tmp/model.pth")),
    )

    runner = sample_factory_policy.SampleFactoryPolicyRunner(
        config=_config("/tmp/model.pth"),
        action_resolver=ActionResolver(_config("/tmp/model.pth")["env"]),
        integration_module=_FakeIntegration(),
    )

    class FakePredictActorCritic:
        def __call__(self, normalized_obs, rnn_states):
            return {
                "actions": torch.tensor([99], device=runner.device),
                "new_rnn_states": rnn_states,
            }

        def action_distribution(self):
            return object()

    def fake_preprocess_actions(env_info, actions):
        observed_shapes.append(tuple(actions.shape))
        assert tuple(actions.shape) == (1, 1)
        return actions

    runner.actor_critic = FakePredictActorCritic()
    runner._argmax_actions = lambda action_distribution: torch.tensor([4], device=runner.device)
    runner._preprocess_actions = fake_preprocess_actions

    output = runner.predict(
        Observation(
            data=torch.zeros(1),
            env_step=1,
            sim_time_ms=0.0,
        )
    )

    assert observed_shapes == [(1, 1)]
    assert output.raw_output == 4
    assert output.action.value == 4


@pytest.mark.parametrize("obs_dim,action_dim,horizon,bound", [(198, 43, 8, 1.0), (3, 1, 8, 2.0), (3, 1, 1, 2.0)])
def test_sample_factory_policy_emits_chunk_from_physical_observation(monkeypatch, obs_dim, action_dim, horizon, bound):
    actor = _ChunkActorCritic(horizon * action_dim)
    _install_fake_sample_factory_modules(
        monkeypatch,
        load_from_checkpoint=lambda cfg: cfg,
        actor_critic_factory=lambda cfg, observation_space, action_space: actor,
    )
    monkeypatch.setattr(
        sf_runtime,
        "load_sample_factory_policy_state_dict",
        lambda cfg, actor_critic, device, checkpoint_path: ({"train_step": 123}, Path("/tmp/model.pth")),
    )
    config = _config("/tmp/model.pth")
    config["policy"]["deterministic"] = False
    from latency_bench.envs import factory
    monkeypatch.setattr(factory, "_ENV_ADAPTER_BUILDERS", dict(factory._ENV_ADAPTER_BUILDERS))
    factory.register_env_adapter("chunk_box_policy")(factory._build_gymnasium_env)
    config["env"]["name"] = "chunk_box_policy"
    config["env"]["noop_action"] = [0.0] * action_dim
    del config["env"]["action_map"]
    config["env"]["action_space"] = {"type": "box", "labels": [f"a{i}" for i in range(action_dim)], "low": [-bound] * action_dim, "high": [bound] * action_dim}
    action_space = Box(-1.0, 1.0, (horizon * action_dim,), dtype=np.float32)
    runner = sample_factory_policy.SampleFactoryPolicyRunner(
        config=config,
        action_resolver=ActionResolver(config["env"]),
        integration_module=_FakeIntegration(
            action_space=action_space,
            observation_space=Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32),
            action_chunk_horizon=horizon, algo="FAST_TD3",
        ),
    )
    observation = Observation(data=np.zeros(obs_dim, dtype=np.float32), env_step=3, sim_time_ms=60.0)
    output = runner.predict(observation)

    if horizon > 1:
        assert len(output.action_chunk) == horizon
        assert output.action is output.action_chunk[0]
    else:
        assert output.action_chunk is None
    np.testing.assert_allclose(output.raw_output, actor.actions.reshape(horizon, action_dim) if horizon > 1 else actor.actions.reshape(-1))
    native = output.action_chunk if horizon > 1 else [output.action]
    np.testing.assert_allclose([a.value for a in native], actor.actions.reshape(horizon, action_dim) * bound)
    np.testing.assert_array_equal(actor.observations[0]["obs"], np.zeros((1, obs_dim)))
    batch = runner.predict_batch([observation, observation])
    assert len(batch) == 2
    np.testing.assert_array_equal(actor.observations[1]["obs"], np.zeros((2, obs_dim)))
    assert all((len(item.action_chunk) == horizon if horizon > 1 else item.action_chunk is None) for item in batch)
    assert all(item.raw_output == output.raw_output for item in batch)


def test_sample_factory_predict_batch_size_1_matches_predict_for_discrete_policy(monkeypatch):
    runner, actor = _build_discrete_runner(monkeypatch)
    observation = Observation(
        data=np.full((3, 2, 2), 2, dtype=np.uint8),
        env_step=0,
        sim_time_ms=0.0,
    )

    single_output = runner.predict(observation)
    batch_outputs = runner.predict_batch([observation])

    assert len(batch_outputs) == 1
    assert batch_outputs[0].raw_output == single_output.raw_output
    assert batch_outputs[0].action.value == single_output.action.value
    assert actor.forward_batch_sizes == [1, 1]


def test_sample_factory_predict_batch_runs_one_forward_for_four_discrete_observations(monkeypatch):
    runner, actor = _build_discrete_runner(monkeypatch)
    observations = [
        Observation(
            data=np.full((3, 2, 2), value, dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
        )
        for value in range(4)
    ]

    outputs = runner.predict_batch(observations)

    assert actor.forward_batch_sizes == [4]
    assert [output.raw_output for output in outputs] == [1, 2, 3, 4]
    assert [output.action.value for output in outputs] == [1, 2, 3, 4]
    assert [output.metadata["policy_type"] for output in outputs] == ["sample_factory"] * 4
    assert [output.metadata["checkpoint_path"] for output in outputs] == ["/tmp/model.pth"] * 4
    assert [output.metadata["checkpoint_global_step"] for output in outputs] == [123] * 4


def test_obs_batch_to_sf_inputs_stacks_homogeneous_numpy_chw_before_tensor_conversion(monkeypatch):
    observations = [
        Observation(
            data=np.full((3, 2, 2), value, dtype=np.uint8),
            env_step=value,
            sim_time_ms=float(value),
        )
        for value in range(4)
    ]
    expected = _old_obs_batch_to_sf_inputs_reference(observations, torch.device("cpu"))
    original_as_tensor = torch.as_tensor
    converted_shapes = []

    def tracking_as_tensor(value, *args, **kwargs):
        converted_shapes.append(tuple(np.asarray(value).shape) if not isinstance(value, torch.Tensor) else tuple(value.shape))
        return original_as_tensor(value, *args, **kwargs)

    monkeypatch.setattr(sample_factory_policy.torch, "as_tensor", tracking_as_tensor)

    actual = sample_factory_policy._obs_batch_to_sf_inputs(observations, torch.device("cpu"))

    _assert_sf_input_batches_equal(actual, expected)
    assert converted_shapes == [(4, 3, 2, 2)]


def test_obs_batch_to_sf_inputs_batches_homogeneous_torch_chw_without_numpy_conversion(monkeypatch):
    observations = [
        Observation(
            data=torch.full((3, 2, 2), value, dtype=torch.uint8),
            env_step=value,
            sim_time_ms=float(value),
        )
        for value in range(4)
    ]
    expected = _old_obs_batch_to_sf_inputs_reference(observations, torch.device("cpu"))

    def fail_asarray(value, *args, **kwargs):
        if isinstance(value, torch.Tensor):
            raise AssertionError("torch tensor observations must not be converted through numpy")
        return np.asarray(value, *args, **kwargs)

    monkeypatch.setattr(sample_factory_policy.np, "asarray", fail_asarray)

    actual = sample_factory_policy._obs_batch_to_sf_inputs(observations, torch.device("cpu"))

    _assert_sf_input_batches_equal(actual, expected)


@pytest.mark.parametrize(
    "observations",
    [
        [
            Observation(
                data=np.full((2, 2, 3), value, dtype=np.uint8),
                env_step=value,
                sim_time_ms=float(value),
            )
            for value in range(3)
        ],
        [
            Observation(
                data={
                    "obs": np.full((3, 2, 2), value, dtype=np.uint8),
                    "extra": np.asarray([value, value + 1], dtype=np.float32),
                },
                env_step=value,
                sim_time_ms=float(value),
            )
            for value in range(3)
        ],
        [
            Observation(
                data={
                    "obs": np.full((3, 2, 2), value, dtype=np.uint8),
                    "action_mask": np.asarray([1, value % 2], dtype=np.float32),
                },
                env_step=value,
                sim_time_ms=float(value),
            )
            for value in range(3)
        ],
        [
            Observation(
                data=torch.full((3, 2, 2), value, dtype=torch.uint8),
                env_step=value,
                sim_time_ms=float(value),
            )
            for value in range(3)
        ],
        [
            Observation(
                data=np.full((3, 2, 2), value, dtype=np.uint8 if value % 2 == 0 else np.float32),
                env_step=value,
                sim_time_ms=float(value),
            )
            for value in range(3)
        ],
    ],
)
def test_obs_batch_to_sf_inputs_matches_old_tensor_semantics(observations):
    device = torch.device("cpu")
    expected = _old_obs_batch_to_sf_inputs_reference(observations, device)
    actual = sample_factory_policy._obs_batch_to_sf_inputs(observations, device)

    _assert_sf_input_batches_equal(actual, expected)


def test_sample_factory_single_predict_does_not_carry_state_for_non_recurrent_policy(monkeypatch):
    runner, actor = _build_state_echo_discrete_runner(monkeypatch, use_rnn=False)
    observation = Observation(
        data=np.zeros((3, 2, 2), dtype=np.uint8),
        env_step=1,
        sim_time_ms=0.0,
    )

    first = runner.predict(observation)
    second = runner.predict(observation)

    assert first.raw_output == 1
    assert second.raw_output == 1
    assert torch.equal(actor.seen_rnn_states[0], torch.zeros((1, 2)))
    assert torch.equal(actor.seen_rnn_states[1], torch.zeros((1, 2)))


def test_sample_factory_predict_batch_tuple_n1_matches_single_predict_format(monkeypatch):
    runner, actor = _build_tuple_runner(monkeypatch, component_actions=[[1, 2, 0, 1]])
    observation = Observation(
        data=np.full((12, 2, 2), 1, dtype=np.uint8),
        env_step=0,
        sim_time_ms=0.0,
    )

    single_output = runner.predict(observation)
    batch_outputs = runner.predict_batch([observation])

    assert actor.forward_batch_sizes == [1, 1]
    assert single_output.raw_output == [1, 2, 0, 1]
    assert single_output.action.value == (1, 2, 0, 1)
    assert batch_outputs[0].raw_output == [1, 2, 0, 1]
    assert batch_outputs[0].action.value == single_output.action.value


def test_sample_factory_predict_batch_tuple_returns_one_tuple_action_per_slot(monkeypatch):
    runner, actor = _build_tuple_runner(monkeypatch)
    observations = [
        Observation(
            data=np.full((12, 2, 2), value, dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
        )
        for value in range(4)
    ]

    outputs = runner.predict_batch(observations)

    assert actor.forward_batch_sizes == [4]
    assert [output.raw_output for output in outputs] == [
        [0, 1, 2, 1],
        [1, 2, 0, 0],
        [2, 0, 1, 1],
        [1, 1, 1, 0],
    ]
    assert [output.action.value for output in outputs] == [
        (0, 1, 2, 1),
        (1, 2, 0, 0),
        (2, 0, 1, 1),
        (1, 1, 1, 0),
    ]
    for output in outputs:
        turn, move, strafe, attack = output.action.value
        assert 0 <= turn < 3
        assert 0 <= move < 3
        assert 0 <= strafe < 3
        assert 0 <= attack < 2


def test_sample_factory_predict_batch_multidiscrete_returns_tuple_per_slot(monkeypatch):
    runner, _actor = _build_tuple_runner(
        monkeypatch,
        action_space=MultiDiscrete([3, 3, 3, 2]),
    )

    outputs = runner.predict_batch([
        Observation(
            data=np.full((12, 2, 2), value, dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
        )
        for value in range(4)
    ])

    assert [output.action.value for output in outputs] == [
        (0, 1, 2, 1),
        (1, 2, 0, 0),
        (2, 0, 1, 1),
        (1, 1, 1, 0),
    ]


@pytest.mark.parametrize(
    "action_space",
    [
        Tuple((Tuple((Discrete(2), Discrete(2))), Discrete(2))),
        object(),
    ],
)
def test_sample_factory_predict_batch_rejects_unsupported_composite_spaces(monkeypatch, action_space):
    runner, _actor = _build_tuple_runner(monkeypatch, action_space=action_space)

    with pytest.raises(NotImplementedError, match="Discrete|Tuple|MultiDiscrete|Phase 3D"):
        runner.predict_batch([
            Observation(
                data=np.zeros((12, 2, 2), dtype=np.uint8),
                env_step=0,
                sim_time_ms=0.0,
            )
        ])


def test_latency_bench_run_with_batched_sample_factory_discrete_policy_completes_exact_episodes(
    monkeypatch,
    tmp_path,
):
    import json

    import latency_bench.executors.factory as executor_factory
    import latency_bench.eval.driver as evaluation

    runner, _actor = _build_discrete_runner(monkeypatch)
    monkeypatch.setattr(executor_factory, "build_policy", lambda config, action_resolver: runner)
    config = {
        'experiment': {
            'name': 'sf_discrete_batched_run',
            'seed': 0,
        },
        'executor': {
            'mode': 'simulated',
            'parallel_envs': 4,
        },
        'env': {
            'name': 'sf_discrete_image_policy_test',
            'env_fps': 30,
            'obs_fps': 30,
            'frame_stack': 1,
            'noop_action': 'noop',
            'action_map': {
                'noop': 0,
                'jump': 1,
                'two': 2,
                'three': 3,
                'four': 4,
            },
        },
        'latency': {
            'method': 'zero',
        },
        'scheduler': {
            'hold_policy': 'one_frame_then_noop',
            'ordering_policy': 'latest_ready',
        },
        'policy': {
            'type': 'sample_factory',
            'checkpoint': 'model.pth',
            'source': 'local',
            'root': str(tmp_path),
        },
        'logging': {
            'output_dir': str(tmp_path),
            'save_step_records': True,
            'save_action_records': True,
            'save_latency_records': True,
        },
        'evaluation': {
            'episodes': 5,
            'max_env_steps': 1,
        },
    }

    (tmp_path / "model.pth").write_bytes(b"model")
    metrics = evaluation.run_from_config(config)

    assert [metric.episode_id for metric in metrics] == [0, 1, 2, 3, 4]
    action_records = [
        json.loads(line)
        for line in (tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    latency_records = [
        json.loads(line)
        for line in (tmp_path / "latencies.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len({record["action_id"] for record in action_records}) == len(action_records)
    assert len({record["latency"]["obs_id"] for record in action_records}) == len(action_records)
    assert sorted(record["latency"]["obs_id"] for record in action_records) == sorted(
        record["obs_id"] for record in latency_records
    )
    assert {"action_id", "action", "ready_time_ms", "latency", "metadata"} <= set(action_records[0])
    assert {"obs_id", "latency_ms", "metadata"} <= set(action_records[0]["latency"])


def test_latency_bench_run_with_batched_deadly_tuple_policy_completes_exact_episodes(
    monkeypatch,
    tmp_path,
):
    import json

    import latency_bench.executors.factory as executor_factory
    import latency_bench.eval.driver as evaluation

    runner, _actor = _build_tuple_runner(monkeypatch)
    monkeypatch.setattr(executor_factory, "build_policy", lambda config, action_resolver: runner)
    config = {
        'experiment': {
            'name': 'sf_tuple_batched_run',
            'seed': 0,
        },
        'executor': {
            'mode': 'simulated',
            'parallel_envs': 4,
        },
        'env': {
            'name': 'sf_tuple_image_policy_test',
            'env_id': 'doom_deadly_corridor',
            'env_fps': 35,
            'obs_fps': 35,
            'frame_stack': 4,
            'noop_action': [0, 0, 0, 0],
        },
        'latency': {
            'method': 'zero',
        },
        'scheduler': {
            'hold_policy': 'hold',
            'ordering_policy': 'latest_ready',
        },
        'policy': {
            'type': 'sample_factory',
            'checkpoint': 'model.pth',
            'source': 'local',
            'root': str(tmp_path),
        },
        'logging': {
            'output_dir': str(tmp_path),
            'save_step_records': True,
            'save_action_records': True,
            'save_latency_records': True,
        },
        'evaluation': {
            'episodes': 5,
            'max_env_steps': 1,
        },
    }

    (tmp_path / "model.pth").write_bytes(b"model")
    metrics = evaluation.run_from_config(config)

    assert [metric.episode_id for metric in metrics] == [0, 1, 2, 3, 4]
    action_records = [
        json.loads(line)
        for line in (tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    latency_records = [
        json.loads(line)
        for line in (tmp_path / "latencies.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len({record["action_id"] for record in action_records}) == len(action_records)
    assert len({record["latency"]["obs_id"] for record in action_records}) == len(action_records)
    assert sorted(record["latency"]["obs_id"] for record in action_records) == sorted(
        record["obs_id"] for record in latency_records
    )
    assert isinstance(action_records[0]["action"]["value"], list)
    assert len(action_records[0]["action"]["value"]) == 4

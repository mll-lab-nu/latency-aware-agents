from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from latency_bench.core.actions import ActionResolver
from latency_bench.policy import sample_factory_common


def _install_fake_sample_factory_modules(monkeypatch, *, load_from_checkpoint):
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
    batched_sampling.preprocess_actions = lambda env_info, actions: actions
    monkeypatch.setitem(sys.modules, batched_sampling.__name__, batched_sampling)

    action_distributions = ModuleType("sample_factory.algo.utils.action_distributions")
    action_distributions.argmax_actions = lambda action_distribution: 0
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
    actor_critic.create_actor_critic = (
        lambda cfg, observation_space, action_space: FakeActorCritic()
    )
    monkeypatch.setitem(sys.modules, actor_critic.__name__, actor_critic)

    model_utils = ModuleType("sample_factory.model.model_utils")
    model_utils.get_rnn_size = lambda cfg: 1
    monkeypatch.setitem(sys.modules, model_utils.__name__, model_utils)

    arguments = ModuleType("sample_factory.cfg.arguments")
    arguments.load_from_checkpoint = load_from_checkpoint
    monkeypatch.setitem(sys.modules, arguments.__name__, arguments)


def _config(checkpoint_path: str) -> dict:
    return {
        "env": {
            "name": "dummy",
            "noop_action": "noop",
            "action_map": {
                "noop": 0,
            },
        },
        "policy": {
            "type": "flappy_sf",
            "checkpoint_path": checkpoint_path,
            "device": "cpu",
            "deterministic": True,
        },
    }


class _FakeReferenceEnv:
    observation_space = object()
    action_space = object()

    def close(self):
        pass


class _FakeIntegration:
    def register_components(self):
        pass

    def build_sf_cfg_from_config(self, config):
        return SimpleNamespace(policy_index=0, load_checkpoint_kind="latest")

    def make_reference_env(self, cfg):
        return _FakeReferenceEnv()


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
        return {"train_step": 123}

    _install_fake_sample_factory_modules(
        monkeypatch,
        load_from_checkpoint=fake_load_from_checkpoint,
    )
    monkeypatch.setattr(
        sample_factory_common,
        "_load_sf_policy_state_dict",
        fake_load_state_dict,
    )

    runner = sample_factory_common.SampleFactoryPolicyRunner(
        config=_config(""),
        action_resolver=ActionResolver(_config("")["env"]),
        integration_module=_FakeIntegration(),
        policy_type="flappy_sf",
    )

    assert len(load_calls) == 1
    assert state_loads == [(True, None)]
    assert runner.checkpoint_global_step == 123


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
        return {"train_step": 123}

    _install_fake_sample_factory_modules(
        monkeypatch,
        load_from_checkpoint=fake_load_from_checkpoint,
    )
    monkeypatch.setattr(
        sample_factory_common,
        "_load_sf_policy_state_dict",
        fake_load_state_dict,
    )

    runner = sample_factory_common.SampleFactoryPolicyRunner(
        config=_config("/tmp/model.pth"),
        action_resolver=ActionResolver(_config("/tmp/model.pth")["env"]),
        integration_module=_FakeIntegration(),
        policy_type="flappy_sf",
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
        sample_factory_common,
        "_load_sf_policy_state_dict",
        lambda cfg, actor_critic, device, checkpoint_path: {"train_step": 123},
    )

    runner = sample_factory_common.SampleFactoryPolicyRunner(
        config=_config("/tmp/model.pth"),
        action_resolver=ActionResolver(_config("/tmp/model.pth")["env"]),
        integration_module=_FakeIntegration(),
        policy_type="flappy_sf",
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


def test_sample_factory_checkpoint_loader_rejects_flappy_ppo_checkpoint(tmp_path):
    path = tmp_path / "checkpoint_000000001_000000000128.pth"
    torch.save({"backend_type": "flappy_ppo", "format": "flappy_ppo", "agent": {}}, path)

    with pytest.raises(ValueError, match="flappy_ppo"):
        sample_factory_common._checkpoint_state_dict_from_path(path, torch.device("cpu"))

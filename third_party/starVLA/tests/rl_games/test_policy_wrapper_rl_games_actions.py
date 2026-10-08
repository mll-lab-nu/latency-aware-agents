from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


STARVLA_ROOT = Path(__file__).resolve().parents[2]
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))


def _load_policy_wrapper_module():
    module_names = [
        "starVLA.model.framework.base_framework",
        "starVLA.model.framework.share_tools",
        "deployment.model_server.policy_norm_processor",
        "torch",
    ]
    original_modules = {name: sys.modules[name] for name in module_names if name in sys.modules}
    torch_module = ModuleType("torch")
    torch_module.bfloat16 = object()
    base_framework_module = ModuleType("starVLA.model.framework.base_framework")
    base_framework_module.baseframework = SimpleNamespace()
    share_tools_module = ModuleType("starVLA.model.framework.share_tools")
    share_tools_module.read_mode_config = lambda ckpt_path: ({}, {})
    norm_processor_module = ModuleType("deployment.model_server.policy_norm_processor")
    norm_processor_module.PolicyNormProcessor = object
    sys.modules["starVLA.model.framework.base_framework"] = base_framework_module
    sys.modules["starVLA.model.framework.share_tools"] = share_tools_module
    sys.modules["deployment.model_server.policy_norm_processor"] = norm_processor_module
    sys.modules["torch"] = torch_module

    module_path = STARVLA_ROOT / "deployment/model_server/policy_wrapper.py"
    spec = importlib.util.spec_from_file_location("policy_wrapper_module", module_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name in module_names:
            if name in original_modules:
                sys.modules[name] = original_modules[name]
            else:
                sys.modules.pop(name, None)
    return module


class FakeProcessor:
    def __init__(self):
        self.calls = []
        self.state_calls = []
        self.unnorm_key = "new_embodiment"
        self.available_unnorm_keys = ["new_embodiment"]
        self.action_keys = ["action"]
        self.state_keys = []

    def unapply_actions(self, values):
        self.calls.append(np.asarray(values).copy())
        return np.asarray(values) + 10.0

    def apply_state(self, values):
        self.state_calls.append(np.asarray(values).copy())
        return np.asarray(values) + 1.0


class FakeFramework:
    def __init__(self, normalized_actions):
        self.normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        self.calls = []
        self.to_calls = []
        self.eval_calls = 0

    def to(self, target):
        self.to_calls.append(target)
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def predict_action(self, examples, **kwargs):
        self.calls.append({"examples": examples, "kwargs": kwargs})
        return {"normalized_actions": self.normalized_actions}


class FakeProfiler:
    def __init__(self):
        self.timings = {}

    def time(self, stage):
        @contextmanager
        def timer():
            yield
            self.timings[stage] = 1.0

        return timer()


def test_decode_flappy_logits_to_discrete_action_payload():
    from deployment.model_server.rl_games_action_decode import decode_rl_games_actions

    decoded = decode_rl_games_actions(
        normalized_actions=np.asarray([[[0.1, 0.9], [0.8, 0.2]]], dtype=np.float32),
        env_name="flappy",
    )

    assert decoded["action_output_type"] == "rl_games_discrete_id"
    assert decoded["actions"].tolist() == [[[1], [0]]]
    assert np.allclose(decoded["raw_action_scores"], [[[0.1, 0.9], [0.8, 0.2]]])


def test_decode_demon_attack_logits_to_discrete_action_payload():
    from deployment.model_server.rl_games_action_decode import decode_rl_games_actions

    decoded = decode_rl_games_actions(
        normalized_actions=np.asarray([[[0.0, 0.1, 3.2, -0.4, 0.2, 0.0]]], dtype=np.float32),
        env_name="demon_attack",
    )

    assert decoded["action_output_type"] == "rl_games_discrete_id"
    assert decoded["actions"].tolist() == [[[2]]]


def test_decode_gymnasium_logits_uses_explicit_action_env_dim():
    from deployment.model_server.rl_games_action_decode import decode_rl_games_actions

    raw = np.zeros((1, 1, 11), dtype=np.float32)
    raw[0, 0, 8] = 1.0
    raw[0, 0, 9:] = 99.0
    decoded = decode_rl_games_actions(
        normalized_actions=raw,
        env_name="gymnasium",
        action_env_dim=9,
    )

    assert decoded["action_output_type"] == "rl_games_discrete_id"
    assert decoded["actions"].tolist() == [[[8]]]
    assert np.array_equal(decoded["raw_action_scores"], raw)


def test_decode_gymnasium_requires_explicit_action_env_dim():
    from deployment.model_server.rl_games_action_decode import decode_rl_games_actions

    with pytest.raises(TypeError):
        decode_rl_games_actions(
            normalized_actions=np.asarray([[[0.1, 0.8, 0.2]]], dtype=np.float32),
            env_name="gymnasium",
        )
def test_decode_gymnasium_box_uses_env_actions_and_keeps_normalized_scores():
    from deployment.model_server.rl_games_action_decode import decode_rl_games_actions

    decoded = decode_rl_games_actions(
        normalized_actions=np.asarray([[[0.1, 0.2, 0.3]]], dtype=np.float32),
        action_values=np.asarray([[[10.0, 20.0, 30.0]]], dtype=np.float32),
        env_name="gymnasium",
        action_env_dim=2,
        gymnasium_action_space_type="box",
    )

    assert decoded["action_output_type"] == "rl_games_continuous"
    assert decoded["actions"].tolist() == [[[10.0, 20.0]]]
    assert np.allclose(decoded["raw_action_scores"], [[[0.1, 0.2, 0.3]]])


def test_decode_deadly_corridor_factorized_logits_to_tuple_payload():
    from deployment.model_server.rl_games_action_decode import decode_rl_games_actions

    decoded = decode_rl_games_actions(
        normalized_actions=np.asarray(
            [[[0.0, 0.2, 0.9, 0.8, 0.1, 0.2, 0.1, 0.7, 0.0, 0.3, 0.9]]],
            dtype=np.float32,
        ),
        env_name="deadly_corridor",
    )

    assert decoded["action_output_type"] == "rl_games_deadly_corridor_tuple"
    assert decoded["actions"].tolist() == [[[2, 0, 1, 1]]]


def test_decode_deadly_corridor_joint_logits_to_tuple_payload():
    from deployment.model_server.rl_games_action_decode import decode_rl_games_actions

    raw = np.zeros((1, 1, 54), dtype=np.float32)
    raw[0, 0, 11] = 1.0
    decoded = decode_rl_games_actions(normalized_actions=raw, env_name="deadly_corridor")

    assert decoded["action_output_type"] == "rl_games_deadly_corridor_tuple"
    assert decoded["actions"].tolist() == [[[0, 1, 2, 1]]]


def test_decode_deadly_corridor_multibinary_logits_independently():
    from deployment.model_server.rl_games_action_decode import decode_rl_games_actions

    decoded = decode_rl_games_actions(
        normalized_actions=np.asarray(
            [[[-0.2, 0.8, 0.7, 0.1, 0.9, -0.4, 0.6]]],
            dtype=np.float32,
        ),
        env_name="deadly_corridor",
        deadly_action_layout="multibinary_7",
        deadly_multibinary_threshold=0.5,
    )

    assert decoded["action_output_type"] == "rl_games_deadly_corridor_multibinary"
    assert decoded["actions"].tolist() == [[[0, 1, 1, 0, 1, 0, 1]]]


def test_resolve_deadly_corridor_multibinary_threshold_from_model_config():
    from deployment.model_server.rl_games_action_decode import resolve_deadly_action_decode_spec

    discrete_ce_config = {
        "framework": {"action_model": {"loss_type": "discrete_ce"}},
        "rl_games": {
            "env_eval": {
                "deadly": {
                    "action_layout": "multibinary_7",
                    "multibinary_threshold": None,
                }
            }
        },
    }
    bce_config = {
        "framework": {"action_model": {"loss_type": "multibinary_bce"}},
        "rl_games": {
            "env_eval": {
                "deadly": {
                    "action_layout": "multibinary_7",
                    "multibinary_threshold": None,
                }
            }
        },
    }
    current_bce_config = {
        "framework": {"action_model": {"loss_type": "current_multibinary_bce"}},
        "rl_games": {
            "env_eval": {
                "deadly": {
                    "action_layout": "multibinary_7",
                    "multibinary_threshold": None,
                }
            }
        },
    }

    assert resolve_deadly_action_decode_spec(discrete_ce_config) == ("multibinary_7", 0.5)
    assert resolve_deadly_action_decode_spec(bce_config) == ("multibinary_7", 0.0)
    assert resolve_deadly_action_decode_spec(current_bce_config) == ("multibinary_7", 0.0)


def test_resolve_historical_deadly_corridor_action_layout_metadata():
    from deployment.model_server.rl_games_action_decode import resolve_deadly_action_decode_spec

    model_config = {
        "framework": {
            "action_model": {
                "action_layout": "deadly_corridor_multibinary_7",
                "loss_type": "discrete_ce",
            }
        }
    }

    assert resolve_deadly_action_decode_spec(model_config) == ("multibinary_7", 0.5)
    assert resolve_deadly_action_decode_spec(
        model_config,
        action_layout="factorized_11",
        multibinary_threshold=0.25,
    ) == ("factorized_11", 0.25)


def test_policy_wrapper_default_mode_keeps_unnormalized_actions():
    policy_wrapper_module = _load_policy_wrapper_module()
    processor = FakeProcessor()
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = FakeFramework([[[1.0, 2.0]]])
    wrapper._default_unnorm_key = "new_embodiment"
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._action_output_mode = "deployment"
    wrapper._rl_games_env_name = None
    wrapper._get_processor = lambda unnorm_key: processor

    prediction = wrapper.predict_action(examples=[{"image": [], "lang": ""}], unnorm_key="new_embodiment")

    assert prediction["actions"].tolist() == [[[11.0, 12.0]]]
    assert len(processor.calls) == 1


def test_policy_wrapper_rl_games_mode_returns_decoded_actions_without_unapply():
    policy_wrapper_module = _load_policy_wrapper_module()
    processor = FakeProcessor()
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = FakeFramework([[[0.1, 0.9]]])
    wrapper._default_unnorm_key = "new_embodiment"
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._action_output_mode = "rl_games"
    wrapper._rl_games_env_name = "flappy"
    wrapper._get_processor = lambda unnorm_key: processor

    prediction = wrapper.predict_action(examples=[{"image": [], "lang": ""}], unnorm_key="new_embodiment")

    assert prediction["actions"].tolist() == [[[1]]]
    assert np.allclose(prediction["raw_action_scores"], [[[0.1, 0.9]]])
    assert prediction["action_output_type"] == "rl_games_discrete_id"
    assert processor.calls == []


def test_policy_wrapper_gymnasium_mode_decodes_only_action_env_dim_logits():
    policy_wrapper_module = _load_policy_wrapper_module()
    processor = FakeProcessor()
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = FakeFramework([[[0.1, 0.8, 0.2, 99.0]]])
    wrapper._default_unnorm_key = "new_embodiment"
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._action_output_mode = "rl_games"
    wrapper._rl_games_env_name = "gymnasium"
    wrapper._rl_games_action_env_dim = 3
    wrapper._get_processor = lambda unnorm_key: processor

    prediction = wrapper.predict_action(
        examples=[{"image": [], "lang": ""}],
        unnorm_key="new_embodiment",
    )

    assert prediction["actions"].tolist() == [[[1]]]
    assert prediction["action_output_type"] == "rl_games_discrete_id"
    assert processor.calls == []


def test_policy_wrapper_gymnasium_box_normalizes_state_and_unnormalizes_actions():
    policy_wrapper_module = _load_policy_wrapper_module()
    processor = FakeProcessor()
    framework = FakeFramework([[[0.1, 0.2, 0.3]]])
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = framework
    wrapper._default_unnorm_key = "new_embodiment"
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._action_output_mode = "rl_games"
    wrapper._rl_games_env_name = "gymnasium"
    wrapper._rl_games_action_env_dim = 2
    wrapper._rl_games_gymnasium_action_space_type = "box"
    wrapper._get_processor = lambda unnorm_key: processor

    prediction = wrapper.predict_action(
        examples=[{"image": [], "lang": "", "state": np.asarray([[1.0, 2.0]])}],
        unnorm_key="new_embodiment",
    )

    assert framework.calls[0]["examples"][0]["state"].tolist() == [[2.0, 3.0]]
    assert processor.state_calls[0].tolist() == [[1.0, 2.0]]
    assert np.allclose(prediction["actions"], [[[10.1, 10.2]]])
    assert np.allclose(prediction["raw_action_scores"], [[[0.1, 0.2, 0.3]]])
    assert prediction["action_output_type"] == "rl_games_continuous"


def test_policy_wrapper_gymnasium_stateless_example_skips_state_normalization():
    policy_wrapper_module = _load_policy_wrapper_module()
    processor = FakeProcessor()
    framework = FakeFramework([[[0.1, 0.9]]])
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = framework
    wrapper._default_unnorm_key = "new_embodiment"
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._action_output_mode = "rl_games"
    wrapper._rl_games_env_name = "gymnasium"
    wrapper._rl_games_action_env_dim = 2
    wrapper._rl_games_gymnasium_action_space_type = "discrete"
    wrapper._get_processor = lambda unnorm_key: processor
    example = {"image": [], "lang": ""}

    prediction = wrapper.predict_action(
        examples=[example],
        unnorm_key="new_embodiment",
    )

    assert framework.calls[0]["examples"] == [example]
    assert processor.state_calls == []
    assert prediction["actions"].tolist() == [[[1]]]


def test_policy_wrapper_gymnasium_uses_registered_generic_data_config():
    policy_wrapper_module = _load_policy_wrapper_module()
    calls = []

    class CapturingProcessor:
        def __init__(self, ckpt_path, **kwargs):
            calls.append((ckpt_path, kwargs))

    policy_wrapper_module.PolicyNormProcessor = CapturingProcessor
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(
        policy_wrapper_module.PolicyServerWrapper
    )
    wrapper._ckpt_path = "checkpoint.safetensors"
    wrapper._rl_games_env_name = "gymnasium"
    wrapper._norm_processors = {}

    wrapper._get_processor("new_embodiment")

    assert calls == [
        (
            "checkpoint.safetensors",
            {
                "unnorm_key": "new_embodiment",
                "robot_type": "rl_games_gymnasium",
            },
        )
    ]


def test_policy_wrapper_gymnasium_uses_registered_native_data_config():
    policy_wrapper_module = _load_policy_wrapper_module()
    calls = []

    class CapturingProcessor:
        def __init__(self, ckpt_path, **kwargs):
            calls.append(kwargs)

    policy_wrapper_module.PolicyNormProcessor = CapturingProcessor
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(
        policy_wrapper_module.PolicyServerWrapper
    )
    wrapper._ckpt_path = "checkpoint.safetensors"
    wrapper._rl_games_env_name = "gymnasium"
    wrapper._rl_games_gymnasium_robot_type = "rl_games_gymnasium_native"
    wrapper._norm_processors = {}

    wrapper._get_processor("new_embodiment")

    assert calls == [
        {
            "unnorm_key": "new_embodiment",
            "robot_type": "rl_games_gymnasium_native",
        }
    ]


def test_policy_wrapper_rl_games_mode_uses_deadly_multibinary_layout():
    policy_wrapper_module = _load_policy_wrapper_module()
    processor = FakeProcessor()
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = FakeFramework([[[-0.2, 0.8, 0.7, 0.1, 0.9, -0.4, 0.6]]])
    wrapper._default_unnorm_key = "new_embodiment"
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._action_output_mode = "rl_games"
    wrapper._rl_games_env_name = "deadly_corridor"
    wrapper._rl_games_action_layout = "multibinary_7"
    wrapper._rl_games_multibinary_threshold = 0.5
    wrapper._get_processor = lambda unnorm_key: processor

    prediction = wrapper.predict_action(
        examples=[{"image": [], "lang": ""}],
        unnorm_key="new_embodiment",
    )

    assert prediction["actions"].tolist() == [[[0, 1, 1, 0, 1, 0, 1]]]
    assert prediction["action_output_type"] == "rl_games_deadly_corridor_multibinary"
    assert processor.calls == []


def test_policy_wrapper_default_mode_records_unnormalize_timing():
    policy_wrapper_module = _load_policy_wrapper_module()
    processor = FakeProcessor()
    framework = FakeFramework([[[1.0, 2.0]]])
    profiler = FakeProfiler()
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = framework
    wrapper._default_unnorm_key = "new_embodiment"
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._action_output_mode = "deployment"
    wrapper._rl_games_env_name = None
    wrapper._get_processor = lambda unnorm_key: processor

    prediction = wrapper.predict_action(
        examples=[{"image": [], "lang": ""}],
        unnorm_key="new_embodiment",
        profiler=profiler,
    )

    assert prediction["actions"].tolist() == [[[11.0, 12.0]]]
    assert framework.calls[0]["kwargs"]["profiler"] is profiler
    assert profiler.timings["starvla_wrapper_unnormalize_ms"] == 1.0


def test_policy_wrapper_rl_games_mode_records_decode_timing():
    policy_wrapper_module = _load_policy_wrapper_module()
    processor = FakeProcessor()
    framework = FakeFramework([[[0.1, 0.9]]])
    profiler = FakeProfiler()
    wrapper = policy_wrapper_module.PolicyServerWrapper.__new__(policy_wrapper_module.PolicyServerWrapper)
    wrapper._framework = framework
    wrapper._default_unnorm_key = "new_embodiment"
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._action_output_mode = "rl_games"
    wrapper._rl_games_env_name = "flappy"
    wrapper._get_processor = lambda unnorm_key: processor

    prediction = wrapper.predict_action(
        examples=[{"image": [], "lang": ""}],
        unnorm_key="new_embodiment",
        profiler=profiler,
    )

    assert prediction["actions"].tolist() == [[[1]]]
    assert framework.calls[0]["kwargs"]["profiler"] is profiler
    assert profiler.timings["starvla_wrapper_rl_games_decode_ms"] == 1.0
    assert processor.calls == []


def test_policy_wrapper_loads_checkpoint_framework_in_eval_mode():
    policy_wrapper_module = _load_policy_wrapper_module()
    framework = FakeFramework([[[0.1, 0.9]]])
    policy_wrapper_module.baseframework.from_pretrained = lambda ckpt_path: framework
    policy_wrapper_module.read_mode_config = lambda ckpt_path: (
        {"framework": {"action_model": {"action_horizon": 1}}},
        {},
    )

    wrapper = policy_wrapper_module.PolicyServerWrapper(
        ckpt_path="/tmp/checkpoint",
        device="cuda",
        use_bf16=True,
        action_output_mode="rl_games",
        rl_games_env_name="flappy",
    )

    assert wrapper._framework is framework
    assert framework.eval_calls == 1

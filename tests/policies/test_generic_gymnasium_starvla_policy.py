from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Observation
from latency_bench.envs.gymnasium_task import gymnasium_task_contract
from latency_bench.policy.factory import build_policy
from latency_bench.policy.starvla_prompts import resolve_starvla_prompt


class GenericPolicyServerWrapper:
    instances: list["GenericPolicyServerWrapper"] = []

    def __init__(
        self,
        ckpt_path,
        device="cuda",
        use_bf16=False,
        unnorm_key=None,
        action_output_mode=None,
        rl_games_env_name=None,
        rl_games_action_layout=None,
        rl_games_multibinary_threshold=None,
        rl_games_action_env_dim=None,
    ):
        self.rl_games_env_name = rl_games_env_name
        self.rl_games_action_env_dim = rl_games_action_env_dim
        self.calls = []
        self._model_cfg = {
            "datasets": {"vla_data": {"include_state": False}},
            "framework": {"action_model": {}},
        }
        GenericPolicyServerWrapper.instances.append(self)

    def predict_action(self, examples, unnorm_key=None, **kwargs):
        self.calls.append({"examples": examples, "unnorm_key": unnorm_key, "kwargs": kwargs})
        return {
            "actions": np.asarray([[[0]]] * len(examples), dtype=np.int64),
            "raw_action_scores": np.asarray(
                [[[0.9, 0.1, 0.0]]] * len(examples), dtype=np.float32
            ),
            "action_output_type": "rl_games_discrete_id",
        }


class GenericLiveFramework:
    def __init__(self):
        self.calls = []

    def predict_action(self, examples, **kwargs):
        self.calls.append({"examples": examples, "kwargs": kwargs})
        return {
            "normalized_actions": np.asarray(
                [[[0.9, 0.1, 0.0]]] * len(examples), dtype=np.float32
            )
        }


def _generic_config(tmp_path: Path) -> dict:
    env = {
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
        "action_order": ["push_right", "coast", "push_left"],
        "base_prompt": "Drive the car up the mountain.",
    }
    contract = gymnasium_task_contract(env)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "integration_name": "gymnasium",
                "active_action_dim": 3,
                "gymnasium_task": contract,
            }
        ),
        encoding="utf-8",
    )
    model_config_path = tmp_path / "config.full.yaml"
    model_config_path.write_text(
        json.dumps(
            {
                "rl_games": {"gymnasium": {"task_contract": contract}},
                "datasets": {
                    "vla_data": {
                        "include_state": False,
                        "gymnasium_task_contract": contract,
                    }
                },
                "framework": {"action_model": {"action_env_dim": 3}},
            }
        ),
        encoding="utf-8",
    )
    return {
        "env": env,
        "policy": {
            "type": "starvla",
            "checkpoint_path": str(tmp_path / "checkpoint"),
            "model_config_path": str(model_config_path),
            "task_manifest_path": str(manifest_path),
            "device": "cuda:0",
            "actions": ["push_left", "coast", "push_right"],
        },
    }


def _observation(metadata: dict | None = None) -> Observation:
    return Observation(
        data=np.zeros((2,), dtype=np.float32),
        env_step=0,
        sim_time_ms=0.0,
        metadata={
            "env_raw_rgb_frame_stack": np.zeros((8, 8, 3), dtype=np.uint8),
            **(metadata or {}),
        },
    )


def _read_model_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_generic_checkpoint_policy_separates_integration_task_and_native_action(
    monkeypatch,
    tmp_path: Path,
):
    from latency_bench.policy import starvla

    GenericPolicyServerWrapper.instances.clear()
    monkeypatch.setattr(
        starvla,
        "_load_policy_wrapper_class",
        lambda: GenericPolicyServerWrapper,
    )
    monkeypatch.setattr(
        starvla,
        "_normalized_model_cfg",
        lambda model_cfg: model_cfg,
    )
    monkeypatch.setattr(starvla, "_load_starvla_model_config", _read_model_config)
    config = _generic_config(tmp_path)

    policy = build_policy(config, ActionResolver(config["env"]))
    output = policy.predict(_observation())

    wrapper = GenericPolicyServerWrapper.instances[0]
    assert wrapper.rl_games_env_name == "gymnasium"
    assert wrapper.rl_games_action_env_dim == 3
    assert policy._env_name == "mountain_car"
    assert wrapper.calls[0]["examples"][0]["lang"] == "Drive the car up the mountain."
    assert output.action.name == "push_right"
    assert output.action.value == 2
    assert output.metadata["raw_action_id"] == 0


@pytest.mark.parametrize(
    ("normalization", "metadata", "expected_state"),
    [
        (None, {}, [[0.0, 0.0]]),
        (
            {"type": "min_max", "min": [0.0, 2.0], "max": [2.0, 4.0]},
            {"gymnasium_state": [1.0, 3.0]},
            [[0.0, 0.0]],
        ),
    ],
)
def test_generic_policy_uses_state_only_with_explicit_normalization(
    monkeypatch,
    tmp_path: Path,
    normalization,
    metadata,
    expected_state,
):
    from latency_bench.policy import starvla

    monkeypatch.setattr(starvla, "_normalized_model_cfg", lambda model_cfg: model_cfg)

    config = _generic_config(tmp_path)
    wrapper = GenericPolicyServerWrapper(None)
    model_cfg = {
        "datasets": {
            "vla_data": {
                "include_state": True,
                "num_obs_frames": 1,
                "image_mode": "single",
            }
        },
        "framework": {
            "action_model": {"state_dim": 2},
            "kv_memory": {"enabled": False},
        },
    }
    runner = starvla.StarVlaPolicyRunner(
        wrapper=wrapper,
        checkpoint_path="checkpoint.pt",
        device="cpu",
        unnorm_key=None,
        env_name="mountain_car",
        action_resolver=ActionResolver(config["env"]),
        action_refs=config["env"]["action_order"],
        base_prompt=config["env"]["base_prompt"],
        observation_stride_raw_frames=3,
        model_cfg=model_cfg,
        state_normalization=normalization,
    )

    runner.predict(_observation(metadata))

    assert wrapper.calls[0]["examples"][0]["state"].tolist() == expected_state


def test_generic_policy_allows_eval_only_reset_wrapper_overrides(
    monkeypatch,
    tmp_path: Path,
):
    from latency_bench.policy import starvla

    GenericPolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: GenericPolicyServerWrapper)
    monkeypatch.setattr(starvla, "_normalized_model_cfg", lambda model_cfg: model_cfg)
    monkeypatch.setattr(starvla, "_load_starvla_model_config", _read_model_config)
    config = _generic_config(tmp_path)
    training_contract = gymnasium_task_contract(config["env"])
    config["policy"]["gymnasium_training_task_contract"] = training_contract
    config["env"]["make_kwargs"] = {
        "render_mode": "rgb_array",
        "noop_max": 30,
        "fire_reset": True,
    }

    policy = build_policy(config, ActionResolver(config["env"]))

    assert policy._env_name == "mountain_car"


def test_generic_prompt_map_takes_priority_over_explicit_base_prompt(
    monkeypatch,
    tmp_path: Path,
):
    from latency_bench.policy import starvla

    GenericPolicyServerWrapper.instances.clear()
    monkeypatch.setattr(
        starvla,
        "_load_policy_wrapper_class",
        lambda: GenericPolicyServerWrapper,
    )
    monkeypatch.setattr(
        starvla,
        "_normalized_model_cfg",
        lambda model_cfg: model_cfg,
    )
    monkeypatch.setattr(starvla, "_load_starvla_model_config", _read_model_config)
    prompt_map_path = tmp_path / "latency_prompt_map.json"
    prompt_map_path.write_text(
        json.dumps({"3": {"prompt": "Use the latency-three policy."}}),
        encoding="utf-8",
    )
    config = _generic_config(tmp_path)
    config["policy"]["latency_prompt_map_path"] = str(prompt_map_path)

    policy = build_policy(config, ActionResolver(config["env"]))
    policy.predict(_observation({"latency_raw_frames": 3}))

    example = GenericPolicyServerWrapper.instances[0].calls[0]["examples"][0]
    assert example["lang"] == "Use the latency-three policy."


def test_generic_live_policy_passes_action_dim_to_decoder_and_decodes_native_value(
    monkeypatch,
    tmp_path: Path,
):
    from latency_bench.policy import starvla

    decode_calls = []

    def decode_rl_games_actions(
        *,
        normalized_actions,
        env_name,
        deadly_action_layout,
        deadly_multibinary_threshold,
        action_env_dim,
    ):
        decode_calls.append(
            {
                "env_name": env_name,
                "action_env_dim": action_env_dim,
                "normalized_actions": normalized_actions,
            }
        )
        actions = np.argmax(
            normalized_actions[..., :action_env_dim], axis=-1
        )[..., None]
        return {
            "actions": actions,
            "raw_action_scores": normalized_actions,
            "action_output_type": "rl_games_discrete_id",
        }

    monkeypatch.setattr(
        starvla,
        "_load_rl_games_action_decode",
        lambda: (decode_rl_games_actions, None),
    )
    monkeypatch.setattr(
        starvla,
        "_normalized_model_cfg",
        lambda model_cfg: model_cfg,
    )
    monkeypatch.setattr(starvla, "_load_starvla_model_config", _read_model_config)
    config = _generic_config(tmp_path)
    framework = GenericLiveFramework()
    model_cfg = json.loads(
        Path(config["policy"]["model_config_path"]).read_text(encoding="utf-8")
    )

    policy = starvla.build_live_starvla_policy(
        framework=framework,
        model_cfg=model_cfg,
        config=config,
        action_resolver=ActionResolver(config["env"]),
    )
    output = policy.predict(_observation())

    assert policy._env_name == "mountain_car"
    assert policy._wrapper._rl_games_env_name == "gymnasium"
    assert policy._wrapper._rl_games_action_env_dim == 3
    assert decode_calls[0]["env_name"] == "gymnasium"
    assert decode_calls[0]["action_env_dim"] == 3
    assert framework.calls[0]["examples"][0]["lang"] == "Drive the car up the mountain."
    assert output.action.name == "push_right"
    assert output.action.value == 2


@pytest.mark.parametrize(
    "mutate_env",
    [
        lambda env: env.update(action_order=["push_left", "coast", "push_right"]),
        lambda env: env.update(noop_action="push_left"),
    ],
)
def test_generic_starvla_rejects_eval_task_contract_drift(
    monkeypatch,
    tmp_path: Path,
    mutate_env,
):
    from latency_bench.policy import starvla

    monkeypatch.setattr(
        starvla,
        "_load_policy_wrapper_class",
        lambda: GenericPolicyServerWrapper,
    )
    monkeypatch.setattr(
        starvla,
        "_normalized_model_cfg",
        lambda model_cfg: model_cfg,
    )
    config = _generic_config(tmp_path)
    mutate_env(config["env"])
    monkeypatch.setattr(starvla, "_load_starvla_model_config", _read_model_config)

    with pytest.raises(ValueError, match="dataset manifest"):
        build_policy(config, ActionResolver(config["env"]))


def test_generic_starvla_rejects_model_config_contract_drift(
    monkeypatch,
    tmp_path: Path,
):
    from latency_bench.policy import starvla

    monkeypatch.setattr(
        starvla,
        "_load_policy_wrapper_class",
        lambda: GenericPolicyServerWrapper,
    )
    monkeypatch.setattr(starvla, "_normalized_model_cfg", lambda model_cfg: model_cfg)
    monkeypatch.setattr(starvla, "_load_starvla_model_config", _read_model_config)
    config = _generic_config(tmp_path)
    model_cfg = _read_model_config(config["policy"]["model_config_path"])
    model_cfg["datasets"]["vla_data"]["gymnasium_task_contract"]["action_values"] = [0, 2, 1]
    Path(config["policy"]["model_config_path"]).write_text(
        json.dumps(model_cfg),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model config"):
        build_policy(config, ActionResolver(config["env"]))


def test_explicit_generic_base_prompt_supports_timing_suffix_without_registry_entry():
    prompt = resolve_starvla_prompt(
        env_name="mountain_car",
        observation_metadata={
            "latency_raw_frames": 3,
            "latency_ms": 100.0,
            "env_fps": 30.0,
            "obs_fps": 10.0,
        },
        latency_prompt_map=None,
        base_prompt="Drive the car up the mountain.",
    )

    assert prompt.startswith("Drive the car up the mountain.")
    assert "3 raw frames (100.00 ms)" in prompt

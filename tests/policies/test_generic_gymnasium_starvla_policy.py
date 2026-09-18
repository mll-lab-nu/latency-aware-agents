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
        rl_games_gymnasium_action_space_type=None,
        rl_games_gymnasium_robot_type=None,
        backbone_path=None,
    ):
        self.rl_games_env_name = rl_games_env_name
        self.rl_games_action_env_dim = rl_games_action_env_dim
        self.rl_games_gymnasium_action_space_type = rl_games_gymnasium_action_space_type
        self.rl_games_gymnasium_robot_type = rl_games_gymnasium_robot_type
        self.backbone_path = backbone_path
        self.calls = []
        self._model_cfg = {
            "datasets": {"vla_data": {"include_state": False}},
            "framework": {"action_model": {}},
        }
        GenericPolicyServerWrapper.instances.append(self)

    def predict_action(self, examples, unnorm_key=None, **kwargs):
        self.calls.append({"examples": examples, "unnorm_key": unnorm_key, "kwargs": kwargs})
        if self.rl_games_gymnasium_action_space_type == "box":
            actions = np.zeros(
                (len(examples), 1, self.rl_games_action_env_dim), dtype=np.float32
            )
            return {
                "actions": actions,
                "raw_action_scores": actions,
                "action_output_type": "rl_games_continuous",
            }
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
        "state_space": {"labels": ["position", "velocity"]},
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
                "robot_type": "rl_games_gymnasium",
                "active_action_dim": 3,
                "uses_state": True,
                "state_dim": 2,
                "state_labels": ["position", "velocity"],
                "state_normalization": {
                    "type": "min_max",
                    "min": [-10.0, -10.0],
                    "max": [10.0, 10.0],
                },
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
                        "include_state": True,
                        "gymnasium_task_contract": contract,
                    }
                },
                "framework": {"action_model": {"action_env_dim": 3, "state_dim": 2}},
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
        data=np.asarray([1.25, -2.5], dtype=np.float32),
        env_step=0,
        sim_time_ms=0.0,
        metadata={
            "env_raw_rgb_frame_stack": np.zeros((8, 8, 3), dtype=np.uint8),
            "gymnasium_state": np.asarray([1.25, -2.5], dtype=np.float32),
            **(metadata or {}),
        },
    )


def _read_model_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("backbone_path", [None, "/models/pinned-qwen"])
def test_generic_checkpoint_policy_separates_integration_task_and_state_action(
    monkeypatch,
    tmp_path: Path,
    backbone_path,
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
    if backbone_path is not None:
        config["policy"]["backbone_path"] = backbone_path

    policy = build_policy(config, ActionResolver(config["env"]))
    output = policy.predict(_observation())

    wrapper = GenericPolicyServerWrapper.instances[0]
    assert wrapper.backbone_path == backbone_path
    assert wrapper.rl_games_env_name == "gymnasium"
    assert wrapper.rl_games_action_env_dim == 3
    assert wrapper.rl_games_gymnasium_action_space_type is None
    assert wrapper.rl_games_gymnasium_robot_type is None
    assert policy._env_name == "mountain_car"
    example = wrapper.calls[0]["examples"][0]
    assert example["lang"] == "Drive the car up the mountain."
    np.testing.assert_array_equal(
        example["state"], np.asarray([[0.125, -0.25]], dtype=np.float32)
    )
    assert output.action.name == "push_right"
    assert output.action.value == 2
    assert output.metadata["raw_action_id"] == 0


def test_generic_checkpoint_policy_accepts_legacy_environment_namespace(
    monkeypatch,
    tmp_path: Path,
):
    from latency_bench.policy import starvla

    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: GenericPolicyServerWrapper)
    monkeypatch.setattr(starvla, "_normalized_model_cfg", lambda model_cfg: model_cfg)
    monkeypatch.setattr(starvla, "_load_starvla_model_config", _read_model_config)
    config = _generic_config(tmp_path)
    config["env"].update(
        task_name="hopper",
        env_id="LatencyBench/Hopper-v0",
        registration_imports=["latency_bench.envs.gymnasium_hopper"],
    )
    legacy_contract = gymnasium_task_contract(config["env"])
    legacy_contract.update(
        task_name="hopper_rgb_state",
        env_id="LatencyBench/HopperRgbState-v0",
        registration_imports=["latency_bench.envs.gymnasium_hopper_rgb_state"],
    )
    manifest_path = Path(config["policy"]["task_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["gymnasium_task"] = legacy_contract
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    model_config_path = Path(config["policy"]["model_config_path"])
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model_config["datasets"]["vla_data"]["gymnasium_task_contract"] = legacy_contract
    model_config_path.write_text(json.dumps(model_config), encoding="utf-8")

    policy = build_policy(config, ActionResolver(config["env"]))

    assert policy._env_name == "hopper"


def test_generic_checkpoint_policy_keeps_stateless_gymnasium_stateless(
    monkeypatch,
    tmp_path: Path,
):
    from latency_bench.policy import starvla

    GenericPolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: GenericPolicyServerWrapper)
    monkeypatch.setattr(starvla, "_normalized_model_cfg", lambda model_cfg: model_cfg)
    monkeypatch.setattr(starvla, "_load_starvla_model_config", _read_model_config)
    config = _generic_config(tmp_path)
    del config["env"]["state_space"]
    contract = gymnasium_task_contract(config["env"])
    manifest_path = Path(config["policy"]["task_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "uses_state": False,
            "state_dim": 1,
            "state_labels": ["state"],
            "state_normalization": None,
            "gymnasium_task": contract,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    model_config_path = Path(config["policy"]["model_config_path"])
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model_config["datasets"]["vla_data"].update(
        {"include_state": False, "gymnasium_task_contract": contract}
    )
    model_config_path.write_text(json.dumps(model_config), encoding="utf-8")

    policy = build_policy(config, ActionResolver(config["env"]))
    policy.predict(_observation())

    assert "state" not in GenericPolicyServerWrapper.instances[0].calls[0]["examples"][0]


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("state_labels", "state_labels"),
        ("state_normalization", "state_normalization"),
    ],
)
def test_generic_checkpoint_policy_rejects_state_contract_mismatch(
    monkeypatch,
    tmp_path: Path,
    mismatch: str,
    message: str,
):
    from latency_bench.policy import starvla

    monkeypatch.setattr(starvla, "_normalized_model_cfg", lambda model_cfg: model_cfg)
    monkeypatch.setattr(starvla, "_load_starvla_model_config", _read_model_config)
    config = _generic_config(tmp_path)
    manifest_path = Path(config["policy"]["task_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if mismatch == "state_labels":
        manifest["state_labels"] = ["position"]
    else:
        manifest["state_normalization"] = {}

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_policy(config, ActionResolver(config["env"]))


def test_synthetic_continuous_checkpoint_policy_passes_state_and_box_contract(
    monkeypatch, tmp_path: Path
):
    from latency_bench.policy import starvla

    GenericPolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: GenericPolicyServerWrapper)
    monkeypatch.setattr(starvla, "_normalized_model_cfg", lambda model_cfg: model_cfg)
    config = {
        "env": {
            "name": "gymnasium",
            "task_name": "synthetic_continuous",
            "env_id": "SyntheticContinuous-v0",
            "registration_imports": [],
            "make_kwargs": {"render_mode": "rgb_array"},
            "env_fps": 20,
            "obs_fps": 20,
            "frame_stack": 1,
            "state_space": {"labels": ["position", "velocity"]},
            "action_space": {
                "type": "box",
                "labels": ["left", "right"],
                "low": [-1.0, -1.0],
                "high": [1.0, 1.0],
            },
            "noop_action": [0.0, 0.0],
            "base_prompt": "Move the synthetic system to its target.",
        },
        "policy": {
            "type": "starvla",
            "checkpoint_path": str(tmp_path / "checkpoint"),
            "model_config_path": str(tmp_path / "config.full.yaml"),
            "task_manifest_path": str(tmp_path / "manifest.json"),
            "device": "cuda:0",
        },
    }
    contract = gymnasium_task_contract(config["env"])
    task_manifest_path = tmp_path / "manifest.json"
    task_manifest_path.write_text(
        json.dumps(
            {
                "integration_name": "gymnasium",
                "active_action_dim": 2,
                "uses_state": True,
                "state_dim": 2,
                "state_labels": ["position", "velocity"],
                "state_normalization": {
                    "type": "min_max",
                    "min": [-10.0, -10.0],
                    "max": [10.0, 10.0],
                },
                "gymnasium_task": contract,
            }
        ),
        encoding="utf-8",
    )
    config["policy"]["task_manifest_path"] = str(task_manifest_path)
    model_cfg = {
        "datasets": {
            "vla_data": {
                "include_state": True,
                "gymnasium_task_contract": contract,
            }
        },
        "framework": {"action_model": {"action_env_dim": 2, "state_dim": 2}},
    }
    monkeypatch.setattr(starvla, "_load_starvla_model_config", lambda path: model_cfg)
    state = np.asarray([1.25, -2.5], dtype=np.float32)

    policy = build_policy(config, ActionResolver(config["env"]))
    output = policy.predict(
        Observation(
            data=state,
            env_step=0,
            sim_time_ms=0.0,
            metadata={
                "env_raw_rgb_frame_stack": np.zeros((8, 8, 3), dtype=np.uint8),
                "gymnasium_state": state,
            },
        )
    )

    wrapper = GenericPolicyServerWrapper.instances[0]
    assert wrapper.rl_games_action_env_dim == 2
    assert wrapper.rl_games_gymnasium_action_space_type == "box"
    np.testing.assert_allclose(
        wrapper.calls[0]["examples"][0]["state"],
        (state / 10.0).reshape(1, 2),
    )
    assert output.action.value == [0.0, 0.0]


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


def test_generic_live_policy_uses_manifest_state_and_decodes_actions(
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
        asterix_action_layout,
        action_env_dim,
        gymnasium_action_space_type="discrete",
        action_values=None,
    ):
        decode_calls.append(
            {
                "env_name": env_name,
                "action_env_dim": action_env_dim,
                "gymnasium_action_space_type": gymnasium_action_space_type,
                "normalized_actions": normalized_actions,
            "action_values": action_values,
            }
        )
        actions = (
            normalized_actions[..., :action_env_dim]
            if gymnasium_action_space_type == "box"
            else np.argmax(normalized_actions[..., :action_env_dim], axis=-1)[
                ..., None
            ]
        )
        return {
            "actions": actions,
            "raw_action_scores": normalized_actions,
            "action_output_type": (
                "rl_games_continuous"
                if gymnasium_action_space_type == "box"
                else "rl_games_discrete_id"
            ),
        }

    monkeypatch.setattr(
        starvla,
        "_load_rl_games_action_decode",
        lambda: (decode_rl_games_actions, None, None),
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
    assert decode_calls[0]["gymnasium_action_space_type"] == "discrete"
    assert framework.calls[0]["examples"][0]["lang"] == "Drive the car up the mountain."
    np.testing.assert_array_equal(
        framework.calls[0]["examples"][0]["state"],
        np.asarray([[0.125, -0.25]], dtype=np.float32),
    )
    assert output.action.name == "push_right"
    assert output.action.value == 2

    box_wrapper = starvla.LiveStarVlaWrapper(
        framework=framework,
        model_cfg=model_cfg,
        env_name="gymnasium",
        rl_games_action_env_dim=2,
        gymnasium_action_space_type="box",
    )
    box_prediction = box_wrapper.predict_action(
        examples=[{"image": [], "lang": "", "state": np.asarray([[3.0, 4.0]])}]
    )

    np.testing.assert_allclose(box_prediction["actions"], [[[0.9, 0.1]]])


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

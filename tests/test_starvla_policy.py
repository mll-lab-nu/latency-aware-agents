from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Observation
from latency_bench.policy.factory import build_policy
from scripts.rollout_data.prompts import BASE_PROMPTS


class FakePolicyServerWrapper:
    instances: list["FakePolicyServerWrapper"] = []

    def __init__(self, ckpt_path, device="cuda", use_bf16=False, unnorm_key=None):
        self.ckpt_path = ckpt_path
        self.device = device
        self.use_bf16 = use_bf16
        self.unnorm_key = unnorm_key
        self.calls = []
        self._model_cfg = {
            "datasets": {
                "vla_data": {
                    "include_state": True,
                },
            },
            "framework": {
                "action_model": {
                    "state_dim": 7,
                },
            },
        }
        FakePolicyServerWrapper.instances.append(self)

    def predict_action(self, examples, unnorm_key=None, **kwargs):
        self.calls.append(
            {
                "examples": examples,
                "unnorm_key": unnorm_key,
                "kwargs": kwargs,
            }
        )
        return {
            "actions": np.asarray(
                [
                    [[0.1, 0.9]],
                    [[0.8, 0.2]],
                ][: len(examples)],
                dtype=np.float32,
            )
        }


def _config(tmp_path: Path, *, latency_prompt_map_path: str | None = None) -> dict:
    policy = {
        "type": "starvla",
        "checkpoint_path": str(tmp_path / "checkpoint"),
        "device": "cuda:0",
        "unnorm_key": "flappy",
        "actions": ["noop", "flap"],
    }
    if latency_prompt_map_path is not None:
        policy["latency_prompt_map_path"] = latency_prompt_map_path
    return {
        "env": {
            "name": "flappy",
            "noop_action": "noop",
            "action_map": {"noop": 0, "flap": 1},
            "oneshot_actions": ["flap"],
        },
        "policy": policy,
    }


def test_starvla_passes_all_raw_frames_and_base_prompt(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))
    frame_stack = np.zeros((2, 3, 6), dtype=np.uint8)
    frame_stack[:, :, :3] = 11
    frame_stack[:, :, 3:] = 22

    output = policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": frame_stack},
        )
    )

    wrapper = FakePolicyServerWrapper.instances[0]
    assert wrapper.use_bf16 is True
    assert wrapper.unnorm_key == "flappy"
    assert wrapper.calls[0]["unnorm_key"] == "flappy"
    example = wrapper.calls[0]["examples"][0]
    assert example["lang"] == BASE_PROMPTS["flappy"]
    assert len(example["image"]) == 2
    assert np.asarray(example["image"][0]).mean() == 11
    assert np.asarray(example["image"][1]).mean() == 22
    assert example["state"].shape == (1, 7)
    assert output.metadata["prompt_source"] == "base"
    assert output.action.name == "flap"
    assert output.action.value == 1


def test_starvla_passes_single_raw_frame_and_base_prompt(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))

    policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": np.full((1, 12, 16, 3), 77, dtype=np.uint8)},
        )
    )

    example = FakePolicyServerWrapper.instances[0].calls[0]["examples"][0]
    assert example["lang"] == BASE_PROMPTS["flappy"]
    assert len(example["image"]) == 1
    assert np.asarray(example["image"][0]).shape == (12, 16, 3)
    assert np.asarray(example["image"][0]).mean() == 77


def test_starvla_omits_state_when_checkpoint_excludes_state(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    class NoStatePolicyServerWrapper(FakePolicyServerWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._model_cfg = {
                "datasets": {
                    "vla_data": {
                        "include_state": False,
                    },
                },
                "framework": {
                    "action_model": {},
                },
            }

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: NoStatePolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))

    policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": np.zeros((2, 3, 6), dtype=np.uint8)},
        )
    )

    example = FakePolicyServerWrapper.instances[0].calls[0]["examples"][0]
    assert "state" not in example


def test_starvla_requires_state_dim_when_checkpoint_includes_state(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    class MissingStateDimWrapper(FakePolicyServerWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._model_cfg = {
                "datasets": {
                    "vla_data": {
                        "include_state": True,
                    },
                },
                "framework": {
                    "action_model": {},
                },
            }

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: MissingStateDimWrapper)
    config = _config(tmp_path)
    with pytest.raises(KeyError, match="state_dim"):
        build_policy(config, ActionResolver(config["env"]))


def test_starvla_latency_prompt_map_does_not_change_image_count(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    prompt_map_path = tmp_path / "latency_prompt_map.json"
    prompt_map_path.write_text(
        json.dumps({"3": {"latency_raw_frames": 3, "latency_ms": 100.0, "prompt": "latency prompt 3"}}),
        encoding="utf-8",
    )
    config = _config(tmp_path, latency_prompt_map_path=str(prompt_map_path))
    policy = build_policy(config, ActionResolver(config["env"]))
    frame_stack = np.zeros((6, 8, 9), dtype=np.uint8)
    frame_stack[:3, :, :] = 33
    frame_stack[3:, :, :] = 44

    output = policy.predict(
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={
                "latency_raw_frames": 3,
                "env_raw_rgb_frame_stack": frame_stack,
            },
        )
    )

    wrapper = FakePolicyServerWrapper.instances[0]
    example = wrapper.calls[0]["examples"][0]
    assert example["lang"] == "latency prompt 3"
    assert len(example["image"]) == 2
    assert np.asarray(example["image"][0]).mean() == 33
    assert np.asarray(example["image"][1]).mean() == 44
    assert output.metadata["prompt_source"] == "latency_prompt_map"
    assert output.metadata["unnorm_key"] == "flappy"


def test_starvla_latency_prompt_map_uses_raw_rgb_frame_stack_metadata(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    prompt_map_path = tmp_path / "latency_prompt_map.json"
    prompt_map_path.write_text(
        json.dumps({"2": {"latency_raw_frames": 2, "latency_ms": 33.0, "prompt": "latency prompt 2"}}),
        encoding="utf-8",
    )
    config = _config(tmp_path, latency_prompt_map_path=str(prompt_map_path))
    policy = build_policy(config, ActionResolver(config["env"]))
    policy_observation_stack = np.zeros((4, 8, 8), dtype=np.uint8)
    raw_rgb_stack = np.zeros((4, 12, 16, 3), dtype=np.uint8)
    raw_rgb_stack[-1] = 77

    policy.predict(
        Observation(
            data=policy_observation_stack,
            env_step=0,
            sim_time_ms=0.0,
            metadata={
                "latency_raw_frames": 2,
                "env_raw_rgb_frame_stack": raw_rgb_stack,
            },
        )
    )

    example = FakePolicyServerWrapper.instances[0].calls[0]["examples"][0]
    assert len(example["image"]) == 4
    assert np.asarray(example["image"][-1]).shape == (12, 16, 3)
    assert np.asarray(example["image"][-1]).mean() == 77


def test_starvla_requires_raw_rgb_frame_stack_metadata(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))

    with pytest.raises(KeyError, match="env_raw_rgb_frame_stack"):
        policy.predict(Observation(data=np.zeros((2, 3, 3), dtype=np.uint8), env_step=0, sim_time_ms=0.0))


def test_deadly_corridor_factorized_starvla_vector_decodes_to_tuple_action():
    from latency_bench.policy.starvla import decode_starvla_action

    vector = np.asarray([0.0, 0.2, 0.9, 0.8, 0.1, 0.2, 0.1, 0.7, 0.0, 0.3, 0.9])

    action, metadata = decode_starvla_action(
        vector=vector,
        env_name="deadly_corridor",
        action_by_raw_id={},
    )

    assert action.value == [2, 0, 1, 1]
    assert action.name == "TURN_RIGHT+MOVE_LEFT+ATTACK"
    assert metadata["action_layout"] == "deadly_corridor_factorized_11"

    with pytest.raises(ValueError, match="expected 7 or 11"):
        decode_starvla_action(
            vector=np.zeros(8, dtype=np.float32),
            env_name="deadly_corridor",
            action_by_raw_id={},
        )


def test_predict_batch_calls_starvla_once(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    config = _config(tmp_path)
    policy = build_policy(config, ActionResolver(config["env"]))
    observations = [
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=0,
            sim_time_ms=0.0,
            metadata={"env_raw_rgb_frame_stack": np.zeros((2, 3, 3), dtype=np.uint8)},
        ),
        Observation(
            data=np.zeros((1,), dtype=np.uint8),
            env_step=1,
            sim_time_ms=33.3,
            metadata={"env_raw_rgb_frame_stack": np.zeros((2, 3, 3), dtype=np.uint8)},
        ),
    ]

    outputs = policy.predict_batch(observations)

    wrapper = FakePolicyServerWrapper.instances[0]
    assert len(wrapper.calls) == 1
    assert len(wrapper.calls[0]["examples"]) == 2
    assert [output.action.name for output in outputs] == ["flap", "noop"]

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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


def _config(tmp_path: Path, *, profile: str, latency_prompt_map_path: str | None = None) -> dict:
    policy = {
        "type": "starvla",
        "checkpoint_path": str(tmp_path / "checkpoint"),
        "device": "cuda:0",
        "unnorm_key": "flappy",
        "profile": profile,
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


def test_frame_stack_profile_passes_all_frames_and_base_prompt(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    policy = build_policy(_config(tmp_path, profile="frame_stack"), ActionResolver(_config(tmp_path, profile="frame_stack")["env"]))
    frame_stack = np.zeros((2, 3, 6), dtype=np.uint8)
    frame_stack[:, :, :3] = 11
    frame_stack[:, :, 3:] = 22

    output = policy.predict(Observation(data=frame_stack, env_step=0, sim_time_ms=0.0))

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
    assert output.action.name == "flap"
    assert output.action.value == 1


def test_single_latency_profile_uses_latest_frame_and_latency_prompt(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    prompt_map_path = tmp_path / "latency_prompt_map.json"
    prompt_map_path.write_text(
        json.dumps({"3": {"latency_raw_frames": 3, "latency_ms": 100.0, "prompt": "latency prompt 3"}}),
        encoding="utf-8",
    )
    config = _config(tmp_path, profile="single_latency", latency_prompt_map_path=str(prompt_map_path))
    policy = build_policy(config, ActionResolver(config["env"]))
    frame_stack = np.zeros((6, 8, 9), dtype=np.uint8)
    frame_stack[:3, :, :] = 33
    frame_stack[3:, :, :] = 44

    output = policy.predict(
        Observation(
            data=frame_stack,
            env_step=0,
            sim_time_ms=0.0,
            metadata={"latency_raw_frames": 3},
        )
    )

    wrapper = FakePolicyServerWrapper.instances[0]
    example = wrapper.calls[0]["examples"][0]
    assert example["lang"] == "latency prompt 3"
    assert len(example["image"]) == 1
    assert np.asarray(example["image"][0]).mean() == 44
    assert output.metadata["profile"] == "single_latency"
    assert output.metadata["unnorm_key"] == "flappy"


def test_predict_batch_calls_starvla_once(monkeypatch, tmp_path: Path):
    from latency_bench.policy import starvla

    FakePolicyServerWrapper.instances.clear()
    monkeypatch.setattr(starvla, "_load_policy_wrapper_class", lambda: FakePolicyServerWrapper)
    config = _config(tmp_path, profile="frame_stack")
    policy = build_policy(config, ActionResolver(config["env"]))
    observations = [
        Observation(data=np.zeros((2, 3, 3), dtype=np.uint8), env_step=0, sim_time_ms=0.0),
        Observation(data=np.zeros((2, 3, 3), dtype=np.uint8), env_step=1, sim_time_ms=33.3),
    ]

    outputs = policy.predict_batch(observations)

    wrapper = FakePolicyServerWrapper.instances[0]
    assert len(wrapper.calls) == 1
    assert len(wrapper.calls[0]["examples"]) == 2
    assert [output.action.name for output in outputs] == ["flap", "noop"]

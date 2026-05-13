from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Action, Observation
from latency_bench.envs.demon_attack import DemonAttackEnvAdapter
from latency_bench.policy import demon_attack_openvla
from latency_bench.policy.demon_attack_openvla import (
    DEMON_ATTACK_OPENVLA_ACTION_LABELS,
    DemonAttackOpenVlaPolicyRunner,
    observation_data_to_hwc_rgb_uint8,
)


class _MockOpenVlaModel:
    def __init__(self, action_vector):
        self.action_vector = np.asarray(action_vector, dtype=np.float32)

    def predict_action_batch(self, **kwargs):
        return self.action_vector, {"forward_inputs": {"action_tokens": np.array([1, 2, 3])}}


def _policy_config() -> demon_attack_openvla.DemonAttackOpenVlaPolicyConfig:
    return demon_attack_openvla.DemonAttackOpenVlaPolicyConfig(
        base_model_path=Path("/missing/base"),
        checkpoint_path=Path("/missing/checkpoint"),
        prompt="play demon attack",
        image_size=84,
        action_dim=6,
        num_action_chunks=1,
        add_value_head=False,
        max_prompt_length=50,
        num_images_in_input=1,
        center_crop=True,
        trust_remote_code=True,
        unnorm_key="demon_attack",
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("action_id", range(6))
def test_demon_openvla_argmax_maps_to_integer_action_value(action_id: int):
    vector = np.zeros(6, dtype=np.float32)
    vector[action_id] = 1.0
    policy = DemonAttackOpenVlaPolicyRunner(
        model=_MockOpenVlaModel(vector),
        config=_policy_config(),
        device=torch.device("cpu"),
    )
    observation = Observation(
        data=np.zeros((84, 84, 1), dtype=np.uint8),
        env_step=0,
        sim_time_ms=0.0,
    )

    output = policy.predict(observation)

    assert output.action.value == action_id
    assert output.action.name == DEMON_ATTACK_OPENVLA_ACTION_LABELS[action_id]
    assert output.raw_output == action_id


def test_demon_openvla_grayscale_observation_converts_to_rgb_hwc_uint8():
    gray = np.arange(84 * 84, dtype=np.uint8).reshape(84, 84, 1)

    rgb = observation_data_to_hwc_rgb_uint8(gray, image_size=84)

    assert rgb.shape == (84, 84, 3)
    assert rgb.dtype == np.uint8
    assert np.array_equal(rgb[..., 0], gray[..., 0])
    assert np.array_equal(rgb[..., 1], gray[..., 0])
    assert np.array_equal(rgb[..., 2], gray[..., 0])


def test_demon_openvla_factory_can_build_with_mock_loader(monkeypatch):
    from latency_bench.policy.factory import build_policy

    def fake_load(config):
        return _MockOpenVlaModel([0, 0, 1, 0, 0, 0]), torch.device("cpu")

    monkeypatch.setattr(demon_attack_openvla, "load_openvla_sft_model", fake_load)
    config = {
        "experiment": {"seed": 0},
        "env": {"name": "demon_attack", "env_fps": 60, "noop_action": 0},
        "policy": {
            "type": "demon_attack_openvla",
            "base_model_path": "/missing/base",
            "checkpoint_path": "/missing/checkpoint",
            "prompt": "play demon attack",
            "image_size": 84,
            "action_dim": 6,
            "num_action_chunks": 1,
            "add_value_head": False,
            "max_prompt_length": 50,
            "num_images_in_input": 1,
            "center_crop": True,
            "trust_remote_code": True,
            "unnorm_key": "demon_attack",
            "device": "cpu",
        },
    }

    policy = build_policy(config, ActionResolver(config["env"]))

    output = policy.predict(
        Observation(data=np.zeros((84, 84, 1), dtype=np.uint8), env_step=0, sim_time_ms=0.0)
    )
    assert output.action.value == 2


def test_demon_openvla_missing_dataset_statistics_fails_before_model_load(tmp_path):
    from latency_bench.policy.factory import build_policy

    config = {
        "experiment": {"seed": 0},
        "env": {"name": "demon_attack", "env_fps": 60, "noop_action": 0},
        "policy": {
            "type": "demon_attack_openvla",
            "base_model_path": str(tmp_path / "base"),
            "checkpoint_path": str(tmp_path / "checkpoint"),
            "prompt": "play demon attack",
            "image_size": 84,
            "action_dim": 6,
            "num_action_chunks": 1,
            "add_value_head": False,
            "max_prompt_length": 50,
            "num_images_in_input": 1,
            "center_crop": True,
            "trust_remote_code": True,
            "unnorm_key": "demon_attack",
            "device": "cpu",
            "actions": list(DEMON_ATTACK_OPENVLA_ACTION_LABELS),
        },
    }

    with pytest.raises(FileNotFoundError, match="demon_attack_dataset_statistics.json"):
        build_policy(config, ActionResolver(config["env"]))


def test_demon_action_meanings_mismatch_raises_clear_error(monkeypatch):
    class FakeGymEnv:
        def get_action_meanings(self):
            return ["NOOP", "FIRE", "LEFT"]

        def close(self):
            pass

    monkeypatch.setattr(
        "latency_bench.envs.demon_attack.make_demon_attack_benchmark_env",
        lambda **kwargs: FakeGymEnv(),
    )

    with pytest.raises(ValueError, match="expected_action_meanings.*actual"):
        DemonAttackEnvAdapter(
            noop_action=Action(value=0, name="NOOP", is_noop=True),
            expected_action_meanings=list(DEMON_ATTACK_OPENVLA_ACTION_LABELS),
        )

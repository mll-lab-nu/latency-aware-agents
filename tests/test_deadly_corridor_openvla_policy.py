from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from latency_bench.core.actions import ActionResolver
from latency_bench.core.types import Observation
from latency_bench.policy import deadly_corridor_openvla
from latency_bench.policy.deadly_corridor_openvla import (
    DEADLY_CORRIDOR_OPENVLA_ACTION_LABELS,
    DEADLY_CORRIDOR_OPENVLA_ACTION_VALUES,
    DeadlyCorridorOpenVlaPolicyRunner,
    observation_data_to_single_hwc_rgb_uint8,
)


class _MockOpenVlaModel:
    def __init__(self, action_vector):
        self.action_vector = np.asarray(action_vector, dtype=np.float32)

    def predict_action_batch(self, **kwargs):
        return self.action_vector, {"forward_inputs": {"action_tokens": np.array([4, 5, 6])}}


def _policy_config() -> deadly_corridor_openvla.DeadlyCorridorOpenVlaPolicyConfig:
    return deadly_corridor_openvla.DeadlyCorridorOpenVlaPolicyConfig(
        base_model_path=Path("/missing/base"),
        checkpoint_path=Path("/missing/checkpoint"),
        prompt="play deadly corridor",
        image_size=84,
        action_dim=7,
        num_action_chunks=1,
        add_value_head=False,
        max_prompt_length=50,
        num_images_in_input=1,
        center_crop=True,
        trust_remote_code=True,
        unnorm_key="deadly_corridor",
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("action_id", range(7))
def test_deadly_openvla_argmax_maps_to_expected_list_action(action_id: int):
    vector = np.zeros(7, dtype=np.float32)
    vector[action_id] = 1.0
    policy = DeadlyCorridorOpenVlaPolicyRunner(
        model=_MockOpenVlaModel(vector),
        config=_policy_config(),
        device=torch.device("cpu"),
    )
    observation = Observation(
        data=np.zeros((3, 84, 84), dtype=np.uint8),
        env_step=0,
        sim_time_ms=0.0,
    )

    output = policy.predict(observation)

    assert output.action.value == DEADLY_CORRIDOR_OPENVLA_ACTION_VALUES[action_id]
    assert output.action.name == DEADLY_CORRIDOR_OPENVLA_ACTION_LABELS[action_id]
    assert output.raw_output == action_id


def test_deadly_openvla_accepts_hwc_rgb():
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    frame[..., 1] = 7

    rgb = observation_data_to_single_hwc_rgb_uint8(frame)

    assert rgb.shape == (72, 128, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb[..., 1] == 7)


def test_deadly_openvla_accepts_chw_rgb():
    frame = np.zeros((3, 72, 128), dtype=np.uint8)
    frame[2, :, :] = 9

    rgb = observation_data_to_single_hwc_rgb_uint8(frame)

    assert rgb.shape == (72, 128, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb[..., 2] == 9)


def test_deadly_openvla_stacked_chw_uses_last_three_channels_as_latest_frame():
    stacked = np.zeros((12, 72, 128), dtype=np.uint8)
    stacked[0:3, :, :] = 11
    stacked[-3, :, :] = 21
    stacked[-2, :, :] = 22
    stacked[-1, :, :] = 23

    rgb = observation_data_to_single_hwc_rgb_uint8(stacked)

    assert rgb.shape == (72, 128, 3)
    assert np.all(rgb[..., 0] == 21)
    assert np.all(rgb[..., 1] == 22)
    assert np.all(rgb[..., 2] == 23)


def test_deadly_openvla_extracts_dict_observation():
    frame = np.zeros((3, 72, 128), dtype=np.uint8)
    frame[1, :, :] = 33

    rgb = observation_data_to_single_hwc_rgb_uint8({"obs": frame})

    assert rgb.shape == (72, 128, 3)
    assert np.all(rgb[..., 1] == 33)


def test_deadly_openvla_factory_can_build_with_mock_loader(monkeypatch):
    from latency_bench.policy.factory import build_policy

    def fake_load(config):
        return _MockOpenVlaModel([0, 0, 0, 0, 0, 0, 1]), torch.device("cpu")

    monkeypatch.setattr(deadly_corridor_openvla, "load_openvla_sft_model", fake_load)
    config = {
        "experiment": {"seed": 0},
        "env": {"name": "deadly_corridor", "fps_env": 35, "noop_action": [0, 0, 0, 0]},
        "policy": {
            "type": "deadly_corridor_openvla",
            "base_model_path": "/missing/base",
            "checkpoint_path": "/missing/checkpoint",
            "prompt": "play deadly corridor",
            "image_size": 84,
            "action_dim": 7,
            "num_action_chunks": 1,
            "add_value_head": False,
            "max_prompt_length": 50,
            "num_images_in_input": 1,
            "center_crop": True,
            "trust_remote_code": True,
            "unnorm_key": "deadly_corridor",
            "device": "cpu",
        },
    }

    policy = build_policy(config, ActionResolver(config["env"]))

    output = policy.predict(
        Observation(data=np.zeros((12, 84, 84), dtype=np.uint8), env_step=0, sim_time_ms=0.0)
    )
    assert output.action.value == [0, 0, 0, 1]


def test_deadly_openvla_missing_unnorm_key_fails_before_model_load(tmp_path):
    from latency_bench.policy.factory import build_policy

    base = tmp_path / "base"
    base.mkdir()
    (base / "dataset_statistics.json").write_text('{"other_env": {}}\n', encoding="utf-8")
    config = {
        "experiment": {"seed": 0},
        "env": {"name": "deadly_corridor", "fps_env": 35, "noop_action": [0, 0, 0, 0]},
        "policy": {
            "type": "deadly_corridor_openvla",
            "base_model_path": str(base),
            "checkpoint_path": str(tmp_path / "checkpoint"),
            "prompt": "play deadly corridor",
            "image_size": 84,
            "action_dim": 7,
            "num_action_chunks": 1,
            "add_value_head": False,
            "max_prompt_length": 50,
            "num_images_in_input": 1,
            "center_crop": True,
            "trust_remote_code": True,
            "unnorm_key": "deadly_corridor",
            "device": "cpu",
            "actions": list(DEADLY_CORRIDOR_OPENVLA_ACTION_LABELS),
        },
    }

    with pytest.raises(KeyError, match="deadly_corridor"):
        build_policy(config, ActionResolver(config["env"]))


def test_deadly_openvla_missing_full_weights_fails_before_model_load(tmp_path):
    from latency_bench.policy.factory import build_policy

    base = tmp_path / "base"
    base.mkdir()
    (base / "dataset_statistics.json").write_text('{"deadly_corridor": {}}\n', encoding="utf-8")
    config = {
        "experiment": {"seed": 0},
        "env": {"name": "deadly_corridor", "fps_env": 35, "noop_action": [0, 0, 0, 0]},
        "policy": {
            "type": "deadly_corridor_openvla",
            "base_model_path": str(base),
            "checkpoint_path": str(tmp_path / "checkpoint"),
            "prompt": "play deadly corridor",
            "image_size": 84,
            "action_dim": 7,
            "num_action_chunks": 1,
            "add_value_head": False,
            "max_prompt_length": 50,
            "num_images_in_input": 1,
            "center_crop": True,
            "trust_remote_code": True,
            "unnorm_key": "deadly_corridor",
            "device": "cpu",
            "actions": list(DEADLY_CORRIDOR_OPENVLA_ACTION_LABELS),
        },
    }

    with pytest.raises(FileNotFoundError, match="full_weights.pt"):
        build_policy(config, ActionResolver(config["env"]))

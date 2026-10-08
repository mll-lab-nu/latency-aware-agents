from __future__ import annotations

import pytest
from omegaconf import DictConfig, OmegaConf

from starVLA.training.rl_games.config_validation import (
    sync_kv_memory_obs_window,
    validate_rl_games_config,
)


def _build_cfg(data: dict) -> DictConfig:
    return OmegaConf.create(data)


def test_kv_memory_sync_ties_obs_window_to_rollout_len() -> None:
    cfg = _build_cfg(
        {
            "framework": {"kv_memory": {"enabled": True, "window": 4, "rollout_len": 8}},
            "datasets": {"vla_data": {"num_obs_frames": 1, "image_mode": "single"}},
        }
    )

    sync_kv_memory_obs_window(cfg)

    assert cfg.datasets.vla_data.num_obs_frames == 8
    assert cfg.datasets.vla_data.image_mode == "multiframe"
    # Scheme-B per-frame supervision: dataloader must emit valid/actions_per_frame and
    # read the per-row density weight column.
    assert cfg.datasets.vla_data.kv_memory is True
    assert cfg.datasets.vla_data.density_weight_key == "density_weight"


def test_kv_memory_sync_noop_when_disabled() -> None:
    cfg = _build_cfg(
        {
            "framework": {"kv_memory": {"enabled": False, "window": 4, "rollout_len": 8}},
            "datasets": {"vla_data": {"num_obs_frames": 1, "image_mode": "single"}},
        }
    )

    sync_kv_memory_obs_window(cfg)

    assert cfg.datasets.vla_data.num_obs_frames == 1
    assert cfg.datasets.vla_data.image_mode == "single"


def test_kv_memory_sync_derives_rollout_len_from_window() -> None:
    cfg = _build_cfg(
        {
            "framework": {"kv_memory": {"enabled": True, "window": 4}},
            "datasets": {"vla_data": {}},
        }
    )

    sync_kv_memory_obs_window(cfg)

    # rollout_len defaults to max(window + 1, 2), matching QwenOFT.
    assert cfg.datasets.vla_data.num_obs_frames == 5
    assert cfg.datasets.vla_data.image_mode == "multiframe"


def test_accepts_pi05_scratch() -> None:
    cfg = _build_cfg(
        {
            "rl_games": {
                "model_alias": "pi-0.5",
                "task": "flappy",
                "initialization_mode": "scratch",
                "action_carrier": "native",
                "env_eval": {"latency": {"values": [0]}},
            },
            "dataset": {
                "source_hf": "",
                "converted_name": "flappy_train",
            },
            "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
            "initialization": {
                "checkpoint_local_dir": None,
                "checkpoint_hf_repo_id": None,
                "checkpoint_filename": None,
            },
        }
    )

    validate_rl_games_config(cfg)


def test_rejects_bridge_without_checkpoint_metadata() -> None:
    cfg = _build_cfg(
        {
            "rl_games": {
                "model_alias": "openvla",
                "task": "flappy",
                "initialization_mode": "bridge",
                "action_carrier": "bridge",
                "env_eval": {"latency": {"values": [0]}},
            },
            "dataset": {
                "source_hf": "",
                "converted_name": "flappy_train",
            },
            "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
            "initialization": {
                "checkpoint_local_dir": None,
                "checkpoint_hf_repo_id": None,
                "checkpoint_filename": None,
            },
        }
    )

    with pytest.raises(ValueError, match="bridge initialization requires"):
        validate_rl_games_config(cfg)


def test_rejects_bridge_with_blank_checkpoint_sources() -> None:
    cfg = _build_cfg(
        {
            "rl_games": {
                "model_alias": "openvla",
                "task": "flappy",
                "initialization_mode": "bridge",
                "action_carrier": "bridge",
                "env_eval": {"latency": {"values": [0]}},
            },
            "dataset": {
                "source_hf": "",
                "converted_name": "flappy_train",
            },
            "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
            "initialization": {
                "checkpoint_local_dir": "",
                "checkpoint_hf_repo_id": "",
                "checkpoint_filename": "checkpoints/steps_1000_pytorch_model.pt",
            },
        }
    )

    with pytest.raises(ValueError, match="bridge initialization requires"):
        validate_rl_games_config(cfg)


def test_rejects_pretrained_without_bridge_action_carrier() -> None:
    cfg = _build_cfg(
        {
            "rl_games": {
                "model_alias": "openvla",
                "task": "flappy",
                "initialization_mode": "pre-trained",
                "action_carrier": "native",
                "env_eval": {"latency": {"values": [0]}},
            },
            "dataset": {
                "source_hf": "",
                "converted_name": "flappy_train",
            },
            "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
            "initialization": {
                "checkpoint_local_dir": "playground/checkpoints",
                "checkpoint_hf_repo_id": None,
                "checkpoint_filename": "checkpoints/steps_1000_pytorch_model.pt",
            },
        }
    )

    with pytest.raises(ValueError, match="bridge initialization requires rl_games.action_carrier=bridge"):
        validate_rl_games_config(cfg)


@pytest.mark.parametrize("model_alias", ["openvla", "pi-0.5"])
def test_accepts_bridge_with_checkpoint_metadata(model_alias: str) -> None:
    cfg = _build_cfg(
        {
            "rl_games": {
                "model_alias": model_alias,
                "task": "flappy",
                "initialization_mode": "bridge",
                "action_carrier": "bridge",
                "env_eval": {"latency": {"values": [0]}},
            },
            "dataset": {
                "source_hf": "",
                "converted_name": "flappy_train",
            },
            "base_model": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct"},
            "initialization": {
                "checkpoint_local_dir": "playground/checkpoints",
                "checkpoint_hf_repo_id": None,
                "checkpoint_filename": "checkpoints/steps_1000_pytorch_model.pt",
            },
        }
    )

    validate_rl_games_config(cfg)

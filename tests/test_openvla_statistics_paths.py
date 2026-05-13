from __future__ import annotations

from pathlib import Path

import pytest

from latency_bench.policy.flappy_openvla import (
    extract_openvla_policy_config,
    openvla_loader_config,
)


def _policy_config(
    *,
    base_model_path: Path,
    policy_type: str,
    unnorm_key: str,
) -> dict[str, object]:
    return {
        "type": policy_type,
        "base_model_path": str(base_model_path),
        "checkpoint_path": "/missing/checkpoint",
        "prompt": "play",
        "image_size": 84,
        "action_dim": 2,
        "num_action_chunks": 1,
        "add_value_head": False,
        "max_prompt_length": 50,
        "num_images_in_input": 1,
        "center_crop": True,
        "trust_remote_code": True,
        "unnorm_key": unnorm_key,
        "device": "cpu",
    }


@pytest.mark.parametrize(
    ("policy_type", "unnorm_key", "filename"),
    [
        ("flappy_openvla", "flappy_bird", "flappy_dataset_statistics.json"),
        ("demon_attack_openvla", "demon_attack", "demon_attack_dataset_statistics.json"),
        (
            "deadly_corridor_openvla",
            "deadly_corridor",
            "deadly_corridor_dataset_statistics.json",
        ),
    ],
)
def test_openvla_policy_config_derives_policy_statistics_path(
    tmp_path: Path,
    policy_type: str,
    unnorm_key: str,
    filename: str,
) -> None:
    config = extract_openvla_policy_config(
        _policy_config(
            base_model_path=tmp_path,
            policy_type=policy_type,
            unnorm_key=unnorm_key,
        )
    )

    assert config.dataset_statistics_path == tmp_path / filename


def test_openvla_policy_config_rejects_unknown_policy_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported OpenVLA policy.type"):
        extract_openvla_policy_config(
            _policy_config(
                base_model_path=tmp_path,
                policy_type="other_openvla",
                unnorm_key="other",
            )
        )


def test_openvla_loader_config_includes_loaded_norm_stats(tmp_path: Path) -> None:
    config = extract_openvla_policy_config(
        _policy_config(
            base_model_path=tmp_path,
            policy_type="demon_attack_openvla",
            unnorm_key="demon_attack",
        )
    )
    norm_stats = {"demon_attack": {"action": {"q01": [0.0], "q99": [1.0]}}}

    loader_config = openvla_loader_config(config, norm_stats=norm_stats)

    assert loader_config["norm_stats"] == norm_stats

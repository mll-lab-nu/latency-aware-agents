from __future__ import annotations

from pathlib import Path


OPENVLA_DATASET_STATISTICS_FILENAME_BY_POLICY_TYPE = {
    "flappy_openvla": "flappy_dataset_statistics.json",
    "demon_attack_openvla": "demon_attack_dataset_statistics.json",
    "deadly_corridor_openvla": "deadly_corridor_dataset_statistics.json",
}


def openvla_dataset_statistics_filename(policy_type: str) -> str:
    normalized_policy_type = policy_type.strip().lower()
    try:
        return OPENVLA_DATASET_STATISTICS_FILENAME_BY_POLICY_TYPE[normalized_policy_type]
    except KeyError as error:
        supported = ", ".join(sorted(OPENVLA_DATASET_STATISTICS_FILENAME_BY_POLICY_TYPE))
        raise ValueError(
            f"Unsupported OpenVLA policy.type {policy_type!r}; expected one of: {supported}"
        ) from error


def openvla_dataset_statistics_path(base_model_path: Path, policy_type: str) -> Path:
    return base_model_path / openvla_dataset_statistics_filename(policy_type)

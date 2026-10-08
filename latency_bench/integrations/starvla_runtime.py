"""Lazy access to the pinned StarVLA runtime and saved model configuration."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from latency_bench.utils.paths import REPO_ROOT


STARVLA_ROOT = REPO_ROOT / "third_party" / "starVLA"


STATEFUL_STARVLA_MODEL_IDS: tuple[str, ...] = (
    "pi0",
    "pi-0",
    "pi05",
    "pi-0.5",
    "gr00t",
    "qwenpi",
    "qwenpi_v3",
    "qwengr00t",
)


STATELESS_STARVLA_MODEL_IDS: tuple[str, ...] = (
    "openvla",
    "qwenoft",
)


def ensure_starvla_path() -> None:
    starvla_root = str(STARVLA_ROOT)
    if starvla_root not in sys.path:
        sys.path.insert(0, starvla_root)


def normalize_model_config(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    ensure_starvla_path()
    from omegaconf import OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat

    cfg = OmegaConf.create(model_cfg)
    apply_config_compat(cfg)
    _apply_model_family_include_state_compat(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def load_model_config(path: str | Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    return normalize_model_config(OmegaConf.load(path))


def _apply_model_family_include_state_compat(cfg: Any) -> None:
    from omegaconf import OmegaConf

    if OmegaConf.select(cfg, "datasets.vla_data.include_state") is not None:
        return

    model_ids = (
        _normalized_optional_config_string(cfg, ("model",)),
        _normalized_optional_config_string(cfg, ("rl_games", "model_alias")),
        _normalized_optional_config_string(cfg, ("framework", "name")),
    )
    if any(model_id in STATEFUL_STARVLA_MODEL_IDS for model_id in model_ids if model_id is not None):
        OmegaConf.update(cfg, "datasets.vla_data.include_state", True, force_add=True)
        return
    if any(model_id in STATELESS_STARVLA_MODEL_IDS for model_id in model_ids if model_id is not None):
        OmegaConf.update(cfg, "datasets.vla_data.include_state", False, force_add=True)


def _normalized_optional_config_string(cfg: Any, path: tuple[str, ...]) -> str | None:
    from omegaconf import OmegaConf

    value = OmegaConf.select(cfg, ".".join(path))
    if value is None:
        return None
    return str(value).strip().lower()


def load_checkpoint_config(checkpoint_path: str | Path) -> dict[str, Any]:
    ensure_starvla_path()
    from starVLA.model.framework.share_tools import read_mode_config

    saved_model_cfg, _norm_stats = read_mode_config(checkpoint_path)
    return normalize_model_config(saved_model_cfg)


def load_policy_wrapper_class() -> Any:
    ensure_starvla_path()
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    return PolicyServerWrapper


def load_rl_games_action_decode() -> tuple[Any, Any, Any]:
    ensure_starvla_path()
    from deployment.model_server.rl_games_action_decode import (
        decode_rl_games_actions,
        resolve_asterix_action_decode_spec,
        resolve_deadly_action_decode_spec,
    )

    return decode_rl_games_actions, resolve_deadly_action_decode_spec, resolve_asterix_action_decode_spec

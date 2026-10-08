from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

BRIDGE_INITIALIZATION_MODES: set[str] = {"bridge", "pre-trained", "pretrained"}


def _select_value(cfg: Any, key: str) -> Any:
    return OmegaConf.select(cfg, key)


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _require_non_empty_value(cfg: Any, key: str) -> Any:
    value = _select_value(cfg=cfg, key=key)
    if _is_missing_value(value=value):
        raise ValueError(f"Missing required RL-games config field: {key}")
    return value


def _normalize_mode(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _is_bridge_initialization(cfg: Any) -> bool:
    initialization_mode = _normalize_mode(_select_value(cfg=cfg, key="rl_games.initialization_mode"))
    action_carrier = _normalize_mode(_select_value(cfg=cfg, key="rl_games.action_carrier"))
    return initialization_mode in BRIDGE_INITIALIZATION_MODES or action_carrier == "bridge"


def _has_checkpoint_source(cfg: Any) -> bool:
    checkpoint_local_dir = _select_value(cfg=cfg, key="initialization.checkpoint_local_dir")
    checkpoint_hf_repo_id = _select_value(cfg=cfg, key="initialization.checkpoint_hf_repo_id")
    return not _is_missing_value(value=checkpoint_local_dir) or not _is_missing_value(value=checkpoint_hf_repo_id)


def _validate_bridge_initialization(cfg: Any) -> None:
    action_carrier = _normalize_mode(_select_value(cfg=cfg, key="rl_games.action_carrier"))
    if action_carrier != "bridge":
        raise ValueError("bridge initialization requires rl_games.action_carrier=bridge")

    if not _has_checkpoint_source(cfg=cfg):
        raise ValueError(
            "bridge initialization requires initialization.checkpoint_local_dir or "
            "initialization.checkpoint_hf_repo_id"
        )

    checkpoint_filename = _select_value(cfg=cfg, key="initialization.checkpoint_filename")
    if _is_missing_value(value=checkpoint_filename):
        raise ValueError("bridge initialization requires initialization.checkpoint_filename")


def _validate_latency_values(cfg: Any) -> None:
    latency_values = _require_non_empty_value(cfg=cfg, key="rl_games.env_eval.latency.values")
    if len(latency_values) == 0:
        raise ValueError("Missing required RL-games config field: rl_games.env_eval.latency.values")


def sync_kv_memory_obs_window(cfg: Any) -> None:
    """Tie the observation window to the KV-memory rollout length.

    ``_forward_memory`` unrolls ``len(example["image"])`` steps, and that frame
    count is the dataloader's observation window
    (``datasets.vla_data.num_obs_frames`` driving the video ``delta_indices``).
    When the KV memory is enabled but the window is left at the single-frame
    default, the streaming rollout silently collapses to one step. Derive the
    window — and the multi-frame image mode that emits a frame sequence — from the
    KV config so the two cannot drift apart.
    """
    if not bool(_select_value(cfg=cfg, key="framework.kv_memory.enabled") or False):
        return
    window = int(_select_value(cfg=cfg, key="framework.kv_memory.window") or 0)
    rollout_len = int(_select_value(cfg=cfg, key="framework.kv_memory.rollout_len") or 0)
    if rollout_len <= 0:
        # Mirror QwenOFT's default when rollout_len is unset.
        rollout_len = max(window + 1, 2)
    OmegaConf.update(cfg, "datasets.vla_data.num_obs_frames", rollout_len, force_add=True)
    OmegaConf.update(cfg, "datasets.vla_data.image_mode", "multiframe", force_add=True)
    # Flag the dataloader to emit per-frame KV-memory supervision fields (valid mask,
    # per-frame action chunks, density weights) and widen the action delta window.
    # This distinguishes KV-memory rollouts from plain multi-frame stacking.
    OmegaConf.update(cfg, "datasets.vla_data.kv_memory", True, force_add=True)
    # Read the per-row density weight column the exporter writes (clean_v1 event
    # balancing). Absent column -> uniform weights (see datasets._kv_density_weight).
    OmegaConf.update(cfg, "datasets.vla_data.density_weight_key", "density_weight", force_add=True)


def validate_rl_games_config(cfg: Any) -> None:
    _require_non_empty_value(cfg=cfg, key="rl_games.model_alias")
    _require_non_empty_value(cfg=cfg, key="rl_games.task")
    _require_non_empty_value(cfg=cfg, key="dataset.converted_name")
    _require_non_empty_value(cfg=cfg, key="base_model.repo_id")
    _validate_latency_values(cfg=cfg)

    if _is_bridge_initialization(cfg=cfg):
        _validate_bridge_initialization(cfg=cfg)

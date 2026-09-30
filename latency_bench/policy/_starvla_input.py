"""Resolve saved StarVLA model configuration and temporal image inputs."""
from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from latency_bench.core.clock import EnvClock
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


def _normalized_model_cfg(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    _ensure_starvla_path()
    from omegaconf import OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat

    cfg = OmegaConf.create(model_cfg)
    apply_config_compat(cfg)
    _apply_model_family_include_state_compat(cfg)
    return OmegaConf.to_container(cfg, resolve=True)



def _load_starvla_model_config(path: str | Path) -> dict[str, Any]:
    from omegaconf import OmegaConf

    return _normalized_model_cfg(OmegaConf.load(path))



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



def _ensure_starvla_path() -> None:
    starvla_root = str(STARVLA_ROOT)
    if starvla_root not in sys.path:
        sys.path.insert(0, starvla_root)



def _observation_stride_raw_frames(config: Mapping[str, Any]) -> int:
    env_cfg = config["env"]
    return EnvClock(
        env_fps=float(env_cfg["env_fps"]),
        obs_fps=float(env_cfg["obs_fps"]),
    ).obs_stride_raw_frames



@dataclass(frozen=True)
class ModelInputPlan:
    """The temporal frame selection shared by environment setup and inference."""

    packed: bool
    frame_count: int
    stride: int
    image_mode: str
    stitch_grid: tuple[int, int]
    kv_memory: bool

    @property
    def raw_span(self) -> int:
        return 1 + (self.frame_count - 1) * self.stride


def resolve_model_input_plan(model_cfg: Mapping[str, Any], stride: int) -> ModelInputPlan:
    data = model_cfg["datasets"]["vla_data"]
    # Saved model configs default to one image; packed and temporal inputs opt in.
    packed = data["pack_image_sequence"] if "pack_image_sequence" in data else False
    image_mode = data["image_mode"] if "image_mode" in data else "single"
    # Input-only packed configs may omit the model framework entirely.
    memory = (model_cfg["framework"]["kv_memory"]
              if "framework" in model_cfg and "kv_memory" in model_cfg["framework"] else {})
    kv_memory = memory["enabled"] if "enabled" in memory else False
    count = (data["image_sequence_length"] if packed else
             data["num_obs_frames"] if image_mode != "single" and not kv_memory else 1)
    return ModelInputPlan(packed, count, stride, image_mode,
                          tuple(data["stitch_grid"]) if "stitch_grid" in data else (2, 2),
                          kv_memory)


def apply_starvla_model_input_config(config, *, model_cfg, image_transform="raw_rgb") -> None:
    """Allocate the raw frame window selected by the saved model input plan."""
    plan = resolve_model_input_plan(model_cfg, _observation_stride_raw_frames(config))
    raw_transform = image_transform.strip().lower() in {"", "none", "raw", "raw_rgb"}
    if plan.packed and not raw_transform:
        raise ValueError("WanOFT packed image sequences require image_transform=raw_rgb")
    if raw_transform:
        config["env"]["frame_stack"] = max(config["env"]["frame_stack"], plan.raw_span)

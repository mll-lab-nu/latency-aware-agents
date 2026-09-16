"""Build the shared StarVLA framework configuration."""

from __future__ import annotations

from typing import Any


_FRAMEWORK_NAMES = {
    "qwenoft": "QwenOFT",
    "qwenpi_v3": "QwenPI_v3",
    "qwengr00t": "QwenGR00T",
    "wanoft": "WanOFT",
}


def make_model_config(
    model: str,
    *,
    state_dim: int,
    action_dim: int,
    backbone_path: str,
    action_horizon: int = 1,
    task_objective: Any = None,
) -> dict[str, Any]:
    """Return the framework section for one continuous StarVLA model."""

    framework_name = _FRAMEWORK_NAMES[model]
    action_model: dict[str, Any] = {
        "action_dim": action_dim,
        "action_env_dim": action_dim,
        "state_dim": state_dim,
        "action_horizon": action_horizon,
        "future_action_window_size": action_horizon - 1,
        "past_action_window_size": 0,
        "task_objective": task_objective,
    }
    if model in {"qwenoft", "qwenpi_v3", "wanoft"}:
        action_model["state_encoding"] = "continuous_projector"
    if model in {"qwenoft", "wanoft"}:
        action_model["loss_type"] = "l1"

    config: dict[str, Any] = {
        "name": framework_name,
        "qwenvl": {"base_vlm": backbone_path, "enable_gradient_checkpointing": True},
        "action_model": action_model,
    }
    if model == "wanoft":
        config["world_model"] = {"base_wm": backbone_path}
    return config

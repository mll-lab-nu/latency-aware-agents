from __future__ import annotations

from typing import Dict


MODEL_ALIAS_TO_FRAMEWORK: Dict[str, str] = {
    "openvla": "QwenOFT",
    "pi-0": "QwenPI",
    "pi-0.5": "QwenPI_v3",
    "gr00t": "QwenGR00T",
    "wan_oft": "WanOFT",
}


def apply_model_alias(cfg) -> None:
    """Resolve rl_games.model_alias -> framework.name in-place if provided."""
    rl_games = getattr(cfg, "rl_games", None)
    if rl_games is None:
        return

    model_alias = getattr(rl_games, "model_alias", None)
    if not model_alias:
        return

    if model_alias not in MODEL_ALIAS_TO_FRAMEWORK:
        valid_aliases = ", ".join(sorted(MODEL_ALIAS_TO_FRAMEWORK))
        raise ValueError(f"Unknown rl_games.model_alias={model_alias!r}. Valid aliases: {valid_aliases}")

    cfg.framework.name = MODEL_ALIAS_TO_FRAMEWORK[model_alias]

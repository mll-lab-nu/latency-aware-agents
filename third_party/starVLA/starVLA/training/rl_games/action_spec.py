from __future__ import annotations

BRIDGE_ACTION_DIM = 7
BRIDGE_INIT_MODES = {"pre-trained", "pretrained", "bridge"}
BRIDGE_CHUNK_PRESERVING_MODEL_ALIASES = {"wan_oft"}
DEADLY_LOSS_TYPE_ALIASES = {
    "l1": "l1",
    "mae": "l1",
    "current_multibinary_bce": "current_multibinary_bce",
    "current_bce": "current_multibinary_bce",
    "multibinary_ce": "multibinary_bce",
    "multibinary_bce": "multibinary_bce",
    "bce": "multibinary_bce",
    "binary_cross_entropy": "multibinary_bce",
}
ASTERIX_ACTION_LAYOUT_ALIASES = {
    "discrete_9": "discrete_9",
    "joint_9": "discrete_9",
    "asterix_discrete_9": "discrete_9",
    "factorized_6": "factorized_6",
    "factorized6": "factorized_6",
    "asterix_factorized_6": "factorized_6",
    "asterix_factorized6": "factorized_6",
}


def _deadly_action_dim(cfg) -> int:
    deadly_cfg = getattr(getattr(cfg.rl_games, "env_eval", None), "deadly", None)
    layout = str(getattr(deadly_cfg, "action_layout", "multibinary_7"))
    if layout == "multibinary_7":
        return 7
    if layout == "factorized_11":
        return 11
    raise ValueError(f"Unsupported deadly action layout: {layout}")


def _asterix_action_dim(cfg) -> int:
    asterix_cfg = getattr(getattr(cfg.rl_games, "env_eval", None), "asterix", None)
    layout = str(getattr(asterix_cfg, "action_layout", "discrete_9")).strip().lower()
    layout = ASTERIX_ACTION_LAYOUT_ALIASES.get(layout, layout)
    if layout == "discrete_9":
        return 9
    if layout == "factorized_6":
        return 6
    raise ValueError(f"Unsupported Asterix action layout: {layout}")


def _env_action_dim(cfg):
    rl_games = getattr(cfg, "rl_games", None)
    if rl_games is None:
        return None
    task = str(getattr(rl_games, "task", "flappy"))
    if task == "flappy":
        return 2
    if task == "demon_attack":
        return 6
    if task == "air_raid":
        return 6
    if task == "defend_the_line":
        return 6
    if task == "asterix":
        return _asterix_action_dim(cfg)
    if task == "atlantis":
        return 4
    if task == "deadly_corridor":
        return _deadly_action_dim(cfg)
    if task == "gymnasium":
        return cfg.framework.action_model.action_env_dim
    if task == "cross_task":
        return None
    return None


def _normalize_deadly_loss_type(value) -> str:
    normalized = str(value).strip().lower()
    if normalized not in DEADLY_LOSS_TYPE_ALIASES:
        supported = "|".join(sorted(DEADLY_LOSS_TYPE_ALIASES))
        raise ValueError(f"Unsupported Deadly Corridor loss type: {value!r}. Expected one of: {supported}")
    return DEADLY_LOSS_TYPE_ALIASES[normalized]


def _apply_deadly_loss_spec(cfg, action_cfg) -> None:
    rl_games = getattr(cfg, "rl_games", None)
    if rl_games is None:
        return
    if str(getattr(rl_games, "task", "flappy")) != "deadly_corridor":
        return

    loss_type = getattr(rl_games, "deadly_corridor_loss_type", None)
    if loss_type in (None, ""):
        return
    action_cfg.loss_type = _normalize_deadly_loss_type(loss_type)


def _initialization_mode(cfg) -> str:
    rl_games = getattr(cfg, "rl_games", None)
    if rl_games is None:
        return "scratch"
    return str(getattr(rl_games, "initialization_mode", "scratch") or "scratch").lower()


def _action_carrier(cfg) -> str:
    rl_games = getattr(cfg, "rl_games", None)
    if rl_games is None:
        return "native"
    return str(getattr(rl_games, "action_carrier", "native") or "native").lower()


def _is_bridge_initialization(cfg) -> bool:
    return _initialization_mode(cfg) in BRIDGE_INIT_MODES or _action_carrier(cfg) == "bridge"


def apply_action_spec(cfg) -> None:
    """Apply env-specific action dimensionality policy for RL-games setups."""
    rl_games = getattr(cfg, "rl_games", None)
    framework = getattr(cfg, "framework", None)
    if rl_games is None or framework is None or not hasattr(framework, "action_model"):
        return

    env_dim = _env_action_dim(cfg)
    if env_dim is None:
        return

    action_cfg = framework.action_model
    model_alias = str(getattr(rl_games, "model_alias", "openvla"))
    bridge_init = _is_bridge_initialization(cfg)

    if bridge_init:
        carrier_dim = BRIDGE_ACTION_DIM
        if env_dim > carrier_dim:
            raise ValueError(
                f"Bridge initialization uses a {carrier_dim}D action carrier, but task="
                f"{getattr(rl_games, 'task', None)} resolved active action dim={env_dim}. "
                "Use native/scratch initialization, or a task layout that fits the 7D bridge carrier."
            )
        if model_alias not in BRIDGE_CHUNK_PRESERVING_MODEL_ALIASES:
            action_cfg.action_horizon = 1
            action_cfg.future_action_window_size = 0
            if hasattr(action_cfg, "past_action_window_size"):
                action_cfg.past_action_window_size = 0

        # Bridge mode is intentionally a shared 7D carrier across models.
        # Loss and rollout decode use only the first action_env_dim task dims.
        action_cfg.action_dim = int(carrier_dim)
        action_cfg.action_env_dim = int(env_dim)
        _apply_deadly_loss_spec(cfg, action_cfg)
        return

    if model_alias in {"pi-0", "pi-0.5", "gr00t"}:
        model_action_dim = int(getattr(action_cfg, "action_dim", 0))
        if model_action_dim < env_dim:
            raise ValueError(
                f"Model action_dim={model_action_dim} is smaller than env action dim={env_dim} "
                f"for model_alias={model_alias}, task={getattr(rl_games, 'task', None)}"
            )
        action_cfg.action_env_dim = int(env_dim)
        _apply_deadly_loss_spec(cfg, action_cfg)
        return

    # openvla and other dense/discrete heads should emit env action space directly.
    action_cfg.action_dim = int(env_dim)
    action_cfg.action_env_dim = int(env_dim)
    _apply_deadly_loss_spec(cfg, action_cfg)

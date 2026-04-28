from __future__ import annotations

from collections.abc import Callable
from typing import Any

from latency_bench.core.types import Action
from latency_bench.envs.base import DummyEnvAdapter, EnvAdapter
from latency_bench.envs.flappy import FlappyBirdEnvAdapter


EnvAdapterBuilder = Callable[[dict[str, Any], Action], EnvAdapter]

_ENV_ADAPTER_BUILDERS: dict[str, EnvAdapterBuilder] = {}


def register_env_adapter(name: str) -> Callable[[EnvAdapterBuilder], EnvAdapterBuilder]:
    normalized_name = _normalize_env_name(name)

    def decorator(builder: EnvAdapterBuilder) -> EnvAdapterBuilder:
        if normalized_name in _ENV_ADAPTER_BUILDERS:
            raise ValueError(f"Env adapter already registered: {normalized_name}")
        _ENV_ADAPTER_BUILDERS[normalized_name] = builder
        return builder

    return decorator


def build_env_adapter(config: dict[str, Any], noop_action: Action) -> EnvAdapter:
    env_cfg = config["env"]
    env_name = _normalize_env_name(env_cfg["name"])
    builder = _ENV_ADAPTER_BUILDERS.get(env_name)
    if builder is None:
        registered = ", ".join(get_registered_env_names()) or "<none>"
        raise ValueError(f"Unsupported env.name: {env_name}. Registered env adapters: {registered}")
    return builder(config, noop_action)


def get_registered_env_names() -> list[str]:
    return sorted(_ENV_ADAPTER_BUILDERS)


def _normalize_env_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("env.name cannot be empty")
    return normalized


@register_env_adapter("dummy")
def _build_dummy_env(config: dict[str, Any], noop_action: Action) -> EnvAdapter:
    env_cfg = config["env"]
    experiment_cfg = config["experiment"]
    return DummyEnvAdapter(
        fps_env=float(env_cfg["fps_env"]),
        max_steps=int(experiment_cfg.get("max_env_steps", 10000)),
        noop_action=noop_action,
    )


@register_env_adapter("flappy")
def _build_flappy_env(config: dict[str, Any], noop_action: Action) -> EnvAdapter:
    env_cfg = config["env"]
    screen_size = env_cfg.get("screen_size")
    if screen_size is not None:
        screen_size = tuple(int(value) for value in screen_size)
    return FlappyBirdEnvAdapter(
        gym_id=env_cfg.get("gym_id", "FlappyBird-v0"),
        fps_env=float(env_cfg["fps_env"]),
        noop_action=noop_action,
        render_mode=env_cfg.get("render_mode", "rgb_array"),
        observation_mode=env_cfg.get("observation_mode", "image"),
        use_lidar=bool(env_cfg.get("use_lidar", False)),
        normalize_obs=bool(env_cfg.get("normalize_obs", True)),
        audio_on=bool(env_cfg.get("audio_on", False)),
        screen_size=screen_size,
        pipe_gap=int(env_cfg.get("pipe_gap", 100)),
        bird_color=env_cfg.get("bird_color", "yellow"),
        pipe_color=env_cfg.get("pipe_color", "green"),
        background=env_cfg.get("background", "day"),
        score_limit=env_cfg.get("score_limit"),
        debug=bool(env_cfg.get("debug", False)),
    )


@register_env_adapter("demon_attack")
def _build_demon_attack_env(config: dict[str, Any], noop_action: Action) -> EnvAdapter:
    from latency_bench.envs.demon_attack import DemonAttackEnvAdapter

    env_cfg = config["env"]
    return DemonAttackEnvAdapter(
        env_id=env_cfg.get("env_id", env_cfg.get("gym_id", "ALE/DemonAttack-v5")),
        fps_env=float(env_cfg["fps_env"]),
        noop_action=noop_action,
        screen_size=int(env_cfg.get("screen_size", 84)),
        noop_max=int(env_cfg.get("noop_max", 30)),
        terminal_on_life_loss=bool(env_cfg.get("terminal_on_life_loss", False)),
        clip_reward=bool(env_cfg.get("clip_reward", True)),
        repeat_action_probability=float(env_cfg.get("repeat_action_probability", 0.0)),
        full_action_space=bool(env_cfg.get("full_action_space", False)),
    )

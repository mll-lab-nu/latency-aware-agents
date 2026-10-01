from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from latency_bench.core.clock import EnvClock
from latency_bench.core.types import Action
from latency_bench.envs.base import DummyEnvAdapter, EnvAdapter


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


def _find_key_path(value: Any, target_key: str, prefix: str = "") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) == target_key:
                return path
            found = _find_key_path(item, target_key, path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = _find_key_path(item, target_key, f"{prefix}[{index}]")
            if found is not None:
                return found
    return None


def _uses_raw_rgb_vla_policy(config: dict[str, Any]) -> bool:
    policy_type = str(config["policy"]["type"]).strip().lower()
    return policy_type == "starvla"


def _exports_raw_rgb_frames(config: dict[str, Any]) -> bool:
    return bool(config["env"].get("export_env_raw_rgb_frames", False)) or _uses_raw_rgb_vla_policy(config)


@register_env_adapter("dummy")
def _build_dummy_env(config: dict[str, Any], noop_action: Action) -> EnvAdapter:
    env_cfg = config["env"]
    evaluation_cfg = config.get("evaluation", {})
    return DummyEnvAdapter(
        env_fps=float(env_cfg.get("env_fps", 30.0)),
        max_steps=int(evaluation_cfg.get("eval_max_steps", env_cfg.get("max_steps", 5))),
        noop_action=noop_action,
    )


@register_env_adapter("gymnasium")
def _build_gymnasium_env(config: dict[str, Any], noop_action: Action) -> EnvAdapter:
    from latency_bench.envs.gymnasium import GymnasiumEnvAdapter

    env_cfg = config["env"]
    return GymnasiumEnvAdapter(
        env_id=env_cfg["env_id"],
        registration_imports=env_cfg["registration_imports"],
        make_kwargs=env_cfg["make_kwargs"],
        env_fps=env_cfg["env_fps"],
        noop_action=noop_action,
        frame_stack=env_cfg["frame_stack"],
        export_env_raw_rgb_frames=_exports_raw_rgb_frames(config),
        include_state="state_space" in env_cfg,
    )


@register_env_adapter("mikasa_intercept_grab_fast")
def _build_mikasa_env(config: dict[str, Any], noop_action: Action) -> EnvAdapter:
    # ManiSkill is installed in the environment interpreter, not the model worker.
    from scripts.mikasa.evaluate import MikasaEnvStepBackend
    from latency_bench.executors.env_step_backend import EnvStepBackendEnvAdapter

    return EnvStepBackendEnvAdapter(MikasaEnvStepBackend(config=config, num_slots=1))


@register_env_adapter("balance_profile")
def _build_balance_profile_env(config: dict[str, Any], noop_action: Action) -> EnvAdapter:
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    from latency_bench.envs.gymnasium import BalanceProfileEnvAdapter

    env = config["env"]
    method = config["latency"]["method"]
    context_frames = None
    if method == "zero":
        context_frames = 0
    elif method == "fixed":
        context_frames = EnvClock(env_fps=env["env_fps"], obs_fps=env["obs_fps"]).latency_ms_to_raw_frames(
            config["latency"]["fixed_latency_ms"])
    return BalanceProfileEnvAdapter(
        env_id=env["env_id"], registration_imports=env["registration_imports"],
        make_kwargs=env["make_kwargs"], env_fps=env["env_fps"], noop_action=noop_action,
        frame_stack=env["frame_stack"], export_env_raw_rgb_frames=True, include_state=True,
        action_horizon=env["action_context_horizon"], context_frames=context_frames,
    )


@register_env_adapter("flappy")
def _build_flappy_env(config: dict[str, Any], noop_action: Action) -> EnvAdapter:
    env_cfg = config["env"]
    policy_type = str(config.get("policy", {}).get("type", "")).strip().lower()
    screen_size = env_cfg.get("screen_size")
    screen_size_path = _find_key_path(config, "screen_size") if policy_type == "flappy_sf" else None
    if screen_size_path is not None:
        raise ValueError(
            f"{screen_size_path} is not supported for Flappy Sample Factory eval; "
            "use env.obs_resize to resize observations without changing env geometry"
        )
    if screen_size is not None:
        screen_size = tuple(int(value) for value in screen_size)
    obs_resize = env_cfg.get("obs_resize")
    if obs_resize is not None:
        obs_resize = tuple(int(value) for value in obs_resize)
    from latency_bench.envs.flappy import FlappyBirdEnvAdapter

    return FlappyBirdEnvAdapter(
        gym_id=env_cfg.get("gym_id", "FlappyBird-v0"),
        env_fps=float(env_cfg["env_fps"]),
        noop_action=noop_action,
        screen_size=screen_size,
        pipe_gap=int(env_cfg.get("pipe_gap", 100)),
        bird_color=env_cfg.get("bird_color", "yellow"),
        pipe_color=env_cfg.get("pipe_color", "green"),
        background=env_cfg.get("background", "day"),
        score_limit=env_cfg.get("score_limit"),
        obs_resize=obs_resize,
        frame_stack=int(env_cfg.get("frame_stack", 1) or 1),
        use_gpu_render=bool(env_cfg.get("use_gpu_render", False)),
        gpu_render_device=str(env_cfg.get("gpu_render_device", "auto")),
        gpu_render_profile=bool(env_cfg.get("gpu_render_profile", False)),
        gpu_render_profile_interval=int(env_cfg.get("gpu_render_profile_interval", 1000) or 1000),
        policy_observation_layout="chw" if policy_type == "flappy_sf" else "hwc",
        export_env_raw_rgb_frames=_exports_raw_rgb_frames(config),
    )


@register_env_adapter("demon_attack")
def _build_demon_attack_env(config: dict[str, Any], noop_action: Action) -> EnvAdapter:
    from latency_bench.envs.demon_attack import DemonAttackEnvAdapter

    env_cfg = config["env"]
    eval_raw_reward = bool(config.get("evaluation", {}).get("eval_raw_reward", False))
    is_latency_bench_eval = str(config.get("backend", {}).get("run_mode", "")).lower() == "eval"
    terminal_on_life_loss = env_cfg["terminal_on_life_loss"]
    clip_reward = env_cfg["clip_reward"]
    if is_latency_bench_eval and eval_raw_reward:
        clip_reward = False
    continue_life_loss_terminal = (
        terminal_on_life_loss and is_latency_bench_eval and eval_raw_reward
    )
    return DemonAttackEnvAdapter(
        env_id=env_cfg.get("env_id", env_cfg.get("gym_id", "ALE/DemonAttack-v5")),
        env_fps=float(env_cfg["env_fps"]),
        noop_action=noop_action,
        screen_size=int(env_cfg.get("screen_size", 84)),
        noop_max=int(env_cfg.get("noop_max", 30)),
        reset_mode=env_cfg["reset_mode"],
        terminal_on_life_loss=terminal_on_life_loss,
        clip_reward=clip_reward,
        continue_life_loss_terminal=continue_life_loss_terminal,
        repeat_action_probability=float(env_cfg.get("repeat_action_probability", 0.0)),
        full_action_space=bool(env_cfg.get("full_action_space", False)),
        expected_action_meanings=env_cfg.get("expected_action_meanings"),
        frame_stack=int(env_cfg.get("frame_stack", 1) or 1),
        export_env_raw_rgb_frames=_exports_raw_rgb_frames(config),
    )


@register_env_adapter("deadly_corridor")
@register_env_adapter("defend_the_line")
def _build_deadly_corridor_env(config: dict[str, Any], noop_action: Action) -> EnvAdapter:
    # StarVLA evals with a multibinary head need the gymnasium MultiBinary env
    # (matching eval_core) so the model can fire arbitrary button subsets. The
    # sample_factory factorised env can only emit one button per category and is
    # kept for teacher (sample_factory) eval/training. Scoped to StarVLA only.
    action_layout = str(config.get("policy", {}).get("action_layout", "") or "").strip().lower()
    if _uses_raw_rgb_vla_policy(config) and action_layout == "multibinary_7":
        from latency_bench.envs.deadly_corridor import DeadlyCorridorVlaEnvAdapter

        return DeadlyCorridorVlaEnvAdapter(
            config=config,
            noop_action=None,  # adapter derives an all-zeros multibinary no-op
            export_env_raw_rgb_frames=_exports_raw_rgb_frames(config),
        )

    from latency_bench.envs.deadly_corridor import DeadlyCorridorEnvAdapter

    return DeadlyCorridorEnvAdapter(
        config=config,
        noop_action=noop_action,
        export_env_raw_rgb_frames=_exports_raw_rgb_frames(config),
    )

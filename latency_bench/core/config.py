from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from latency_bench.core.clock import compute_obs_stride_raw_frames
from latency_bench.core.types import ExecutorMode, LatencySourceType, PersistenceMode, StaleActionMode


REQUIRED_TOP_LEVEL_KEYS = (
    "experiment",
    "executor",
    "env",
    "control",
    "latency",
    "scheduler",
    "policy",
    "logging",
)


def merge_dicts(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "__base__":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict:
    config = _load_config_unvalidated(Path(path))
    validate_config(config)
    return config


def _load_config_unvalidated(path: Path) -> dict:
    import yaml

    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        current = yaml.safe_load(f) or {}
    if not isinstance(current, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    base_ref = current.get("__base__")
    if not base_ref:
        return {k: v for k, v in current.items() if k != "__base__"}

    if isinstance(base_ref, (str, Path)):
        base_paths = [base_ref]
    elif isinstance(base_ref, list):
        base_paths = base_ref
    else:
        raise ValueError("__base__ must be a path string or list of path strings")

    merged: dict[str, Any] = {}
    for base_item in base_paths:
        base_path = Path(base_item)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = merge_dicts(merged, _load_config_unvalidated(base_path))

    return merge_dicts(merged, current)


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    experiment = _require_dict(config, "experiment")
    executor = _require_dict(config, "executor")
    env = _require_dict(config, "env")
    control = _require_dict(config, "control")
    latency = _require_dict(config, "latency")
    scheduler = _require_dict(config, "scheduler")
    policy = _require_dict(config, "policy")
    logging = _require_dict(config, "logging")
    evaluation = config.get("evaluation")
    realtime = config.get("realtime")
    clock_config = resolve_clock_config(config)
    config["clock"] = clock_config
    env.setdefault("fps_env", clock_config["env_fps"])
    control.setdefault("frequency_hz", clock_config["obs_fps"])
    control.setdefault("observe_every_n_frames", clock_config["obs_stride_raw_frames"])

    _require_field(experiment, "name")
    executor_mode = _require_field(executor, "mode")
    _validate_enum_value(executor_mode, ExecutorMode, "executor.mode")
    if "max_pending_observations" in executor and int(executor["max_pending_observations"]) <= 0:
        raise ValueError("executor.max_pending_observations must be positive")
    if realtime is not None:
        if not isinstance(realtime, dict):
            raise ValueError("realtime must be a mapping")
        max_catchup_steps = realtime.get("max_catchup_steps_per_loop")
        if max_catchup_steps is not None and int(max_catchup_steps) <= 0:
            raise ValueError("realtime.max_catchup_steps_per_loop must be positive or null")

    _require_field(env, "name")
    fps_env = _require_field(env, "fps_env")
    if float(fps_env) <= 0:
        raise ValueError("env.fps_env must be positive")
    _require_field(env, "noop_action")

    if "frequency_hz" in control and float(control["frequency_hz"]) <= 0:
        raise ValueError("control.frequency_hz must be positive")
    if "observe_every_n_frames" in control and int(control["observe_every_n_frames"]) <= 0:
        raise ValueError("control.observe_every_n_frames must be positive")

    latency_type = _require_field(latency, "type")
    _validate_enum_value(latency_type, LatencySourceType, "latency.type")
    if latency_type == LatencySourceType.FIXED.value:
        _require_field(latency, "fixed_latency_ms")
    if latency_type == LatencySourceType.TRACE_REPLAY.value:
        _require_field(latency, "trace_path")
    if latency_type in {
        LatencySourceType.EMPIRICAL_SAMPLE.value,
        LatencySourceType.EMPIRICAL_TRACE_REPLAY.value,
    }:
        if "profile_path" not in latency:
            raise ValueError("Missing required field: latency.profile_path")
        profile_path = latency["profile_path"]
        if profile_path is None or str(profile_path).strip() == "":
            raise ValueError("latency.profile_path must be provided")
        if "seed" in latency:
            _validate_latency_seed(latency["seed"])
    if latency_type == LatencySourceType.EMPIRICAL_TRACE_REPLAY.value:
        trace_replay_order = latency.get("trace_replay_order", "sequential")
        if trace_replay_order not in {"sequential", "random"}:
            raise ValueError("latency.trace_replay_order must be one of ['random', 'sequential']")
    if latency_type == LatencySourceType.NORMAL_DISTRIBUTION.value:
        if "mean_latency_ms" not in latency:
            raise ValueError("Missing required field: latency.mean_latency_ms")
        if "std_latency_ms" not in latency:
            raise ValueError("Missing required field: latency.std_latency_ms")
        std_latency_ms = float(latency["std_latency_ms"])
        if std_latency_ms < 0:
            raise ValueError("latency.std_latency_ms must be non-negative")
        min_latency_ms = float(latency.get("min_latency_ms", 0.0))
        if min_latency_ms < 0:
            raise ValueError("latency.min_latency_ms must be non-negative")
        max_latency_ms = latency.get("max_latency_ms")
        if max_latency_ms is not None and float(max_latency_ms) < min_latency_ms:
            raise ValueError("latency.max_latency_ms must be greater than or equal to latency.min_latency_ms")
    if latency_type == LatencySourceType.UNIFORM_DISTRIBUTION.value:
        if "min_latency_ms" not in latency:
            raise ValueError("Missing required field: latency.min_latency_ms")
        if "max_latency_ms" not in latency:
            raise ValueError("Missing required field: latency.max_latency_ms")
        min_latency_ms = float(latency["min_latency_ms"])
        max_latency_ms = float(latency["max_latency_ms"])
        if min_latency_ms < 0:
            raise ValueError("latency.min_latency_ms must be non-negative")
        if max_latency_ms < min_latency_ms:
            raise ValueError("latency.max_latency_ms must be greater than or equal to latency.min_latency_ms")
        if "seed" in latency:
            _validate_latency_seed(latency["seed"])

    if "persistence_mode" in scheduler:
        _validate_enum_value(scheduler["persistence_mode"], PersistenceMode, "scheduler.persistence_mode")
    if "stale_action_mode" in scheduler:
        _validate_enum_value(scheduler["stale_action_mode"], StaleActionMode, "scheduler.stale_action_mode")

    _require_field(policy, "type")
    _require_field(logging, "output_dir")
    if isinstance(evaluation, dict) and "latency_bench_env_backend" in evaluation:
        backend = str(evaluation["latency_bench_env_backend"]).strip().lower()
        allowed_backends = {"serial", "subproc", "auto"}
        if backend not in allowed_backends:
            raise ValueError(
                "evaluation.latency_bench_env_backend must be one of "
                f"{sorted(allowed_backends)}"
            )


def resolve_clock_config(config: dict) -> dict[str, float | int]:
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    env = _require_dict(config, "env")
    control = _require_dict(config, "control")
    clock = config.get("clock", {})
    if clock is None:
        clock = {}
    if not isinstance(clock, dict):
        raise ValueError("clock must be a mapping")

    clock_env_fps = clock.get("env_fps")
    legacy_env_fps = env.get("fps_env")
    if clock_env_fps is not None and legacy_env_fps is not None:
        if _numeric_conflict(clock_env_fps, legacy_env_fps):
            raise ValueError(
                f"clock.env_fps conflicts with env.fps_env: "
                f"clock.env_fps={clock_env_fps}, env.fps_env={legacy_env_fps}"
            )
    if clock_env_fps is None and legacy_env_fps is None:
        raise ValueError("Missing required field: clock.env_fps or env.fps_env")
    env_fps = float(clock_env_fps if clock_env_fps is not None else legacy_env_fps)

    clock_stride = clock.get("obs_stride_raw_frames")
    legacy_obs_fps, legacy_obs_source = _legacy_obs_fps(env_fps, control)
    if "obs_fps" in clock:
        obs_fps = float(clock["obs_fps"])
        if legacy_obs_source is not None and _numeric_conflict(obs_fps, legacy_obs_fps):
            raise ValueError(
                f"clock.obs_fps conflicts with {legacy_obs_source}: "
                f"clock.obs_fps={obs_fps}, {legacy_obs_source}={legacy_obs_fps}"
            )
    elif legacy_obs_source is not None:
        obs_fps = legacy_obs_fps
    elif clock_stride is not None:
        stride = _coerce_positive_int(clock_stride, "clock.obs_stride_raw_frames")
        obs_fps = env_fps / stride
    else:
        obs_fps = env_fps

    computed_stride = compute_obs_stride_raw_frames(env_fps, obs_fps)
    if clock_stride is not None:
        explicit_stride = _coerce_positive_int(clock_stride, "clock.obs_stride_raw_frames")
        if explicit_stride != computed_stride:
            raise ValueError(
                f"clock.obs_stride_raw_frames conflicts with clock rates: "
                f"clock.obs_stride_raw_frames={explicit_stride}, computed={computed_stride}"
            )

    return {
        "env_fps": float(env_fps),
        "obs_fps": float(obs_fps),
        "obs_stride_raw_frames": int(computed_stride),
    }


def _legacy_obs_fps(env_fps: float, control: dict[str, Any]) -> tuple[float, str | None]:
    if "observe_every_n_frames" in control and int(control["observe_every_n_frames"]) > 1:
        stride = _coerce_positive_int(control["observe_every_n_frames"], "control.observe_every_n_frames")
        return float(env_fps) / stride, "control.observe_every_n_frames"
    if "frequency_hz" in control:
        return float(control["frequency_hz"]), "control.frequency_hz"
    return float(env_fps), None


def _coerce_positive_int(value: Any, field_name: str) -> int:
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if int_value <= 0 or not math.isclose(float(value), float(int_value), rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"{field_name} must be a positive integer")
    return int_value


def _numeric_conflict(left: Any, right: Any) -> bool:
    return not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


def _require_dict(config: dict, key: str) -> dict:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _require_field(config: dict, key: str) -> Any:
    if key not in config:
        raise ValueError(f"Missing required field: {key}")
    return config[key]


def _validate_enum_value(value: Any, enum_type: type, field_name: str) -> None:
    allowed = {item.value for item in enum_type}
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")


def _validate_latency_seed(seed: Any) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("latency.seed must be an int or null")

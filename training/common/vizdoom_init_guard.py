from __future__ import annotations

from dataclasses import dataclass
import inspect
import random
import time
from pathlib import Path
from types import MethodType
from typing import Any, Iterable


_TARGET_ENV_NAME = "doom_deadly_corridor"
_INSTALLED_ATTR = "_latency_bench_vizdoom_guard_installed"
_ORIGINAL_GAME_INIT_ATTR = "_latency_bench_original_game_init"
_CONFIG_ATTR = "_latency_bench_vizdoom_init_guard_config"
_CONTEXT_ATTR = "_latency_bench_vizdoom_init_guard_context"


@dataclass(frozen=True)
class VizdoomInitGuardConfig:
    max_parallel: int = 10
    lock_timeout_sec: float = 5.0
    max_wait_sec: float = 60.0
    retry_jitter_sec: float = 0.25

    @classmethod
    def from_cfg(cls, cfg: object) -> "VizdoomInitGuardConfig":
        return cls(
            max_parallel=_cfg_int(cfg, "vizdoom_init_max_parallel", cls.max_parallel, minimum=1),
            lock_timeout_sec=_cfg_float(
                cfg,
                "vizdoom_init_lock_timeout_sec",
                cls.lock_timeout_sec,
                minimum=0.001,
            ),
            max_wait_sec=_cfg_float(cfg, "vizdoom_init_lock_max_wait_sec", cls.max_wait_sec, minimum=0.001),
            retry_jitter_sec=_cfg_float(
                cfg,
                "vizdoom_init_retry_jitter_sec",
                cls.retry_jitter_sec,
                minimum=0.0,
            ),
        )


def ensure_vizdoom_init_guard_installed(
    env: object,
    *,
    env_name: str,
    cfg: object,
    env_config: object | None = None,
) -> bool:
    """Install the ViZDoom init lock guard on this Deadly Corridor env instance."""

    if str(env_name) != _TARGET_ENV_NAME:
        return False

    target = _find_vizdoom_env(env)
    if target is None:
        return False

    _validate_vizdoom_game_init_signature(type(target))
    config = VizdoomInitGuardConfig.from_cfg(cfg)
    setattr(target, _CONFIG_ATTR, config)
    setattr(
        target,
        _CONTEXT_ATTR,
        {
            "env_name": str(env_name),
            "env_config": env_config,
        },
    )

    if bool(getattr(target, _INSTALLED_ATTR, False)):
        return True

    setattr(target, _ORIGINAL_GAME_INIT_ATTR, target._game_init)
    setattr(target, "_game_init", MethodType(_guarded_game_init, target))
    setattr(target, _INSTALLED_ATTR, True)
    return True


def _guarded_game_init(self, with_locking=True, max_parallel=10):
    from filelock import FileLock, Timeout
    from sample_factory.envs.env_utils import EnvCriticalError
    from sample_factory.utils.utils import log
    from sf_examples.vizdoom.doom import doom_gym

    if not with_locking:
        try:
            self.game.init()
            return
        except Exception as exc:
            log.warning("VizDoom game.init() threw an exception %r. Terminate process...", exc)
            raise EnvCriticalError() from exc

    config = getattr(self, _CONFIG_ATTR, VizdoomInitGuardConfig())
    context = getattr(self, _CONTEXT_ATTR, {})
    start_time = time.monotonic()
    deadline = start_time + float(config.max_wait_sec)
    attempts = 0
    tried_lock_paths: list[str] = []
    effective_max_parallel = int(config.max_parallel or max_parallel)

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _lock_wait_exceeded_error(
                context=context,
                config=config,
                attempts=attempts,
                tried_lock_paths=tried_lock_paths,
                elapsed_sec=time.monotonic() - start_time,
            )

        lock_path = str(doom_gym.doom_lock_file(effective_max_parallel))
        tried_lock_paths.append(lock_path)
        attempts += 1

        try:
            with FileLock(lock_path).acquire(timeout=min(float(config.lock_timeout_sec), max(0.001, remaining))):
                self.game.init()
            return
        except Timeout:
            log.debug(
                "Another process currently holds the lock %s, attempt: %d, max_wait_sec=%.3f",
                lock_path,
                attempts,
                float(config.max_wait_sec),
            )
            _sleep_with_jitter(config.retry_jitter_sec, deadline)
        except Exception as exc:
            log.warning("VizDoom game.init() threw an exception %r. Terminate process...", exc)
            raise EnvCriticalError() from exc


def _lock_wait_exceeded_error(
    *,
    context: dict[str, Any],
    config: VizdoomInitGuardConfig,
    attempts: int,
    tried_lock_paths: list[str],
    elapsed_sec: float,
):
    from sample_factory.envs.env_utils import EnvCriticalError

    unique_paths = sorted(set(tried_lock_paths))
    env_config = context.get("env_config")
    message = (
        "ViZDoom initialization lock wait exceeded "
        f"env_name={context.get('env_name', _TARGET_ENV_NAME)} "
        f"{_env_config_details(env_config)} "
        f"elapsed_sec={elapsed_sec:.3f} "
        f"attempts={attempts} "
        f"max_parallel={config.max_parallel} "
        f"lock_timeout_sec={config.lock_timeout_sec:.3f} "
        f"max_wait_sec={config.max_wait_sec:.3f} "
        f"retry_jitter_sec={config.retry_jitter_sec:.3f} "
        f"unique_locks_tried={unique_paths} "
        f"lock_inventory={_lock_inventory(unique_paths)}"
    )
    return EnvCriticalError(message)


def _find_vizdoom_env(env: object):
    from sf_examples.vizdoom.doom.doom_gym import VizdoomEnv

    for candidate in _walk_env_chain(env):
        if isinstance(candidate, VizdoomEnv):
            return candidate
    return None


def _walk_env_chain(env: object) -> Iterable[object]:
    seen: set[int] = set()
    stack = [env]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current

        wrapped_env = getattr(current, "env", None)
        if wrapped_env is not None:
            stack.append(wrapped_env)

        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is not None and unwrapped is not current:
            stack.append(unwrapped)


def _validate_vizdoom_game_init_signature(vizdoom_env_cls: type) -> None:
    signature = inspect.signature(vizdoom_env_cls._game_init)
    parameters = list(signature.parameters.values())
    valid = (
        len(parameters) == 3
        and parameters[0].name == "self"
        and parameters[1].name == "with_locking"
        and parameters[1].default is True
        and parameters[2].name == "max_parallel"
        and parameters[2].default == 10
    )
    if not valid:
        raise RuntimeError(f"Unsupported VizdoomEnv._game_init signature: {signature}")


def _cfg_int(cfg: object, name: str, default: int, *, minimum: int) -> int:
    value = getattr(cfg, name, None)
    if value is None:
        return int(default)
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return parsed


def _cfg_float(cfg: object, name: str, default: float, *, minimum: float) -> float:
    value = getattr(cfg, name, None)
    if value is None:
        return float(default)
    parsed = float(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return parsed


def _sleep_with_jitter(retry_jitter_sec: float, deadline: float) -> None:
    if retry_jitter_sec <= 0:
        return
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    time.sleep(min(random.uniform(0.0, retry_jitter_sec), remaining))


def _env_config_details(env_config: object | None) -> str:
    if env_config is None:
        return "worker_index=None vector_index=None env_id=None"
    return (
        f"worker_index={getattr(env_config, 'worker_index', None)} "
        f"vector_index={getattr(env_config, 'vector_index', None)} "
        f"env_id={getattr(env_config, 'env_id', None)}"
    )


def _lock_inventory(tried_lock_paths: list[str]) -> list[str]:
    roots = {Path(path).expanduser().resolve().parent for path in tried_lock_paths}
    inventory: list[str] = []
    for root in sorted(roots):
        if root.is_dir():
            inventory.extend(str(path) for path in sorted(root.glob("doom_*.lockfile")))
    return inventory

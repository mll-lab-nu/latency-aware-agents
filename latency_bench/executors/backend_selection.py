"""Select and construct the env-step backend for batched simulated eval."""

from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Tuple

from latency_bench.core.types import Action
from latency_bench.envs.base import EnvAdapter
from latency_bench.envs.factory import build_env_adapter
from latency_bench.executors.env_step_backend import (
    EnvStepBackend,
    SerialEnvStepBackend,
    SubprocEnvStepBackend,
    env_action_space,
)

ENV_BACKEND_DEBUG_ENV = "LATENCY_BENCH_ENV_BACKEND"
_AUTO_SUBPROC_ENV_NAMES = {"deadly_corridor", "defend_the_line"}
_AUTO_FLAPPY_GPU_BATCHED_POLICY_TYPES = {"flappy_sf", "starvla"}


def _build_serial_backend(*, config: dict[str, Any], noop_action: Action, num_slots: int) -> EnvStepBackend:
    envs: list[EnvAdapter] = []
    try:
        for _slot in range(num_slots):
            envs.append(build_env_adapter(config, noop_action))
    except Exception:
        for env in envs:
            env.close()
        raise
    return SerialEnvStepBackend(envs)


def _build_subproc_backend(*, config: dict[str, Any], noop_action: Action, num_slots: int) -> EnvStepBackend:
    return SubprocEnvStepBackend(config=config, noop_action=noop_action, num_slots=num_slots)


def _build_flappy_gpu_batched_backend(*, config: dict[str, Any], noop_action: Action, num_slots: int) -> EnvStepBackend:
    from latency_bench.executors.flappy_gpu_batched_backend import FlappyGpuBatchedEnvStepBackend

    return FlappyGpuBatchedEnvStepBackend(config=config, noop_action=noop_action, num_slots=num_slots)


def _build_demon_attack_gpu_batched_backend(*, config: dict[str, Any], noop_action: Action, num_slots: int) -> EnvStepBackend:
    from latency_bench.executors.demon_attack_gpu_batched_backend import DemonAttackGpuBatchedEnvStepBackend

    return DemonAttackGpuBatchedEnvStepBackend(config=config, noop_action=noop_action, num_slots=num_slots)


def _build_atlantis_gpu_batched_backend(*, config: dict[str, Any], noop_action: Action, num_slots: int) -> EnvStepBackend:
    from latency_bench.executors.demon_attack_gpu_batched_backend import AtlantisGpuBatchedEnvStepBackend

    return AtlantisGpuBatchedEnvStepBackend(config=config, noop_action=noop_action, num_slots=num_slots)


def _build_asterix_gpu_batched_backend(*, config: dict[str, Any], noop_action: Action, num_slots: int) -> EnvStepBackend:
    from latency_bench.executors.demon_attack_gpu_batched_backend import AsterixGpuBatchedEnvStepBackend

    return AsterixGpuBatchedEnvStepBackend(config=config, noop_action=noop_action, num_slots=num_slots)


_BACKEND_FACTORIES: dict[str, Callable[..., EnvStepBackend]] = {
    "serial": _build_serial_backend,
    "subproc": _build_subproc_backend,
    "flappy_gpu_batched": _build_flappy_gpu_batched_backend,
    "demon_attack_gpu_batched": _build_demon_attack_gpu_batched_backend,
    "atlantis_gpu_batched": _build_atlantis_gpu_batched_backend,
    "asterix_gpu_batched": _build_asterix_gpu_batched_backend,
}

LATENCY_BENCH_ENV_BACKENDS = frozenset(_BACKEND_FACTORIES) | {"auto"}


def build_env_step_backend(
    backend_name: str,
    *,
    config: dict[str, Any],
    noop_action: Action,
    num_slots: int,
) -> EnvStepBackend:
    return _BACKEND_FACTORIES[backend_name](config=config, noop_action=noop_action, num_slots=num_slots)


def resolve_latency_bench_env_backend(config: dict[str, Any], *, eval_parallel_envs: int) -> str:
    configured = os.environ.get(ENV_BACKEND_DEBUG_ENV)
    if configured is None:
        configured = config.get("evaluation", {}).get("latency_bench_env_backend", "auto")
    backend = str(configured).strip().lower()
    if backend not in LATENCY_BENCH_ENV_BACKENDS:
        raise ValueError(
            "evaluation.latency_bench_env_backend must be one of "
            f"{sorted(LATENCY_BENCH_ENV_BACKENDS)}; got {configured!r}"
        )
    if int(eval_parallel_envs) <= 1:
        return "serial"
    if backend == "auto":
        return auto_latency_bench_env_backend(config)
    return backend


def auto_latency_bench_env_backend(config: dict[str, Any]) -> str:
    env_cfg = config.get("env", {})
    env_name = str(env_cfg.get("name", "")).strip().lower()
    policy_type = str(config.get("policy", {}).get("type", "")).strip().lower()
    if env_name == "flappy" and policy_type == "starvla":
        # starvla consumes only the native raw RGB stack, which is bit-exact
        # between the pygame and torch render paths
        # (tests/executors/test_flappy_cpu_vs_gpu_render_equivalence.py), so
        # batched GPU render is safe regardless of env.use_gpu_render. Other
        # policies (flappy_sf) consume obs.data, whose resize provenance
        # differs between the two paths, and keep the use_gpu_render gate.
        return "flappy_gpu_batched"
    if (
        env_name == "flappy"
        and policy_type in _AUTO_FLAPPY_GPU_BATCHED_POLICY_TYPES
        and bool(env_cfg.get("use_gpu_render", False))
    ):
        return "flappy_gpu_batched"
    if env_name in _AUTO_SUBPROC_ENV_NAMES:
        return "subproc"
    return "serial"


def raise_if_unsupported_batched_action_space(envs: list[EnvAdapter]) -> None:
    for env in envs:
        action_space = env_action_space(env)
        if action_space is None or isinstance(action_space, Discrete):
            continue
        if isinstance(action_space, Box) and len(action_space.shape) == 1:
            continue
        if isinstance(action_space, Tuple):
            if all(isinstance(subspace, Discrete) for subspace in action_space.spaces):
                continue
            raise ValueError(
                "evaluation.eval_parallel_envs > 1 supports only flat Tuple(Discrete, ...) action spaces; "
                f"got unsupported or nested Tuple action space {action_space!r}"
            )
        if isinstance(action_space, MultiDiscrete):
            if np.asarray(action_space.nvec).ndim == 1:
                continue
            raise ValueError(
                "evaluation.eval_parallel_envs > 1 supports only one-dimensional MultiDiscrete action spaces; "
                f"got nvec={action_space.nvec!r}"
            )

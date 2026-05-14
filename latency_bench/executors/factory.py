from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ActionHoldPolicy, ExecutorMode, LatencySourceType, OrderingPolicy
from latency_bench.envs.base import EnvAdapter
from latency_bench.envs.factory import build_env_adapter
from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor
from latency_bench.executors.env_step_backend import SerialEnvStepBackend, SubprocEnvStepBackend
from latency_bench.executors.paused import PausedLatencyExecutor
from latency_bench.executors.realtime import RealtimeLatencyExecutor
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.latency.sources import (
    EmpiricalSampleLatencySource,
    EmpiricalTraceReplayLatencySource,
    FixedLatencySource,
    LatencySource,
    NormalDistributionLatencySource,
    TraceReplayLatencySource,
    UniformDistributionLatencySource,
    ZeroLatencySource,
)
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy.factory import build_policy
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import (
    DecisionScheduler,
    ObservationCadenceDecisionScheduler,
)
from latency_bench.scheduler.persistence import (
    ActionPersistencePolicy,
    HoldPolicy,
    OneFrameThenNoopPolicy,
)


ExecutorBuilder = Callable[[dict[str, Any]], Any]

_EXECUTOR_BUILDERS: dict[ExecutorMode, ExecutorBuilder] = {}
_LATENCY_BENCH_ENV_BACKENDS = {"serial", "subproc", "flappy_gpu_batched", "auto"}
_ENV_BACKEND_DEBUG_ENV = "LATENCY_BENCH_ENV_BACKEND"
_AUTO_SUBPROC_ENV_NAMES = {"demon_attack", "deadly_corridor"}
_SCHEDULER_ALLOWED_KEYS = frozenset({"ordering_policy", "hold_policy"})


def register_executor(mode: ExecutorMode | str) -> Callable[[ExecutorBuilder], ExecutorBuilder]:
    executor_mode = ExecutorMode(mode)

    def decorator(builder: ExecutorBuilder) -> ExecutorBuilder:
        if executor_mode in _EXECUTOR_BUILDERS:
            raise ValueError(f"Executor already registered: {executor_mode.value}")
        _EXECUTOR_BUILDERS[executor_mode] = builder
        return builder

    return decorator


def build_executor(config: dict[str, Any]) -> Any:
    executor_cfg = config["executor"]
    mode = ExecutorMode(executor_cfg["mode"])
    eval_parallel_envs = _eval_parallel_envs(config)
    if eval_parallel_envs > 1 and mode in {ExecutorMode.PAUSED, ExecutorMode.REALTIME}:
        raise ValueError(
            "evaluation.eval_parallel_envs > 1 is only supported for executor.mode=simulated; "
            f"got executor.mode={mode.value}"
        )
    builder = _EXECUTOR_BUILDERS.get(mode)
    if builder is None:
        registered = ", ".join(get_registered_executor_modes()) or "<none>"
        raise NotImplementedError(
            f"executor.mode '{mode.value}' is recognized but not implemented. "
            f"Registered executor modes: {registered}"
        )
    return builder(config)


def get_registered_executor_modes() -> list[str]:
    return sorted(mode.value for mode in _EXECUTOR_BUILDERS)


@register_executor(ExecutorMode.SIMULATED)
def _build_simulated_executor(config: dict[str, Any]) -> SimulatedLatencyExecutor | BatchedSimulatedLatencyExecutor:
    eval_parallel_envs = _eval_parallel_envs(config)
    num_episodes = int(config.get("experiment", {}).get("num_episodes", 1))
    if (
        eval_parallel_envs > 1
        and num_episodes > 0
        and _supports_phase_3b_batched_latency(config.get("latency", {}))
    ):
        return _build_batched_simulated_executor(config, max_parallel_envs=min(eval_parallel_envs, num_episodes))

    return _build_single_simulated_executor(config)


def _build_single_simulated_executor(config: dict[str, Any]) -> SimulatedLatencyExecutor:
    scheduler_cfg = config["scheduler"]

    action_resolver = ActionResolver(config["env"])
    env = build_env_adapter(config, action_resolver.noop_action)
    noop_action = env.noop_action
    clock = _build_clock(config)
    policy = build_policy(config, action_resolver)
    decision_scheduler = _build_decision_scheduler(clock)
    latency_source = _build_latency_source(config["latency"])
    persistence_policy = _build_persistence_policy(scheduler_cfg, clock=clock)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop_action,
        persistence_policy=persistence_policy,
        ordering_policy=_build_ordering_policy(scheduler_cfg),
    )
    logger = TrajectoryLogger(config["logging"]["output_dir"])
    return SimulatedLatencyExecutor(
        env=env,
        policy=policy,
        decision_scheduler=decision_scheduler,
        latency_source=latency_source,
        action_scheduler=action_scheduler,
        clock=clock,
        logger=logger,
    )


def _build_batched_simulated_executor(
    config: dict[str, Any],
    *,
    max_parallel_envs: int,
) -> BatchedSimulatedLatencyExecutor:
    scheduler_cfg = config["scheduler"]
    if max_parallel_envs <= 0:
        raise ValueError("max_parallel_envs must be positive")

    action_resolver = ActionResolver(config["env"])
    envs: list[EnvAdapter] = []
    env_backend = None
    try:
        backend_name = _resolve_latency_bench_env_backend(config, eval_parallel_envs=max_parallel_envs)
        if backend_name == "subproc":
            env_backend = SubprocEnvStepBackend(
                config=config,
                noop_action=action_resolver.noop_action,
                num_slots=int(max_parallel_envs),
            )
        elif backend_name == "flappy_gpu_batched":
            from latency_bench.executors.flappy_gpu_batched_backend import FlappyGpuBatchedEnvStepBackend

            env_backend = FlappyGpuBatchedEnvStepBackend(
                config=config,
                noop_action=action_resolver.noop_action,
                num_slots=int(max_parallel_envs),
            )
        else:
            for _slot in range(int(max_parallel_envs)):
                envs.append(build_env_adapter(config, action_resolver.noop_action))
            env_backend = SerialEnvStepBackend(envs)

        _raise_if_unsupported_batched_action_space(env_backend.slot_handles)

        clock = _build_clock(config)
        policy = build_policy(config, action_resolver)
        if type(policy).predict_batch is PolicyRunner.predict_batch:
            raise ValueError(
                f"policy.type={config['policy']['type']} does not support safe batched simulated eval yet; "
                "implement PolicyRunner.predict_batch before using evaluation.eval_parallel_envs > 1"
            )
        decision_scheduler = _build_decision_scheduler(clock)
        action_schedulers = [
            ActionScheduler(
                clock=clock,
                noop_action=env.noop_action,
                persistence_policy=_build_persistence_policy(scheduler_cfg, clock=clock),
                ordering_policy=_build_ordering_policy(scheduler_cfg),
            )
            for env in env_backend.slot_handles
        ]
        logger = TrajectoryLogger(config["logging"]["output_dir"])
        return BatchedSimulatedLatencyExecutor(
            env_backend=env_backend,
            policy=policy,
            decision_scheduler=decision_scheduler,
            latency_sources=[_build_latency_source(config["latency"]) for _env in env_backend.slot_handles],
            action_schedulers=action_schedulers,
            clock=clock,
            logger=logger,
        )
    except Exception:
        if env_backend is not None:
            env_backend.close()
        else:
            for env in envs:
                env.close()
        raise


@register_executor(ExecutorMode.PAUSED)
def _build_paused_executor(config: dict[str, Any]) -> PausedLatencyExecutor:
    scheduler_cfg = config["scheduler"]
    latency_cfg = config["latency"]
    latency_type = str(latency_cfg.get("type", LatencySourceType.ZERO.value)).strip().lower()
    if latency_type != LatencySourceType.ZERO.value:
        raise ValueError("paused executor supports only latency.type=zero")

    action_resolver = ActionResolver(config["env"])
    env = build_env_adapter(config, action_resolver.noop_action)
    noop_action = env.noop_action
    clock = _build_clock(config)
    policy = build_policy(config, action_resolver)
    decision_scheduler = _build_decision_scheduler(clock)
    latency_source = ZeroLatencySource()
    persistence_policy = _build_persistence_policy(scheduler_cfg, clock=clock)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop_action,
        persistence_policy=persistence_policy,
        ordering_policy=_build_ordering_policy(scheduler_cfg),
    )
    logger = TrajectoryLogger(config["logging"]["output_dir"])
    return PausedLatencyExecutor(
        env=env,
        policy=policy,
        decision_scheduler=decision_scheduler,
        latency_source=latency_source,
        action_scheduler=action_scheduler,
        clock=clock,
        logger=logger,
    )


@register_executor(ExecutorMode.REALTIME)
def _build_realtime_executor(config: dict[str, Any]) -> RealtimeLatencyExecutor:
    scheduler_cfg = config["scheduler"]
    executor_cfg = config["executor"]
    latency_cfg = config["latency"]
    latency_type = str(latency_cfg.get("type", LatencySourceType.ZERO.value)).strip().lower()
    if latency_type not in {LatencySourceType.ZERO.value, LatencySourceType.MEASURED.value}:
        raise ValueError("realtime executor does not support injected latency; use latency.type=zero or measured")

    action_resolver = ActionResolver(config["env"])
    env = build_env_adapter(config, action_resolver.noop_action)
    noop_action = env.noop_action
    clock = _build_clock(config)
    decision_scheduler = _build_decision_scheduler(clock)
    latency_source = ZeroLatencySource()
    persistence_policy = _build_persistence_policy(scheduler_cfg, clock=clock)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop_action,
        persistence_policy=persistence_policy,
        ordering_policy=_build_ordering_policy(scheduler_cfg),
    )
    logger = TrajectoryLogger(config["logging"]["output_dir"])
    return RealtimeLatencyExecutor(
        env=env,
        policy=None,
        decision_scheduler=decision_scheduler,
        latency_source=latency_source,
        action_scheduler=action_scheduler,
        clock=clock,
        logger=logger,
        config=config,
        inference_devices=[str(device) for device in executor_cfg["inference_devices"]],
        use_env_process=True,
    )


def _build_clock(config: dict[str, Any]) -> EnvClock:
    env_cfg = config["env"]
    return EnvClock(
        env_fps=float(env_cfg["env_fps"]),
        obs_fps=float(env_cfg["obs_fps"]),
    )


def _build_decision_scheduler(clock: EnvClock) -> DecisionScheduler:
    return ObservationCadenceDecisionScheduler(clock)


def _build_latency_source(latency_cfg: dict[str, Any]) -> LatencySource:
    latency_type = latency_cfg.get("type", LatencySourceType.ZERO.value)
    if latency_type == LatencySourceType.ZERO.value:
        return ZeroLatencySource()
    if latency_type == LatencySourceType.FIXED.value:
        return FixedLatencySource(float(latency_cfg["fixed_latency_ms"]))
    if latency_type == LatencySourceType.NORMAL_DISTRIBUTION.value:
        return NormalDistributionLatencySource(
            mean_latency_ms=float(latency_cfg["mean_latency_ms"]),
            std_latency_ms=float(latency_cfg["std_latency_ms"]),
            seed=latency_cfg.get("seed"),
            min_latency_ms=float(latency_cfg.get("min_latency_ms", 0.0)),
            max_latency_ms=latency_cfg.get("max_latency_ms"),
        )
    if latency_type == LatencySourceType.UNIFORM_DISTRIBUTION.value:
        return UniformDistributionLatencySource(
            min_latency_ms=float(latency_cfg["min_latency_ms"]),
            max_latency_ms=float(latency_cfg["max_latency_ms"]),
            seed=latency_cfg.get("seed"),
        )
    if latency_type == LatencySourceType.TRACE_REPLAY.value:
        return TraceReplayLatencySource(
            trace_path=latency_cfg["trace_path"],
            loop_trace=bool(latency_cfg.get("loop_trace", False)),
        )
    if latency_type == LatencySourceType.EMPIRICAL_SAMPLE.value:
        return EmpiricalSampleLatencySource(
            profile_path=latency_cfg["profile_path"],
            seed=latency_cfg.get("seed"),
        )
    if latency_type == LatencySourceType.EMPIRICAL_TRACE_REPLAY.value:
        return EmpiricalTraceReplayLatencySource(
            profile_path=latency_cfg["profile_path"],
            trace_replay_order=latency_cfg.get("trace_replay_order", "sequential"),
            loop_trace=bool(latency_cfg.get("loop_trace", True)),
            seed=latency_cfg.get("seed"),
        )
    raise ValueError(f"Unsupported latency.type for simulated executor: {latency_type}")


def _build_ordering_policy(scheduler_cfg: dict[str, Any]) -> OrderingPolicy:
    _raise_if_unsupported_scheduler_fields(scheduler_cfg)
    ordering_policy = scheduler_cfg.get("ordering_policy", OrderingPolicy.ISSUE_ORDER_FIFO.value)
    return _normalize_ordering_policy(ordering_policy, field_name="scheduler.ordering_policy")


def _normalize_ordering_policy(value: Any, *, field_name: str) -> OrderingPolicy:
    normalized = str(value).strip().lower()
    try:
        return OrderingPolicy(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be one of ['issue_order_fifo', 'latest_ready']") from exc


def _build_persistence_policy(
    scheduler_cfg: dict[str, Any],
    *,
    clock: EnvClock | None = None,
) -> ActionPersistencePolicy:
    mode = _resolve_hold_policy_mode(scheduler_cfg)
    if mode == ActionHoldPolicy.ONE_FRAME_THEN_NOOP:
        return OneFrameThenNoopPolicy()
    if mode == ActionHoldPolicy.HOLD:
        if clock is None:
            raise ValueError("clock is required for hold policy")
        return HoldPolicy(duration_raw_frames=clock.obs_stride_raw_frames)
    raise ValueError(f"Unsupported hold policy: {mode.value}")


def _resolve_hold_policy_mode(scheduler_cfg: dict[str, Any]) -> ActionHoldPolicy:
    _raise_if_unsupported_scheduler_fields(scheduler_cfg)
    hold_policy = scheduler_cfg.get("hold_policy", ActionHoldPolicy.ONE_FRAME_THEN_NOOP.value)
    return _normalize_hold_policy(hold_policy, field_name="scheduler.hold_policy")


def _raise_if_unsupported_scheduler_fields(scheduler_cfg: dict[str, Any]) -> None:
    if set(scheduler_cfg) - _SCHEDULER_ALLOWED_KEYS:
        raise ValueError(
            "scheduler contains unsupported fields; use only "
            "['hold_policy', 'ordering_policy']"
        )


def _normalize_hold_policy(value: Any, *, field_name: str) -> ActionHoldPolicy:
    normalized = str(value).strip().lower()
    try:
        return ActionHoldPolicy(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of "
            "['one_frame_then_noop', 'hold']"
        ) from exc


def _eval_parallel_envs(config: dict[str, Any]) -> int:
    value = config.get("evaluation", {}).get("eval_parallel_envs", 1)
    eval_parallel_envs = int(value)
    if eval_parallel_envs <= 0:
        raise ValueError("evaluation.eval_parallel_envs must be positive")
    return eval_parallel_envs


def _resolve_latency_bench_env_backend(config: dict[str, Any], *, eval_parallel_envs: int) -> str:
    configured = os.environ.get(_ENV_BACKEND_DEBUG_ENV)
    if configured is None:
        configured = config.get("evaluation", {}).get("latency_bench_env_backend", "auto")
    backend = str(configured).strip().lower()
    if backend not in _LATENCY_BENCH_ENV_BACKENDS:
        raise ValueError(
            "evaluation.latency_bench_env_backend must be one of "
            f"{sorted(_LATENCY_BENCH_ENV_BACKENDS)}; got {configured!r}"
        )
    if int(eval_parallel_envs) <= 1:
        return "serial"
    if backend == "auto":
        return _auto_latency_bench_env_backend(config)
    return backend


def _auto_latency_bench_env_backend(config: dict[str, Any]) -> str:
    env_cfg = config.get("env", {})
    env_name = str(env_cfg.get("name", "")).strip().lower()
    policy_type = str(config.get("policy", {}).get("type", "")).strip().lower()
    if env_name == "flappy" and policy_type == "flappy_sf" and bool(env_cfg.get("use_gpu_render", False)):
        return "flappy_gpu_batched"
    if env_name in _AUTO_SUBPROC_ENV_NAMES:
        return "subproc"
    return "serial"


def _supports_phase_3b_batched_latency(latency_cfg: dict[str, Any]) -> bool:
    latency_type = str(latency_cfg.get("type", LatencySourceType.ZERO.value)).strip().lower()
    return latency_type in {LatencySourceType.ZERO.value, LatencySourceType.FIXED.value}


def _raise_if_unsupported_batched_action_space(envs: list[EnvAdapter]) -> None:
    try:
        import numpy as np
        from gymnasium.spaces import Discrete, MultiDiscrete, Tuple
    except Exception:
        return

    for env in envs:
        action_space = _env_action_space(env)
        if action_space is None or isinstance(action_space, Discrete):
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


def _env_action_space(env: EnvAdapter) -> Any:
    for owner in (env, getattr(env, "gym_env", None), getattr(env, "sf_env", None)):
        if owner is None:
            continue
        action_space = getattr(owner, "action_space", None)
        if action_space is not None:
            return action_space
    return None

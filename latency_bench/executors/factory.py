from __future__ import annotations

from collections.abc import Callable
from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ExecutorMode, LatencySourceType, PersistenceMode, StaleActionMode
from latency_bench.envs.factory import build_env_adapter
from latency_bench.executors.paused import PausedLatencyExecutor
from latency_bench.executors.realtime import RealtimeLatencyExecutor
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.latency.profiler import LatencyProfiler
from latency_bench.latency.sources import (
    FixedLatencySource,
    LatencySource,
    NormalDistributionLatencySource,
    TraceReplayLatencySource,
    ZeroLatencySource,
)
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.observation.builder import ObservationBuilder
from latency_bench.observation.history import HistoryBuffer
from latency_bench.policy.factory import build_policy
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import (
    DecisionScheduler,
    EveryFrameDecisionScheduler,
    FixedFrameIntervalDecisionScheduler,
    FixedFrequencyDecisionScheduler,
)
from latency_bench.scheduler.persistence import ActionPersistencePolicy, HoldPreviousActionPolicy, NoopAfterOneShotPolicy


ExecutorBuilder = Callable[[dict[str, Any]], Any]

_EXECUTOR_BUILDERS: dict[ExecutorMode, ExecutorBuilder] = {}


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
def _build_simulated_executor(config: dict[str, Any]) -> SimulatedLatencyExecutor:
    control_cfg = config["control"]
    scheduler_cfg = config["scheduler"]

    action_resolver = ActionResolver(config["env"])
    noop_action = action_resolver.noop_action
    env = build_env_adapter(config, noop_action)
    clock = EnvClock(fps_env=env.fps_env)
    policy = build_policy(config, action_resolver)
    decision_scheduler = _build_decision_scheduler(control_cfg, env.fps_env)
    latency_source = _build_latency_source(config["latency"])
    persistence_policy = _build_persistence_policy(scheduler_cfg)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop_action,
        persistence_policy=persistence_policy,
        stale_action_mode=StaleActionMode(scheduler_cfg.get("stale_action_mode", StaleActionMode.LATEST_READY.value)),
        expiration_ms=scheduler_cfg.get("expiration_ms"),
    )
    logger = TrajectoryLogger(config["logging"]["output_dir"])
    return SimulatedLatencyExecutor(
        env=env,
        policy=policy,
        observation_builder=ObservationBuilder(),
        history=HistoryBuffer(max_len=int(control_cfg.get("history_len", 16))),
        decision_scheduler=decision_scheduler,
        latency_source=latency_source,
        action_scheduler=action_scheduler,
        clock=clock,
        logger=logger,
    )


@register_executor(ExecutorMode.PAUSED)
def _build_paused_executor(config: dict[str, Any]) -> PausedLatencyExecutor:
    control_cfg = config["control"]
    scheduler_cfg = config["scheduler"]

    action_resolver = ActionResolver(config["env"])
    noop_action = action_resolver.noop_action
    env = build_env_adapter(config, noop_action)
    clock = EnvClock(fps_env=env.fps_env)
    policy = build_policy(config, action_resolver)
    decision_scheduler = _build_decision_scheduler(control_cfg, env.fps_env)
    latency_source = _build_latency_source(config["latency"])
    persistence_policy = _build_persistence_policy(scheduler_cfg)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop_action,
        persistence_policy=persistence_policy,
        stale_action_mode=StaleActionMode(scheduler_cfg.get("stale_action_mode", StaleActionMode.LATEST_READY.value)),
        expiration_ms=scheduler_cfg.get("expiration_ms"),
    )
    logger = TrajectoryLogger(config["logging"]["output_dir"])
    return PausedLatencyExecutor(
        env=env,
        policy=policy,
        observation_builder=ObservationBuilder(),
        history=HistoryBuffer(max_len=int(control_cfg.get("history_len", 16))),
        decision_scheduler=decision_scheduler,
        latency_source=latency_source,
        action_scheduler=action_scheduler,
        clock=clock,
        logger=logger,
    )


@register_executor(ExecutorMode.REALTIME)
def _build_realtime_executor(config: dict[str, Any]) -> RealtimeLatencyExecutor:
    control_cfg = config["control"]
    scheduler_cfg = config["scheduler"]
    executor_cfg = config["executor"]

    action_resolver = ActionResolver(config["env"])
    noop_action = action_resolver.noop_action
    env = build_env_adapter(config, noop_action)
    clock = EnvClock(fps_env=env.fps_env)
    policy = build_policy(config, action_resolver)
    decision_scheduler = _build_decision_scheduler(control_cfg, env.fps_env)
    latency_cfg = config["latency"]
    latency_profiler = None
    if latency_cfg.get("type", LatencySourceType.ZERO.value) == LatencySourceType.MEASURED.value:
        latency_source = ZeroLatencySource()
        latency_profiler = LatencyProfiler(
            device=latency_cfg.get("device"),
            sync_cuda=bool(latency_cfg.get("sync_cuda", True)),
        )
    else:
        latency_source = _build_latency_source(latency_cfg)
    persistence_policy = _build_persistence_policy(scheduler_cfg)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop_action,
        persistence_policy=persistence_policy,
        stale_action_mode=StaleActionMode(scheduler_cfg.get("stale_action_mode", StaleActionMode.LATEST_READY.value)),
        expiration_ms=scheduler_cfg.get("expiration_ms"),
    )
    logger = TrajectoryLogger(config["logging"]["output_dir"])
    return RealtimeLatencyExecutor(
        env=env,
        policy=policy,
        observation_builder=ObservationBuilder(),
        history=HistoryBuffer(max_len=int(control_cfg.get("history_len", 16))),
        decision_scheduler=decision_scheduler,
        latency_source=latency_source,
        action_scheduler=action_scheduler,
        clock=clock,
        logger=logger,
        max_pending_observations=int(executor_cfg.get("max_pending_observations", 16)),
        latency_profiler=latency_profiler,
    )


def _build_decision_scheduler(control_cfg: dict[str, Any], fps_env: float) -> DecisionScheduler:
    interval = int(control_cfg.get("observe_every_n_frames", 1))
    if interval > 1:
        return FixedFrameIntervalDecisionScheduler(interval_frames=interval)
    frequency_hz = float(control_cfg.get("frequency_hz", fps_env))
    if abs(frequency_hz - fps_env) < 1e-9:
        return EveryFrameDecisionScheduler()
    return FixedFrequencyDecisionScheduler(fps_env=fps_env, frequency_hz=frequency_hz)


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
    if latency_type == LatencySourceType.TRACE_REPLAY.value:
        return TraceReplayLatencySource(
            trace_path=latency_cfg["trace_path"],
            loop_trace=bool(latency_cfg.get("loop_trace", False)),
        )
    raise ValueError(f"Unsupported latency.type for simulated executor: {latency_type}")


def _build_persistence_policy(scheduler_cfg: dict[str, Any]) -> ActionPersistencePolicy:
    mode = PersistenceMode(scheduler_cfg.get("persistence_mode", PersistenceMode.NOOP_AFTER_ONESHOT.value))
    if mode == PersistenceMode.HOLD_PREVIOUS:
        return HoldPreviousActionPolicy()
    if mode == PersistenceMode.NOOP_AFTER_ONESHOT:
        return NoopAfterOneShotPolicy()
    raise ValueError(f"Unsupported persistence mode: {mode}")

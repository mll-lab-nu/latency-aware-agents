from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from latency_bench.core.actions import ActionResolver
from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ExecutorMode
from latency_bench.envs.base import EnvAdapter
from latency_bench.envs.factory import build_env_adapter
from latency_bench.executors.backend_selection import (
    build_env_step_backend,
    raise_if_unsupported_batched_action_space,
    resolve_latency_bench_env_backend,
)
from latency_bench.executors.base import Executor
from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor
from latency_bench.executors.paused import PausedLatencyExecutor
from latency_bench.executors.realtime import RealtimeLatencyExecutor
from latency_bench.executors.realtime.pool import ProcessInferencePool
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.latency.hardware_info import visible_cuda_device_count
from latency_bench.latency.samplers import (
    BATCHED_SIMULATED_METHODS,
    LatencySampler,
    ZeroSampler,
    build_episode_sampler_factory,
    build_sampler,
    resolve_concrete_latency_method,
    resolve_latency_method,
)
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.policy.base import PolicyRunner
from latency_bench.policy.factory import build_policy
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import (
    DecisionScheduler,
    ObservationCadenceDecisionScheduler,
)
from latency_bench.scheduler.factory import build_action_scheduler


class ExecutorBuilder(Protocol):
    def __call__(
        self,
        config: dict[str, Any],
        *,
        policy: PolicyRunner | None = None,
    ) -> Any:
        ...

_EXECUTOR_BUILDERS: dict[ExecutorMode, ExecutorBuilder] = {}


def register_executor(mode: ExecutorMode | str) -> Callable[[ExecutorBuilder], ExecutorBuilder]:
    executor_mode = ExecutorMode(mode)

    def decorator(builder: ExecutorBuilder) -> ExecutorBuilder:
        if executor_mode in _EXECUTOR_BUILDERS:
            raise ValueError(f"Executor already registered: {executor_mode.value}")
        _EXECUTOR_BUILDERS[executor_mode] = builder
        return builder

    return decorator


def build_executor(
    config: dict[str, Any],
    *,
    policy: PolicyRunner | None = None,
) -> Executor:
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
    return builder(config, policy=policy)


def _resolve_policy(
    config: dict[str, Any],
    action_resolver: ActionResolver,
    policy: PolicyRunner | None,
) -> PolicyRunner:
    """Use an injected (e.g. live in-process) policy, else build from config."""
    return policy if policy is not None else build_policy(config, action_resolver)


def get_registered_executor_modes() -> list[str]:
    return sorted(mode.value for mode in _EXECUTOR_BUILDERS)


def _build_trajectory_logger(config: dict[str, Any]) -> TrajectoryLogger:
    logging_cfg = config["logging"]
    return TrajectoryLogger(
        logging_cfg["output_dir"],
        save_step_records=logging_cfg["save_step_records"],
        save_action_records=logging_cfg["save_action_records"],
        save_latency_records=logging_cfg["save_latency_records"],
    )


@dataclass(frozen=True)
class _CommonExecutorComponents:
    action_resolver: ActionResolver
    env: EnvAdapter
    clock: EnvClock
    decision_scheduler: DecisionScheduler
    action_scheduler: ActionScheduler
    logger: TrajectoryLogger | None


def _build_common_executor_components(config: dict[str, Any]) -> _CommonExecutorComponents:
    """Build the executor dependencies that are identical across the single-episode modes."""
    action_resolver = ActionResolver(config["env"])
    env = build_env_adapter(config, action_resolver.noop_action)
    clock = _build_clock(config)
    decision_scheduler = _build_decision_scheduler(clock)
    action_scheduler = build_action_scheduler(config["scheduler"], clock=clock, noop_action=env.noop_action)
    logger = _build_trajectory_logger(config)
    return _CommonExecutorComponents(
        action_resolver=action_resolver,
        env=env,
        clock=clock,
        decision_scheduler=decision_scheduler,
        action_scheduler=action_scheduler,
        logger=logger,
    )


@register_executor(ExecutorMode.SIMULATED)
def _build_simulated_executor(
    config: dict[str, Any],
    *,
    policy: PolicyRunner | None = None,
) -> SimulatedLatencyExecutor | BatchedSimulatedLatencyExecutor:
    eval_parallel_envs = _eval_parallel_envs(config)
    num_episodes = int(config["evaluation"]["eval_episodes"])
    if (
        eval_parallel_envs > 1
        and num_episodes > 0
        and resolve_latency_method(config.get("latency", {})) in BATCHED_SIMULATED_METHODS
    ):
        return _build_batched_simulated_executor(
            config, max_parallel_envs=min(eval_parallel_envs, num_episodes), policy=policy
        )

    return _build_single_simulated_executor(config, policy=policy)


def _build_single_simulated_executor(
    config: dict[str, Any],
    *,
    policy: PolicyRunner | None = None,
) -> SimulatedLatencyExecutor:
    c = _build_common_executor_components(config)
    inference_pool = None
    if _simulated_inference_pool_enabled(config):
        if policy is not None:
            raise ValueError(
                "executor.simulated_inference_pool builds its policy inside subprocess "
                "workers; an injected in-process policy cannot cross the process boundary"
            )
        inference_pool = ProcessInferencePool(
            config=config,
            inference_devices=_resolve_realtime_inference_devices(),
        )
    else:
        policy = _resolve_policy(config, c.action_resolver, policy)
    latency_source = _build_latency_source(config["latency"], c.clock)
    episode_latency_source_factory = build_episode_sampler_factory(config["latency"], c.clock)
    return SimulatedLatencyExecutor(
        env=c.env,
        policy=policy,
        decision_scheduler=c.decision_scheduler,
        latency_source=latency_source,
        action_scheduler=c.action_scheduler,
        clock=c.clock,
        logger=c.logger,
        episode_latency_source_factory=episode_latency_source_factory,
        simulated_worker_capacity=_simulated_worker_capacity(config),
        inference_pool=inference_pool,
    )


def _simulated_inference_pool_enabled(config: dict[str, Any]) -> bool:
    # The pool exists to move CUDA forward passes off-process; inline is
    # draw-for-draw identical, so CPU-only hosts and non-CUDA policies
    # (dry runs, dummy/random test policies) skip the worker spawn entirely.
    if not config["executor"].get("simulated_inference_pool"):
        return False
    if not str(config.get("policy", {}).get("device", "")).startswith("cuda"):
        return False
    return visible_cuda_device_count() > 0


def _build_batched_simulated_executor(
    config: dict[str, Any],
    *,
    max_parallel_envs: int,
    policy: PolicyRunner | None = None,
) -> BatchedSimulatedLatencyExecutor:
    scheduler_cfg = config["scheduler"]
    if max_parallel_envs <= 0:
        raise ValueError("max_parallel_envs must be positive")

    action_resolver = ActionResolver(config["env"])
    backend_name = resolve_latency_bench_env_backend(config, eval_parallel_envs=max_parallel_envs)
    env_backend = build_env_step_backend(
        backend_name,
        config=config,
        noop_action=action_resolver.noop_action,
        num_slots=int(max_parallel_envs),
    )
    try:
        raise_if_unsupported_batched_action_space(env_backend.slot_handles)

        clock = _build_clock(config)
        policy = _resolve_policy(config, action_resolver, policy)
        if type(policy).predict_batch is PolicyRunner.predict_batch:
            raise ValueError(
                f"policy.type={config['policy']['type']} does not support safe batched simulated eval yet; "
                "implement PolicyRunner.predict_batch before using evaluation.eval_parallel_envs > 1"
            )
        decision_scheduler = _build_decision_scheduler(clock)
        action_schedulers = [
            build_action_scheduler(scheduler_cfg, clock=clock, noop_action=env.noop_action)
            for env in env_backend.slot_handles
        ]
        logger = _build_trajectory_logger(config)
        episode_latency_source_factory = build_episode_sampler_factory(config["latency"], clock)
        if resolve_concrete_latency_method(config["latency"]) == "temporal":
            # The hidden regime is session-scoped machine state and a dwell is
            # a draw budget that spans episodes however draws interleave, so
            # one machine timeline serves every slot.
            shared_latency_source = _build_latency_source(config["latency"], clock)
            latency_sources: list[LatencySampler] = [
                shared_latency_source for _env in env_backend.slot_handles
            ]
        else:
            latency_sources = [_build_latency_source(config["latency"], clock) for _env in env_backend.slot_handles]
        return BatchedSimulatedLatencyExecutor(
            env_backend=env_backend,
            policy=policy,
            decision_scheduler=decision_scheduler,
            latency_sources=latency_sources,
            action_schedulers=action_schedulers,
            clock=clock,
            logger=logger,
            episode_latency_source_factory=episode_latency_source_factory,
            simulated_worker_capacity=_simulated_worker_capacity(config),
            profile_pipeline=bool(config.get("logging", {}).get("simulated_pipeline_profile", False)),
        )
    except Exception:
        env_backend.close()
        raise


@register_executor(ExecutorMode.PAUSED)
def _build_paused_executor(
    config: dict[str, Any],
    *,
    policy: PolicyRunner | None = None,
) -> PausedLatencyExecutor:
    latency_cfg = config["latency"]
    if resolve_latency_method(latency_cfg) != "zero":
        raise ValueError("paused executor supports only latency.type=zero")

    c = _build_common_executor_components(config)
    return PausedLatencyExecutor(
        env=c.env,
        policy=_resolve_policy(config, c.action_resolver, policy),
        decision_scheduler=c.decision_scheduler,
        latency_source=ZeroSampler(clock=c.clock),
        action_scheduler=c.action_scheduler,
        clock=c.clock,
        logger=c.logger,
    )


@register_executor(ExecutorMode.REALTIME)
def _build_realtime_executor(
    config: dict[str, Any],
    *,
    policy: PolicyRunner | None = None,
) -> RealtimeLatencyExecutor:
    if policy is not None:
        raise ValueError(
            "realtime executor builds its policy inside subprocess workers; "
            "an injected in-process policy cannot cross the process boundary"
        )
    latency_cfg = config["latency"]
    if resolve_latency_method(latency_cfg) != "zero":
        raise ValueError("realtime executor does not support injected latency")
    inference_devices = _resolve_realtime_inference_devices()

    c = _build_common_executor_components(config)
    video_output_dir = _video_output_dir(config) if _video_enabled(config) else None
    return RealtimeLatencyExecutor(
        env=c.env,
        policy=None,
        decision_scheduler=c.decision_scheduler,
        latency_source=ZeroSampler(clock=c.clock),
        action_scheduler=c.action_scheduler,
        clock=c.clock,
        logger=c.logger,
        config=config,
        inference_devices=inference_devices,
        use_env_process=True,
        video_output_dir=video_output_dir,
        video_fps=c.clock.env_fps,
    )


def _resolve_realtime_inference_devices() -> list[str]:
    device_count = visible_cuda_device_count()
    if device_count == 0:
        raise RuntimeError("realtime inference requires at least one visible CUDA device")
    return [f"cuda:{idx}" for idx in range(device_count)]


def _build_clock(config: dict[str, Any]) -> EnvClock:
    env_cfg = config["env"]
    return EnvClock(
        env_fps=float(env_cfg["env_fps"]),
        obs_fps=float(env_cfg["obs_fps"]),
    )


def _video_enabled(config: dict[str, Any]) -> bool:
    return "video" in config["logging"] and config["logging"]["video"]["enabled"]


def _video_output_dir(config: dict[str, Any]) -> str:
    return str(os.path.join(str(config["logging"]["output_dir"]), "videos"))


def _build_decision_scheduler(clock: EnvClock) -> DecisionScheduler:
    return ObservationCadenceDecisionScheduler(clock)


def _build_latency_source(latency_cfg: dict[str, Any], clock: EnvClock) -> LatencySampler:
    return build_sampler(latency_cfg, clock=clock, seed=latency_cfg.get("seed"))


def _simulated_worker_capacity(config: dict[str, Any]) -> int | None:
    return config["executor"].get("simulated_worker_capacity")


def _eval_parallel_envs(config: dict[str, Any]) -> int:
    value = config.get("evaluation", {}).get("eval_parallel_envs", 1)
    eval_parallel_envs = int(value)
    if eval_parallel_envs <= 0:
        raise ValueError("evaluation.eval_parallel_envs must be positive")
    return eval_parallel_envs



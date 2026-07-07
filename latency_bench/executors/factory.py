from __future__ import annotations

import os
import numpy as np
import torch
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from gymnasium.spaces import Discrete, MultiDiscrete, Tuple

from latency_bench.core.actions import ActionResolver
from latency_bench.core.clock import EnvClock
from latency_bench.core.types import ExecutorMode
from latency_bench.envs.base import EnvAdapter
from latency_bench.envs.factory import build_env_adapter
from latency_bench.executors.base import Executor
from latency_bench.executors.batched_simulated import BatchedSimulatedLatencyExecutor
from latency_bench.executors.env_step_backend import (
    SerialEnvStepBackend,
    SubprocEnvStepBackend,
    _env_action_space,
)
from latency_bench.executors.paused import PausedLatencyExecutor
from latency_bench.executors.realtime import RealtimeLatencyExecutor
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.latency.samplers import (
    BATCHED_SIMULATED_METHODS,
    LatencySampler,
    ZeroSampler,
    build_episode_sampler_factory,
    build_sampler,
    resolve_latency_method,
)
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.logging.video import EpisodeVideoRecorder
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
        video_recorder: EpisodeVideoRecorder | None = None,
        video_episode_metadata: dict[int, dict] | None = None,
        policy: PolicyRunner | None = None,
    ) -> Any:
        ...

_EXECUTOR_BUILDERS: dict[ExecutorMode, ExecutorBuilder] = {}
_LATENCY_BENCH_ENV_BACKENDS = {
    "serial",
    "subproc",
    "flappy_gpu_batched",
    "demon_attack_gpu_batched",
    "auto",
}
_ENV_BACKEND_DEBUG_ENV = "LATENCY_BENCH_ENV_BACKEND"
_AUTO_SUBPROC_ENV_NAMES = {"deadly_corridor"}
_AUTO_FLAPPY_GPU_BATCHED_POLICY_TYPES = {"flappy_sf", "starvla"}


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
    video_recorder: EpisodeVideoRecorder | None = None,
    video_episode_metadata: dict[int, dict] | None = None,
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
    return builder(
        config,
        video_recorder=video_recorder,
        video_episode_metadata=video_episode_metadata,
        policy=policy,
    )


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
    video_recorder: EpisodeVideoRecorder | None = None,
    video_episode_metadata: dict[int, dict] | None = None,
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

    return _build_single_simulated_executor(
        config,
        video_recorder=video_recorder,
        video_episode_metadata=video_episode_metadata,
        policy=policy,
    )


def _build_single_simulated_executor(
    config: dict[str, Any],
    *,
    video_recorder: EpisodeVideoRecorder | None = None,
    video_episode_metadata: dict[int, dict] | None = None,
    policy: PolicyRunner | None = None,
) -> SimulatedLatencyExecutor:
    c = _build_common_executor_components(config)
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
        video_recorder=video_recorder,
        video_episode_metadata=video_episode_metadata,
        episode_latency_source_factory=episode_latency_source_factory,
        simulated_worker_capacity=_simulated_worker_capacity(config),
    )


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
        elif backend_name == "demon_attack_gpu_batched":
            from latency_bench.executors.demon_attack_gpu_batched_backend import DemonAttackGpuBatchedEnvStepBackend

            env_backend = DemonAttackGpuBatchedEnvStepBackend(
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
        return BatchedSimulatedLatencyExecutor(
            env_backend=env_backend,
            policy=policy,
            decision_scheduler=decision_scheduler,
            latency_sources=[_build_latency_source(config["latency"], clock) for _env in env_backend.slot_handles],
            action_schedulers=action_schedulers,
            clock=clock,
            logger=logger,
            episode_latency_source_factory=episode_latency_source_factory,
            simulated_worker_capacity=_simulated_worker_capacity(config),
            profile_pipeline=bool(config.get("logging", {}).get("simulated_pipeline_profile", False)),
        )
    except Exception:
        if env_backend is not None:
            env_backend.close()
        else:
            for env in envs:
                env.close()
        raise


@register_executor(ExecutorMode.PAUSED)
def _build_paused_executor(
    config: dict[str, Any],
    *,
    video_recorder: EpisodeVideoRecorder | None = None,
    video_episode_metadata: dict[int, dict] | None = None,
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
        video_recorder=video_recorder,
        video_episode_metadata=video_episode_metadata,
    )


@register_executor(ExecutorMode.REALTIME)
def _build_realtime_executor(
    config: dict[str, Any],
    *,
    video_recorder: EpisodeVideoRecorder | None = None,
    video_episode_metadata: dict[int, dict] | None = None,
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
        video_recorder=video_recorder,
        video_output_dir=video_output_dir,
        video_fps=c.clock.env_fps,
    )


def _resolve_realtime_inference_devices() -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        device_count = torch.cuda.device_count()
    else:
        stripped = visible_devices.strip()
        device_count = 0 if stripped in {"", "-1"} else len([item for item in stripped.split(",") if item.strip()])
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


def _raise_if_unsupported_batched_action_space(envs: list[EnvAdapter]) -> None:
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

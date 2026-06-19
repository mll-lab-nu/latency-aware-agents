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
from latency_bench.core.latency_distribution import derive_latency_seed
from latency_bench.core.types import ActionHoldPolicy, ExecutorMode, LatencySourceType, OrderingPolicy
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
from latency_bench.latency.sources import (
    FixedLatencySource,
    LatencySource,
    NormalDistributionLatencySource,
    MixedNormalDistributionLatencySource,
    MixedUniformDistributionLatencySource,
    ProfileSampleLatencySource,
    UniformDistributionLatencySource,
    ZeroLatencySource,
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
from latency_bench.scheduler.persistence import (
    ActionPersistencePolicy,
    HoldPolicy,
    OneFrameThenNoopPolicy,
)


class ExecutorBuilder(Protocol):
    def __call__(
        self,
        config: dict[str, Any],
        *,
        video_recorder: EpisodeVideoRecorder | None = None,
        video_episode_metadata: dict[int, dict] | None = None,
    ) -> Any:
        ...

_EXECUTOR_BUILDERS: dict[ExecutorMode, ExecutorBuilder] = {}
_LATENCY_BENCH_ENV_BACKENDS = {
    "serial",
    "subproc",
    "flappy_gpu_batched",
    "auto",
}
_ENV_BACKEND_DEBUG_ENV = "LATENCY_BENCH_ENV_BACKEND"
_AUTO_SUBPROC_ENV_NAMES = {"deadly_corridor"}
_AUTO_FLAPPY_GPU_BATCHED_POLICY_TYPES = {"flappy_sf", "starvla"}
_SCHEDULER_ALLOWED_KEYS = frozenset({"ordering_policy", "hold_policy"})


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
    return builder(config, video_recorder=video_recorder, video_episode_metadata=video_episode_metadata)


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
    scheduler_cfg = config["scheduler"]
    action_resolver = ActionResolver(config["env"])
    env = build_env_adapter(config, action_resolver.noop_action)
    noop_action = env.noop_action
    clock = _build_clock(config)
    decision_scheduler = _build_decision_scheduler(clock)
    persistence_policy = _build_persistence_policy(scheduler_cfg, clock=clock)
    action_scheduler = ActionScheduler(
        clock=clock,
        noop_action=noop_action,
        persistence_policy=persistence_policy,
        ordering_policy=_build_ordering_policy(scheduler_cfg),
    )
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
) -> SimulatedLatencyExecutor | BatchedSimulatedLatencyExecutor:
    eval_parallel_envs = _eval_parallel_envs(config)
    num_episodes = int(config["evaluation"]["eval_episodes"])
    if (
        eval_parallel_envs > 1
        and num_episodes > 0
        and _supports_phase_3b_batched_latency(config.get("latency", {}))
    ):
        return _build_batched_simulated_executor(config, max_parallel_envs=min(eval_parallel_envs, num_episodes))

    return _build_single_simulated_executor(
        config,
        video_recorder=video_recorder,
        video_episode_metadata=video_episode_metadata,
    )


def _build_single_simulated_executor(
    config: dict[str, Any],
    *,
    video_recorder: EpisodeVideoRecorder | None = None,
    video_episode_metadata: dict[int, dict] | None = None,
) -> SimulatedLatencyExecutor:
    c = _build_common_executor_components(config)
    policy = build_policy(config, c.action_resolver)
    latency_source = _build_latency_source(config["latency"])
    episode_latency_source_factory = _build_episode_latency_source_factory(config["latency"])
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
        logger = _build_trajectory_logger(config)
        episode_latency_source_factory = _build_episode_latency_source_factory(config["latency"])
        return BatchedSimulatedLatencyExecutor(
            env_backend=env_backend,
            policy=policy,
            decision_scheduler=decision_scheduler,
            latency_sources=[_build_latency_source(config["latency"]) for _env in env_backend.slot_handles],
            action_schedulers=action_schedulers,
            clock=clock,
            logger=logger,
            episode_latency_source_factory=episode_latency_source_factory,
            simulated_worker_capacity=_simulated_worker_capacity(config),
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
) -> PausedLatencyExecutor:
    latency_cfg = config["latency"]
    latency_type = str(latency_cfg.get("type", LatencySourceType.ZERO.value)).strip().lower()
    if latency_type != LatencySourceType.ZERO.value:
        raise ValueError("paused executor supports only latency.type=zero")

    c = _build_common_executor_components(config)
    return PausedLatencyExecutor(
        env=c.env,
        policy=build_policy(config, c.action_resolver),
        decision_scheduler=c.decision_scheduler,
        latency_source=ZeroLatencySource(),
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
) -> RealtimeLatencyExecutor:
    latency_cfg = config["latency"]
    latency_type = str(latency_cfg.get("type", LatencySourceType.ZERO.value)).strip().lower()
    if latency_type != LatencySourceType.ZERO.value:
        raise ValueError("realtime executor does not support injected latency")
    inference_devices = _resolve_realtime_inference_devices()

    c = _build_common_executor_components(config)
    video_output_dir = _video_output_dir(config) if _video_enabled(config) else None
    return RealtimeLatencyExecutor(
        env=c.env,
        policy=None,
        decision_scheduler=c.decision_scheduler,
        latency_source=ZeroLatencySource(),
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
    if latency_type == LatencySourceType.MIXED_NORMAL_DISTRIBUTION.value:
        return MixedNormalDistributionLatencySource(
            mean_latency_ms=float(latency_cfg["mean_latency_ms"]),
            std_latency_ms=float(latency_cfg["std_latency_ms"]),
            seed=latency_cfg.get("seed"),
            min_latency_ms=float(latency_cfg["min_latency_ms"]),
            max_latency_ms=float(latency_cfg["max_latency_ms"]),
        )
    if latency_type == LatencySourceType.MIXED_UNIFORM_DISTRIBUTION.value:
        return MixedUniformDistributionLatencySource(
            min_latency_ms=float(latency_cfg["min_latency_ms"]),
            max_latency_ms=float(latency_cfg["max_latency_ms"]),
            seed=latency_cfg.get("seed"),
        )
    if latency_type == LatencySourceType.PROFILE_SAMPLE.value:
        return ProfileSampleLatencySource(
            profile_path=latency_cfg["profile_path"],
            seed=latency_cfg.get("seed"),
            sampler=latency_cfg["sampler"] if "sampler" in latency_cfg else "profile_default",
        )
    raise ValueError(f"Unsupported latency.type for simulated executor: {latency_type}")


_SEEDED_STOCHASTIC_LATENCY_TYPES = {
    LatencySourceType.NORMAL_DISTRIBUTION.value,
    LatencySourceType.UNIFORM_DISTRIBUTION.value,
    LatencySourceType.MIXED_NORMAL_DISTRIBUTION.value,
    LatencySourceType.MIXED_UNIFORM_DISTRIBUTION.value,
    LatencySourceType.PROFILE_SAMPLE.value,
}


def _build_episode_latency_source_factory(
    latency_cfg: dict[str, Any],
) -> Callable[[int], LatencySource] | None:
    latency_type = latency_cfg.get("type", LatencySourceType.ZERO.value)
    if latency_type not in _SEEDED_STOCHASTIC_LATENCY_TYPES:
        return None
    if "seed" not in latency_cfg:
        return None

    def build_for_episode(episode_id: int) -> LatencySource:
        episode_latency_cfg = dict(latency_cfg)
        episode_latency_cfg["seed"] = derive_latency_seed(latency_cfg["seed"], episode_idx=episode_id)
        return _build_latency_source(episode_latency_cfg)

    return build_for_episode


def _build_ordering_policy(scheduler_cfg: dict[str, Any]) -> OrderingPolicy:
    _raise_if_unsupported_scheduler_fields(scheduler_cfg)
    ordering_policy = scheduler_cfg.get("ordering_policy", OrderingPolicy.ISSUE_ORDER_FIFO.value)
    return _normalize_ordering_policy(ordering_policy, field_name="scheduler.ordering_policy")


def _simulated_worker_capacity(config: dict[str, Any]) -> int | None:
    return config["executor"].get("simulated_worker_capacity")


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
    if (
        env_name == "flappy"
        and policy_type in _AUTO_FLAPPY_GPU_BATCHED_POLICY_TYPES
        and bool(env_cfg.get("use_gpu_render", False))
    ):
        return "flappy_gpu_batched"
    if env_name in _AUTO_SUBPROC_ENV_NAMES:
        return "subproc"
    return "serial"


def _supports_phase_3b_batched_latency(latency_cfg: dict[str, Any]) -> bool:
    latency_type = str(latency_cfg.get("type", LatencySourceType.ZERO.value)).strip().lower()
    return latency_type in {
        LatencySourceType.ZERO.value,
        LatencySourceType.FIXED.value,
        LatencySourceType.NORMAL_DISTRIBUTION.value,
        LatencySourceType.UNIFORM_DISTRIBUTION.value,
        LatencySourceType.MIXED_NORMAL_DISTRIBUTION.value,
        LatencySourceType.MIXED_UNIFORM_DISTRIBUTION.value,
        LatencySourceType.PROFILE_SAMPLE.value,
    }


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

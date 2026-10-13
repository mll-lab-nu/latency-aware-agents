from __future__ import annotations

from latency_bench.core.config import resolve_runtime_config

import os
from dataclasses import dataclass
from typing import Any

from latency_bench.core.actions import ActionResolver
from latency_bench.core.clock import EnvClock
from latency_bench.core.latency_distribution import derive_seed
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
from latency_bench.executors.env_step_backend import EnvStepBackend
from latency_bench.executors.paused import PausedLatencyExecutor
from latency_bench.executors.realtime.executor import RealtimeLatencyExecutor, RealtimeRuntime
from latency_bench.executors.realtime.env_process import RealtimeEnvProcess
from latency_bench.executors.realtime.pool import ProcessInferencePool
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.latency.config import BATCHED_SIMULATED_METHODS
from latency_bench.latency.samplers import (
    build_episode_sampler_factory,
    build_sampler,
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
from latency_bench.policy.action_prefix import prepare_action_prefix


def build_executor(
    config: dict[str, Any],
    *,
    policy: PolicyRunner | None = None,
    env: EnvAdapter | None = None,
    env_backend: EnvStepBackend | None = None,
    inference_devices: list[str] | None = None,
) -> Executor:
    """Build an executor; injected environments use config["env"]["env_fps"]."""
    resolve_runtime_config(config)
    if config["policy"]["type"] == "coding_agent":
        from latency_bench.policy.coding_agent_contract import prepare_coding_agent_config

        prepare_coding_agent_config(config)
    prepare_action_prefix(config)
    executor_cfg = config["executor"]
    if inference_devices is None and "inference_devices" in executor_cfg:
        inference_devices = executor_cfg["inference_devices"]
    mode = ExecutorMode(executor_cfg["mode"])
    parallel_envs = executor_cfg["parallel_envs"]
    if parallel_envs > 1 and mode in {ExecutorMode.PAUSED, ExecutorMode.REALTIME}:
        raise ValueError(
            "executor.parallel_envs > 1 is only supported for executor.mode=simulated; "
            f"got executor.mode={mode.value}"
        )
    if mode == ExecutorMode.SIMULATED:
        return _build_simulated_executor(
            config,
            policy=policy,
            env=env,
            env_backend=env_backend,
            inference_devices=inference_devices,
        )
    if mode == ExecutorMode.PAUSED:
        return _build_paused_executor(config, policy=policy, env=env)
    return _build_realtime_executor(
        config,
        policy=policy,
        env=env,
        env_backend=env_backend,
        inference_devices=inference_devices,
    )


def _resolve_policy(
    config: dict[str, Any],
    action_resolver: ActionResolver,
    policy: PolicyRunner | None,
) -> PolicyRunner:
    """Use an injected (e.g. live in-process) policy, else build from config."""
    return policy if policy is not None else build_policy(config, action_resolver)


def _build_simulated_policy_and_pool(
    config: dict[str, Any],
    action_resolver: ActionResolver,
    policy: PolicyRunner | None,
    inference_devices: list[str] | None,
) -> tuple[PolicyRunner | None, ProcessInferencePool | None]:
    inference_pool = None
    if config["executor"]["inference_pool"]:
        if policy is not None:
            raise ValueError(
                "executor.inference_pool builds its policy inside subprocess "
                "workers; an injected in-process policy cannot cross the process boundary"
            )
        inference_pool = ProcessInferencePool(
            config=config,
            inference_devices=(
                config["executor"]["inference_devices"]
                if inference_devices is None else inference_devices
            ),
        )
    else:
        policy = _resolve_policy(config, action_resolver, policy)
    return policy, inference_pool


def _build_trajectory_logger(config: dict[str, Any]) -> TrajectoryLogger:
    logging_cfg = config["logging"]
    return TrajectoryLogger(
        logging_cfg["output_dir"],
        save_step_records=logging_cfg["save_step_records"],
        save_action_records=logging_cfg["save_action_records"],
        save_latency_records=logging_cfg["save_latency_records"],
    )


@dataclass(frozen=True)
class _ExecutorRuntimeComponents:
    action_resolver: ActionResolver
    env: EnvAdapter
    clock: EnvClock
    decision_scheduler: DecisionScheduler
    action_scheduler: ActionScheduler


def _build_executor_runtime_components(
    config: dict[str, Any],
    *,
    env: EnvAdapter | None = None,
) -> _ExecutorRuntimeComponents:
    action_resolver = ActionResolver(config["env"])
    if env is None:
        env = build_env_adapter(config, action_resolver.noop_action)
    clock = _build_clock(config)
    decision_scheduler = ObservationCadenceDecisionScheduler(clock)
    action_scheduler = build_action_scheduler(config["scheduler"], clock=clock, noop_action=env.noop_action)
    return _ExecutorRuntimeComponents(
        action_resolver=action_resolver,
        env=env,
        clock=clock,
        decision_scheduler=decision_scheduler,
        action_scheduler=action_scheduler,
    )


def _build_simulated_executor(
    config: dict[str, Any],
    *,
    policy: PolicyRunner | None = None,
    env: EnvAdapter | None = None,
    env_backend: EnvStepBackend | None = None,
    inference_devices: list[str] | None = None,
) -> SimulatedLatencyExecutor | BatchedSimulatedLatencyExecutor:
    parallel_envs = config["executor"]["parallel_envs"]
    num_episodes = int(config["evaluation"]["episodes"])
    if (
        (
            env_backend is not None
            or parallel_envs > 1
            or config["env"].get("action_history_decisions") is not None
        )
        and num_episodes > 0
        and config["latency"]["method"] in BATCHED_SIMULATED_METHODS
    ):
        return _build_batched_simulated_executor(
            config,
            max_parallel_envs=min(parallel_envs, num_episodes),
            policy=policy,
            env_backend=env_backend,
            inference_devices=inference_devices,
        )

    return _build_single_simulated_executor(
        config,
        policy=policy,
        env=env,
        inference_devices=inference_devices,
    )


def _build_single_simulated_executor(
    config: dict[str, Any],
    *,
    policy: PolicyRunner | None = None,
    env: EnvAdapter | None = None,
    inference_devices: list[str] | None = None,
) -> SimulatedLatencyExecutor:
    c = _build_executor_runtime_components(config, env=env)
    inference_pool = None
    try:
        if config["policy"]["type"] == "coding_agent":
            from latency_bench.policy.coding_agent_contract import bind_coding_agent_action_space

            bind_coding_agent_action_space(config, c.env)
        policy, inference_pool = _build_simulated_policy_and_pool(
            config, c.action_resolver, policy, inference_devices
        )
        episode_latency_source_factory = build_episode_sampler_factory(config["latency"], c.clock)
        latency_source = (
            episode_latency_source_factory(0)
            if episode_latency_source_factory is not None
            else build_sampler(config["latency"], clock=c.clock)
        )
        return SimulatedLatencyExecutor(
            env=c.env,
            policy=policy,
            decision_scheduler=c.decision_scheduler,
            latency_source=latency_source,
            action_scheduler=c.action_scheduler,
            clock=c.clock,
            logger=_build_trajectory_logger(config),
            episode_latency_source_factory=episode_latency_source_factory,
            simulated_worker_capacity=config["executor"]["simulated_worker_capacity"],
            inference_pool=inference_pool,
            action_prefix=config["policy"]["action_prefix"],
        )
    except BaseException:
        if inference_pool is not None:
            inference_pool.close()
        c.env.close()
        raise


def _build_batched_simulated_executor(
    config: dict[str, Any],
    *,
    max_parallel_envs: int,
    policy: PolicyRunner | None = None,
    env_backend: EnvStepBackend | None = None,
    inference_devices: list[str] | None = None,
) -> BatchedSimulatedLatencyExecutor:
    scheduler_cfg = config["scheduler"]
    action_resolver = ActionResolver(config["env"])
    if env_backend is None:
        backend_name = resolve_latency_bench_env_backend(config, parallel_envs=max_parallel_envs)
        env_backend = build_env_step_backend(
            backend_name,
            config=config,
            noop_action=action_resolver.noop_action,
            num_slots=max_parallel_envs,
        )
    inference_pool = None
    try:
        raise_if_unsupported_batched_action_space(env_backend.slot_handles)
        if config["policy"]["type"] == "coding_agent":
            from latency_bench.policy.coding_agent_contract import bind_coding_agent_action_space

            bind_coding_agent_action_space(config, env_backend.slot_handles[0])

        clock = _build_clock(config)
        policy, inference_pool = _build_simulated_policy_and_pool(
            config, action_resolver, policy, inference_devices
        )
        if inference_pool is None:
            if type(policy).predict_batch is PolicyRunner.predict_batch:
                raise ValueError(
                    f"policy.type={config['policy']['type']} does not support safe batched simulated eval yet; "
                    "implement PolicyRunner.predict_batch before using executor.parallel_envs > 1"
                )
        decision_scheduler = ObservationCadenceDecisionScheduler(clock)
        action_schedulers = [
            build_action_scheduler(scheduler_cfg, clock=clock, noop_action=env.noop_action)
            for env in env_backend.slot_handles
        ]
        logger = _build_trajectory_logger(config)
        episode_latency_source_factory = build_episode_sampler_factory(config["latency"], clock)
        if episode_latency_source_factory is not None:
            # The real source is selected when each episode starts.  Build
            # initial slot sources through the factory as well so IID's shared
            # profile distribution is not loaded once per slot.
            latency_sources = [
                episode_latency_source_factory(slot_id)
                for slot_id, _env in enumerate(env_backend.slot_handles)
            ]
        else:
            latency_sources = []
            for slot_id, _env in enumerate(env_backend.slot_handles):
                replica_latency = config["latency"].copy()
                if "seed" in replica_latency:
                    replica_latency["seed"] = derive_seed(replica_latency["seed"], vector_index=slot_id)
                latency_sources.append(build_sampler(replica_latency, clock=clock))
        return BatchedSimulatedLatencyExecutor(
            env_backend=env_backend,
            policy=policy,
            decision_scheduler=decision_scheduler,
            latency_sources=latency_sources,
            action_schedulers=action_schedulers,
            clock=clock,
            logger=logger,
            episode_latency_source_factory=episode_latency_source_factory,
            simulated_worker_capacity=config["executor"]["simulated_worker_capacity"],
            inference_pool=inference_pool,
            action_prefix=config["policy"]["action_prefix"],
            action_history_decisions=config["env"].get("action_history_decisions"),
        )
    except Exception:
        if inference_pool is not None:
            inference_pool.close()
        env_backend.close()
        raise


def _build_paused_executor(
    config: dict[str, Any],
    *,
    policy: PolicyRunner | None = None,
    env: EnvAdapter | None = None,
) -> PausedLatencyExecutor:
    c = _build_executor_runtime_components(config, env=env)
    try:
        return PausedLatencyExecutor(
            env=c.env,
            policy=_resolve_policy(config, c.action_resolver, policy),
            decision_scheduler=c.decision_scheduler,
            action_scheduler=c.action_scheduler,
            clock=c.clock,
            logger=_build_trajectory_logger(config),
            action_prefix=config["policy"]["action_prefix"],
        )
    except BaseException:
        c.env.close()
        raise


def _build_realtime_executor(
    config: dict[str, Any],
    *,
    policy: PolicyRunner | None = None,
    env: EnvAdapter | None = None,
    env_backend: EnvStepBackend | None = None,
    inference_devices: list[str] | None = None,
) -> RealtimeLatencyExecutor:
    if policy is not None or env is not None or env_backend is not None:
        raise ValueError(
            "realtime constructs environments and policies inside subprocess workers; "
            "injected live objects cannot cross the process boundary"
        )
    if inference_devices is None:
        inference_devices = config["executor"]["inference_devices"]

    video_output_dir = (
        os.path.join(str(config["logging"]["output_dir"]), "videos")
        if config["logging"]["video"]["enabled"] else None
    )
    clock = _build_clock(config)
    environment = RealtimeEnvProcess(
        config, video_output_dir=video_output_dir, video_fps=clock.env_fps,
    )
    try:
        runtime = RealtimeRuntime(
            env=environment,
            decision_scheduler=ObservationCadenceDecisionScheduler(clock),
            action_scheduler=build_action_scheduler(
                config["scheduler"], clock=clock, noop_action=environment.noop_action,
            ),
            clock=clock,
        )
        return RealtimeLatencyExecutor(
            runtime=runtime,
            logger=_build_trajectory_logger(config),
            config=config,
            inference_devices=inference_devices,
            action_prefix=config["policy"]["action_prefix"],
        )
    except BaseException:
        environment.close()
        raise


def _build_clock(config: dict[str, Any]) -> EnvClock:
    env_cfg = config["env"]
    return EnvClock(
        env_fps=float(env_cfg["env_fps"]),
        obs_fps=float(env_cfg["obs_fps"]),
    )

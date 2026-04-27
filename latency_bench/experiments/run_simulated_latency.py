from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from latency_bench.core.clock import EnvClock
from latency_bench.core.config import load_config
from latency_bench.core.types import Action, LatencySourceType, PersistenceMode, StaleActionMode
from latency_bench.envs.factory import build_env_adapter
from latency_bench.executors.simulated import SimulatedLatencyExecutor
from latency_bench.latency.sources import (
    EmpiricalDistributionLatencySource,
    FixedLatencySource,
    LatencySource,
    TraceReplayLatencySource,
    ZeroLatencySource,
)
from latency_bench.logging.trajectory_logger import TrajectoryLogger
from latency_bench.observation.builder import ObservationBuilder
from latency_bench.observation.history import HistoryBuffer
from latency_bench.policy.base import ConstantPolicyRunner, PolicyRunner, RandomPolicyRunner
from latency_bench.scheduler.action_queue import ActionScheduler
from latency_bench.scheduler.decision import (
    EveryFrameDecisionScheduler,
    FixedFrameIntervalDecisionScheduler,
    FixedFrequencyDecisionScheduler,
)
from latency_bench.scheduler.persistence import HoldPreviousActionPolicy, NoopAfterOneShotPolicy


def build_executor(config: dict[str, Any]) -> SimulatedLatencyExecutor:
    env_cfg = config["env"]
    experiment_cfg = config["experiment"]
    control_cfg = config["control"]
    scheduler_cfg = config["scheduler"]

    noop_action = Action(value=env_cfg["noop_action"], name=str(env_cfg["noop_action"]), is_noop=True)
    env = build_env_adapter(config, noop_action)
    clock = EnvClock(fps_env=env.fps_env)
    policy = _build_policy(config, noop_action)
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


def run_from_config(config: dict[str, Any], extra_metadata: dict[str, Any] | None = None) -> list:
    executor = build_executor(config)
    experiment_cfg = config["experiment"]
    seed = int(experiment_cfg.get("seed", 0))
    num_episodes = int(experiment_cfg.get("num_episodes", 1))
    max_env_steps = int(experiment_cfg.get("max_env_steps", 10000))
    metrics = []
    for episode_id in range(num_episodes):
        episode_metrics = executor.run_episode(
            episode_id=episode_id,
            seed=seed + episode_id,
            max_env_steps=max_env_steps,
        )
        episode_metrics.metadata["config_name"] = experiment_cfg.get("name")
        if extra_metadata:
            episode_metrics.metadata.update(extra_metadata)
        metrics.append(episode_metrics)
    if executor.logger is not None:
        executor.logger.flush()
    return metrics


def _build_policy(config: dict[str, Any], noop_action: Action) -> PolicyRunner:
    policy_cfg = config["policy"]
    policy_type = policy_cfg.get("type", "random")
    action_values = policy_cfg.get("actions", [noop_action.value, "flap"])
    actions = [
        noop_action if value == noop_action.value else Action(value=value, name=str(value), is_oneshot=True)
        for value in action_values
    ]
    if policy_type == "random":
        return RandomPolicyRunner(actions=actions, seed=config["experiment"].get("seed"))
    if policy_type == "constant":
        value = policy_cfg.get("action", actions[0].value)
        action = noop_action if value == noop_action.value else Action(value=value, name=str(value), is_oneshot=True)
        return ConstantPolicyRunner(action=action)
    raise ValueError(f"Unsupported policy.type: {policy_type}")


def _build_decision_scheduler(control_cfg: dict[str, Any], fps_env: float):
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
    if latency_type == LatencySourceType.EMPIRICAL_DISTRIBUTION.value:
        return EmpiricalDistributionLatencySource(
            latency_values_ms=latency_cfg["latency_values_ms"],
            seed=latency_cfg.get("seed"),
        )
    if latency_type == LatencySourceType.TRACE_REPLAY.value:
        return TraceReplayLatencySource(
            trace_path=latency_cfg["trace_path"],
            loop_trace=bool(latency_cfg.get("loop_trace", False)),
        )
    raise ValueError(f"Unsupported latency.type for simulated CLI: {latency_type}")


def _build_persistence_policy(scheduler_cfg: dict[str, Any]):
    mode = PersistenceMode(scheduler_cfg.get("persistence_mode", PersistenceMode.NOOP_AFTER_ONESHOT.value))
    if mode == PersistenceMode.HOLD_PREVIOUS:
        return HoldPreviousActionPolicy()
    if mode == PersistenceMode.NOOP_AFTER_ONESHOT:
        return NoopAfterOneShotPolicy()
    raise ValueError(f"Unsupported persistence mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulated latency control experiment.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    metrics = run_from_config(config)
    for episode_metrics in metrics:
        print(
            f"episode={episode_metrics.episode_id} "
            f"return={episode_metrics.episode_return:.3f} "
            f"steps={episode_metrics.survival_steps} "
            f"mean_latency_ms={episode_metrics.mean_latency_ms}"
        )


if __name__ == "__main__":
    main()

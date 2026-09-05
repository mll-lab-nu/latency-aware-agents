"""Single evaluation driver: run one config's episodes and attach metadata.

This is the core ``run_from_config`` and its episode-side helpers. Sweep/suite
orchestration lives in :mod:`latency_bench.eval.sweeps`; the CLI in
:mod:`latency_bench.run`.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from training.common.utils import seed_everything
from latency_bench.core.types import EpisodeMetrics, ExecutorMode
from latency_bench.eval.config import _episode_seed, _eval_episodes, _eval_max_steps
from latency_bench.eval.reporting import _write_non_sweep_summary
from latency_bench.envs.base import EnvAdapter
from latency_bench.executors.base import BatchedExecutor
from latency_bench.executors.factory import build_executor
from latency_bench.executors.realtime_warmup import plot_realtime_eval_latency
from latency_bench.latency.config import latency_type_from_config
from latency_bench.logging.action_trace_replay import record_videos_from_action_trace
from latency_bench.logging.video import select_episode_return_stratified


def run_from_config(
    config: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
    *,
    write_summary: bool = True,
    on_episode_complete: Callable[[EpisodeMetrics], None] | None = None,
    episode_ids: Sequence[int] | None = None,
    policy: Any | None = None,
    env: EnvAdapter | None = None,
    env_backend: Any | None = None,
    inference_devices: list[str] | None = None,
) -> list[EpisodeMetrics]:
    if (
        policy is None
        and env is None
        and env_backend is None
        and config["policy"]["type"] == "starvla"
    ):
        from latency_bench.policy.starvla import prepare_starvla_checkpoint_input_config

        prepare_starvla_checkpoint_input_config(config)

    experiment_cfg = config["experiment"]
    policy_cfg = config["policy"]
    logging_cfg = config["logging"]
    seed = int(experiment_cfg.get("seed", 0))
    eval_max_steps = _eval_max_steps(config)
    configured_num_episodes = _eval_episodes(config)
    selected_episode_ids = list(range(configured_num_episodes)) if episode_ids is None else list(episode_ids)
    seed_everything(seed)

    executor_kwargs = {}
    if policy is not None:
        executor_kwargs["policy"] = policy
    if env is not None:
        executor_kwargs["env"] = env
    if env_backend is not None:
        executor_kwargs["env_backend"] = env_backend
    if inference_devices is not None:
        executor_kwargs["inference_devices"] = inference_devices
    executor = build_executor(config, **executor_kwargs)
    metrics = []
    warmup_metadata_by_episode: dict[int, dict[str, Any]] = {}
    try:
        if isinstance(executor, BatchedExecutor):
            warmup_metadata = executor.run_warmup()
            run_episodes_kwargs: dict[str, Any] = {
                "episode_ids": selected_episode_ids,
                "seeds": [_episode_seed(config, episode_id) for episode_id in selected_episode_ids],
                "eval_max_steps": eval_max_steps,
            }
            if on_episode_complete is not None:
                run_episodes_kwargs["on_episode_complete"] = on_episode_complete
            metrics = list(executor.run_episodes(**run_episodes_kwargs))
            warmup_metadata_by_episode.update(
                (episode_id, warmup_metadata) for episode_id in selected_episode_ids
            )
        else:
            warmup_metadata = executor.run_warmup()
            for episode_id in selected_episode_ids:
                warmup_metadata_by_episode[episode_id] = warmup_metadata
                episode_metrics = executor.run_episode(
                    episode_id=episode_id,
                    seed=_episode_seed(config, episode_id),
                    eval_max_steps=eval_max_steps,
                )
                metrics.append(episode_metrics)
                if on_episode_complete is not None:
                    on_episode_complete(episode_metrics)
        metrics.sort(key=lambda item: int(item.episode_id))
        for episode_metrics in metrics:
            for key, value in _evaluation_raw_fact_metadata(config, int(episode_metrics.episode_id)).items():
                if episode_metrics.metadata.get(key) is None:
                    episode_metrics.metadata[key] = value
            episode_metrics.metadata.update(warmup_metadata_by_episode[int(episode_metrics.episode_id)])
            if "measurement" in config:
                episode_metrics.metadata["measurement"] = config["measurement"]
            episode_metrics.metadata["config_name"] = experiment_cfg.get("name")
            episode_metrics.metadata["run_name"] = experiment_cfg.get("name")
            if "checkpoint_path" in policy_cfg:
                episode_metrics.metadata["checkpoint_path"] = policy_cfg["checkpoint_path"]
            if policy_cfg["type"] == "official_gr00t":
                episode_metrics.metadata["checkpoint_path"] = policy_cfg["model_path"]
            if "profile_path" in config["latency"]:
                episode_metrics.metadata["source_profile_path"] = config["latency"]["profile_path"]
            if "checkpoint_kind" in policy_cfg:
                episode_metrics.metadata["checkpoint_kind"] = str(policy_cfg["checkpoint_kind"])
            episode_metrics.metadata["output_dir"] = str(logging_cfg["output_dir"])
            if extra_metadata:
                episode_metrics.metadata.update(extra_metadata)
        if executor.logger is not None:
            executor.logger.flush()
        _record_realtime_eval_latency_plot(config, executor)
        if write_summary:
            _write_non_sweep_summary(config, metrics)
        _record_stratified_replay_videos(config, metrics, seed=seed)
    finally:
        executor.close()
    return metrics


def _record_realtime_eval_latency_plot(config: dict[str, Any], executor: Any) -> None:
    if ExecutorMode(config["executor"]["mode"]) != ExecutorMode.REALTIME:
        return
    if not config["logging"]["save_latency_records"]:
        return

    latency_values = list(executor.logger.latency_ms_values)
    plot_realtime_eval_latency(
        latency_values,
        Path(config["logging"]["output_dir"]) / "eval_latency_trace.png",
    )


def _record_stratified_replay_videos(
    config: dict[str, Any],
    metrics: list[EpisodeMetrics],
    *,
    seed: int,
) -> None:
    if "video" not in config["logging"]:
        return
    video_cfg = config["logging"]["video"]
    if not video_cfg["enabled"]:
        return
    if not config["logging"].get("save_step_records", False):
        # Replay reads steps.jsonl, which is only written when save_step_records is on.
        # Without it (e.g. factor-sweep evals) skip video instead of crashing on a missing file.
        return
    if ExecutorMode(config["executor"]["mode"]) == ExecutorMode.REALTIME:
        return
    selections = select_episode_return_stratified(
        metrics,
        num_bins=video_cfg["num_bins"],
        seed=seed,
    )
    record_videos_from_action_trace(config, selections=selections, metrics=metrics)


def _evaluation_raw_fact_metadata(config: dict[str, Any], episode_id: int) -> dict[str, Any]:
    env_cfg = config.get("env", {})
    policy_cfg = config.get("policy", {})
    latency_cfg = config.get("latency", {})
    executor_cfg = config.get("executor", {})
    env_fps = float(env_cfg["env_fps"]) if "env_fps" in env_cfg else None
    obs_fps = float(env_cfg["obs_fps"]) if "obs_fps" in env_cfg else None
    frame_ms = None if env_fps is None or env_fps <= 0 else 1000.0 / env_fps
    executor_mode = str(executor_cfg.get("mode", "")).strip().lower()
    latency_type = latency_type_from_config(latency_cfg)
    if executor_mode == "paused":
        latency_type = "zero"
    elif executor_mode == "realtime":
        latency_type = "measured"
    return {
        "mode": executor_cfg.get("mode"),
        "episode_seed": _episode_seed(config, episode_id),
        "policy_id": _metadata_id(policy_cfg, "policy_id", "id", "type"),
        "env_id": _metadata_id(env_cfg, "env_id", "id", "name"),
        "model_id": latency_cfg.get("model_id"),
        "gpu_class": latency_cfg.get("gpu_class"),
        "workload_id": latency_cfg.get("workload_id"),
        "instance_id": latency_cfg.get("instance_id"),
        "source_run_id": latency_cfg.get("source_run_id"),
        "profile_ref": latency_cfg.get("profile_ref"),
        "hf_commit": latency_cfg.get("hf_commit"),
        "env_fps": env_fps,
        "obs_fps": obs_fps,
        "frame_ms": frame_ms,
        "latency_type": latency_type,
    }


def _metadata_id(config: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return str(value)
    return None

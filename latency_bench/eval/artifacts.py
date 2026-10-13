"""Read pre-migration run snapshots at artifact boundaries, without executing them."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from latency_bench.core.latency_distribution import (
    LatencyDistributionSpec,
    distribution_frames_to_ms_latency_config,
    latency_distribution_label,
)
from latency_bench.policy.model_source import set_model_files
from latency_bench.envs.tasks.contracts import canonical_gymnasium_contract


def read_runtime_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Translate archived run inputs while leaving the stored artifact unchanged."""
    config = deepcopy(snapshot)
    if "profiling" in config:
        config["profiling"].pop("publish_repo", None)
    provenance = deepcopy(config["provenance"]) if "provenance" in config else None
    evaluation = config["evaluation"]
    if "eval_episodes" not in evaluation:
        return config
    executor = config["executor"]
    policy = config["policy"]
    for old, new in (
        ("eval_episodes", "episodes"),
        ("eval_max_steps", "max_env_steps"),
    ):
        if old in evaluation:
            evaluation[new] = evaluation.pop(old)
    if "eval_parallel_envs" in evaluation:
        executor["parallel_envs"] = evaluation.pop("eval_parallel_envs")
    if "latency_bench_env_backend" in evaluation:
        executor["env_backend"] = evaluation.pop("latency_bench_env_backend")
    if "simulated_inference_pool" in executor:
        executor["inference_pool"] = executor.pop("simulated_inference_pool")
    if policy["type"] in {
        "flappy_sf",
        "demon_attack_sf",
        "deadly_corridor_sf",
        "gymnasium_sf",
    }:
        policy["type"] = "sample_factory"
    if "device" in policy:
        device = policy.pop("device")
        if "inference_devices" not in executor:
            executor["inference_devices"] = [device]
    if "eval_deterministic" in evaluation:
        deterministic = evaluation.pop("eval_deterministic")
        if "deterministic" not in policy:
            policy["deterministic"] = deterministic
    env = config["env"]
    if env["name"] in {"gymnasium", "balance_profile"}:
        env.update(canonical_gymnasium_contract(env))
        env["name"] = env.pop("task_name")
        env["integration_name"] = "gymnasium"
    checkpoint = policy.pop("checkpoint_path", None)
    model_config = policy.pop("model_config_path", None)
    if checkpoint or model_config:
        set_model_files(config, checkpoint=checkpoint, model_config=model_config)
    suites = []
    old_suites = evaluation.pop("eval_suites", {}) or {}
    fps = config["env"]["env_fps"]
    seed = config["experiment"].get("seed")
    for values in old_suites.get("fixed", []) or []:
        values = range(values[0], values[1] + 1) if len(values) == 2 else values
        for value in values:
            suites.append(
                {
                    "name": f"fixed_{value}",
                    "latency": {"method": "zero"}
                    if value == 0
                    else {
                        "method": "fixed",
                        "fixed_latency_ms": value * 1000.0 / fps,
                    },
                }
            )
    for method in ("normal", "uniform"):
        for values in old_suites.get(method, []) or []:
            spec = LatencyDistributionSpec(
                type=method,
                mean=values[0] if method == "normal" else None,
                std=values[1] if method == "normal" else None,
                min=values[-2],
                max=values[-1],
            )
            suites.append(
                {
                    "name": latency_distribution_label(spec),
                    "latency": distribution_frames_to_ms_latency_config(
                        spec, env_fps=fps, seed=seed
                    ),
                }
            )
    for method in ("iid", "temporal"):
        for entry in old_suites.get(method, []) or []:
            suites.append(
                {
                    "name": entry["name"],
                    "latency": {
                        "method": method,
                        "profile_path": entry["profile_path"],
                        "seed": entry["seed"],
                    },
                }
            )
    values = evaluation.pop("eval_latency_values", None)
    if not suites and values is not None:
        for value in values:
            suites.append(
                {
                    "name": f"eval_latency_{value}",
                    "latency": {"method": "zero"}
                    if value == 0
                    else {
                        "method": "fixed",
                        "fixed_latency_ms": value * 1000.0 / fps,
                    },
                }
            )
    evaluation["suites"] = suites
    latency = config.get("latency", {})
    if "sync_cuda" in latency:
        executor["sync_cuda"] = latency.pop("sync_cuda")
    latency.pop("device", None)
    if executor["mode"] != "simulated":
        config.pop("latency", None)
    for key in ("realtime_pipeline_profile", "simulated_pipeline_profile"):
        config["logging"].pop(key, None)
    if "profiling" in config:
        options = config["profiling"]
        options.pop("hf_repo", None)
        if "session_count" in options:
            options["repeats"] = options.pop("session_count")
        artifact = options.pop("model_artifact", None)
        if artifact is not None:
            policy.update(
                {
                    key: artifact[key]
                    for key in (
                        "source",
                        "checkpoint",
                        "model_config",
                        "hf_repo",
                        "revision",
                    )
                    if key in artifact
                }
            )
            policy["root"] = (
                artifact["local_root"]
                if artifact["source"] == "local"
                else artifact["hf_path"]
            )
        options.pop("runtime_artifact", None)
    if provenance is not None:
        config["provenance"] = provenance
    return config

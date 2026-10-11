#!/usr/bin/env bash
# Reference dataset recipe: materialize its settings, collect, then optionally upload.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <checkpoint-root> <output-dir>" >&2
    exit 2
fi
"${PYTHON:-python}" - "$1" "$2" <<'PY'
import json
import os
import sys
from pathlib import Path
import yaml
if os.environ.get("DISABLE_GYM_PLUGINS") == "1":
    from importlib import metadata
    entry_points = metadata.entry_points
    def explicit_gym_registration(*args, **kwargs):
        group = kwargs.get("group", args[0] if args else None)
        return () if group == "gymnasium.envs" else entry_points(*args, **kwargs)
    metadata.entry_points = explicit_gym_registration
from latency_bench.data.rollout import collect_rollouts
from latency_bench.integrations.sample_factory.teacher_config import load_rollout_training_config

checkpoint, output = map(Path, sys.argv[1:])
device = os.environ.get("DEVICE", "cpu")
latency_path = os.environ.get("LATENCY_CONFIG")
descriptor, _ = load_rollout_training_config(checkpoint, latency_config_path=latency_path, device_override=device)
flat = descriptor["flat_cfg"]
latency = descriptor["latency_override"]
if latency is None:
    methods = {"zero": "zero", "fixed": "fixed", "normal_distribution": "normal",
               "uniform_distribution": "uniform", "profile_sample": "temporal", "iid": "iid"}
    latency = {"method": methods[flat["latency_type"]]}
    for key in ("fixed_latency_ms", "mean_latency_ms", "std_latency_ms", "min_latency_ms", "max_latency_ms"):
        if key in flat and flat[key] is not None: latency[key] = flat[key]
    for old, new in (("latency_profile_path", "profile_path"), ("latency_seed", "seed"), ("latency_profile_worker_slot", "profile_worker_slot")):
        if old in flat and flat[old] is not None: latency[new] = flat[old]
collection = {
    "two_pass": True,
    "val_fraction": float(os.environ.get("VAL_FRACTION", "0.01")),
    "split_seed": int(os.environ.get("SPLIT_SEED", "0")),
    "replay_num_envs": int(os.environ.get("REPLAY_NUM_ENVS", "16")),
    "image_writer_workers": int(os.environ.get("IMAGE_WRITER_WORKERS", "16")),
    "row_buffer_size": int(os.environ.get("ROW_BUFFER_SIZE", "8192")),
    "context_window": int(os.environ.get("CONTEXT_WINDOW", "1")),
    "filter_preset": os.environ.get("FILTER_PRESET", "clean_v1"),
    "probe_only": os.environ.get("PROBE_ONLY", "0").lower() in {"1", "true"},
    "image_transform": {
        "image_transform": os.environ.get("IMAGE_TRANSFORM", "raw_rgb"),
        "history_frames": int(os.environ.get("GHOST_TRAIL_HISTORY_FRAMES", "5")),
        "gamma": float(os.environ.get("GHOST_TRAIL_GAMMA", "1.3")),
        "min_alpha": int(os.environ.get("GHOST_TRAIL_MIN_ALPHA", "35")),
        "ground_fraction": float(os.environ.get("GHOST_TRAIL_GROUND_FRACTION", "0.22")),
        "scroll_px_per_step": float(os.environ.get("GHOST_TRAIL_SCROLL_PX_PER_STEP", "4.0")),
    },
}
for field, variable, convert in (
    ("max_attempts", "MAX_ATTEMPTS", int),
    ("episode_return_gt", "EPISODE_RETURN_GT", float),
    ("demon_episode_threshold", "DEMON_EPISODE_THRESHOLD", float),
    ("export_zero_action_history", "EXPORT_ZERO_ACTION_HISTORY", int),
):
    if variable in os.environ and os.environ[variable].lower() not in {"", "none"}:
        collection[field] = convert(os.environ[variable])
if os.environ.get("EXPORT_IMAGE_SIZE"):
    collection["export_image_size"] = list(map(int, os.environ["EXPORT_IMAGE_SIZE"].split(",")))
config = {
    "experiment": {"name": "teacher_collection", "seed": int(flat["seed"]) + int(os.environ.get("ROLLOUT_SEED_OFFSET", "0"))},
    "env": {"name": descriptor["env_name"], "env_fps": flat["env_fps"], "obs_fps": flat["obs_fps"]},
    "policy": {"type": "sample_factory", "source": "local", "root": descriptor["checkpoint_experiment_dir"],
        "checkpoint": str(Path(descriptor["checkpoint_path"]).relative_to(descriptor["checkpoint_experiment_dir"])) if descriptor["checkpoint_path"] is not None else None,
        "deterministic": os.environ.get("DETERMINISTIC", "true").lower() in {"1", "true"}},
    "executor": {"mode": "simulated", "parallel_envs": int(os.environ.get("NUM_ENVS", "12")), "inference_devices": [device],
        "simulated_worker_capacity": flat["simulated_worker_capacity"]},
    "scheduler": {"hold_policy": flat["hold_policy"], "ordering_policy": flat["ordering_policy"]},
    "latency": latency,
    "evaluation": {"episodes": int(os.environ.get("MAX_EPISODES", "1000")), "max_env_steps": int(os.environ.get("MAX_STEPS", "1000"))},
    "logging": {"output_dir": str(output)}, "collection": collection,
}
if "GYM_MAKE_KWARGS_JSON" in os.environ:
    config["env"]["make_kwargs"] = json.loads(os.environ["GYM_MAKE_KWARGS_JSON"])
config_path = output.with_name(output.name + ".collect.yaml")
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
result = collect_rollouts(config)
if "metadata" in result and os.environ.get("HF_REPO_ID"):
    from latency_bench.data.hf_parquet import load_parquet_splits
    options = {"repo_id": os.environ["HF_REPO_ID"], "private": os.environ.get("HF_PRIVATE", "0").lower() in {"1", "true"},
               "max_shard_size": os.environ.get("HF_MAX_SHARD_SIZE", "500MB")}
    for variable, field in (("HF_TOKEN", "token"), ("HF_CONFIG_NAME", "config_name"), ("HF_DATA_DIR", "data_dir")):
        if os.environ.get(variable): options[field] = os.environ[variable]
    load_parquet_splits(output).push_to_hub(**options)
print(json.dumps(result, default=str))
PY

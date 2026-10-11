"""Package a history VLA and evaluate the explicit zero/profile environment condition."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml
import numpy as np

from latency_bench.core.config import load_config
from latency_bench.eval.reporting import _log_corrected_eval_summaries_to_wandb


ROOT = Path(__file__).resolve().parents[3]
EVAL_CONFIGS = {
    "flappy": "configs/experiments/flappy/starvla_eval_zero_latency.yaml",
    "demon_attack": "configs/experiments/demon_attack/starvla_eval_zero_latency.yaml",
    "deadly_corridor": "configs/experiments/deadly_corridor/starvla_eval_zero_latency.yaml",
    "atlantis": "configs/experiments/atlantis/starvla_eval_zero_latency.yaml",
    "asterix": "configs/experiments/asterix/starvla_eval_zero_latency.yaml",
}
RATES = {"flappy": (10, 10), "demon_attack": (60, 15), "deadly_corridor": (35, 8.75),
         "atlantis": (60, 15), "asterix": (60, 15)}
HORIZONS = {"flappy": 3600, "demon_attack": 7200, "deadly_corridor": 3600,
            "atlantis": 7200, "asterix": 7200}
EVAL_PROTOCOL = "hist8-history01-sf-render-v4"
EVAL_PREFIX = "eval_history01_v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_evaluation(output_dir: Path, episodes: int) -> dict:
    records = [json.loads(line) for line in (output_dir / "episode_metrics.jsonl").read_text().splitlines()]
    seeds = sorted(record["metadata"]["episode_seed"] for record in records)
    assert seeds == list(range(1_000_000, 1_000_000 + episodes))
    counts = Counter()
    raw_scores = []
    with (output_dir / "actions.jsonl").open() as handle:
        for line in handle:
            action = json.loads(line)
            counts[json.dumps(action["action"]["value"], separators=(",", ":"))] += 1
            raw_scores.append(action["metadata"]["raw_action_scores"])
    scores = np.asarray(raw_scores)
    result = {
        "eval_protocol_version": EVAL_PROTOCOL, "history_encoding": "binary_0_1",
        "episodes": len(records), "episode_seeds": seeds,
        "issued_actions": sum(counts.values()), "action_counts": dict(counts),
        "largest_action_fraction": max(counts.values()) / sum(counts.values()),
        "raw_score_min": scores.min(axis=0).tolist(), "raw_score_max": scores.max(axis=0).tolist(),
        "raw_score_mean": scores.mean(axis=0).tolist(),
    }
    (output_dir / "action_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def package_bundle(
    run_dir: Path, dataset_dir: Path, collection_provenance: Path,
    destination: Path, *, step: int, wandb_run_id: str,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / f"checkpoints/steps_{step}_pytorch_model.pt"
    target = destination / "checkpoints/model.pt"
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        target.unlink()
    os.link(checkpoint, target)
    config = yaml.safe_load((run_dir / "config.full.yaml").read_text())
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    inference = {
        "framework": deepcopy(config["framework"]),
        "version_id": config["version_id"],
        "datasets": {"vla_data": {
            key: config["datasets"]["vla_data"][key]
            for key in ("include_state", "obs_image_size", "image_mode", "num_obs_frames")
        } | {"robot_type": manifest["robot_type"]}},
        "trainer": {"pretrained_checkpoint": None},
        "backbone": {"repo_id": "Qwen/Qwen3-VL-4B-Instruct",
                     "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17"},
    }
    inference["framework"]["qwenvl"]["base_vlm"] = "Qwen/Qwen3-VL-4B-Instruct"
    if manifest["robot_type"] in {"rl_games_atlantis", "rl_games_asterix"}:
        inference["rl_games"] = {"env_eval": {"image_size": config["rl_games"]["env_eval"]["image_size"]}}
    (destination / "config.yaml").write_text(yaml.safe_dump(inference, sort_keys=False))
    shutil.copyfile(run_dir / "config.full.yaml", destination / "config.full.yaml")
    shutil.copyfile(run_dir / "dataset_statistics.json", destination / "dataset_statistics.json")
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        shutil.copyfile(metadata_path, destination / "run_metadata.json")
        metadata = json.loads(metadata_path.read_text())
        shutil.copyfile(ROOT / "code_snapshots" / f"{metadata['code_snapshot_sha256']}.json",
                        destination / "training_code_snapshot.json")
    shutil.copyfile(dataset_dir / "manifest.json", destination / "manifest.json")
    shutil.copyfile(dataset_dir.parent / "quality_report.json", destination / "collection_quality_report.json")
    history_audit = dataset_dir.parent / "history_audit.json"
    if history_audit.exists():
        shutil.copyfile(history_audit, destination / "history_audit.json")
    shutil.copyfile(dataset_dir.parent / "raw/metadata.json", destination / "rollout_metadata.json")
    collection = json.loads(collection_provenance.read_text())
    provenance = {
        "collection": collection,
        "training_run_id": config["run_id"],
        "training_step": step,
        "global_batch": 128,
        "base_model": "Qwen/Qwen3-VL-4B-Instruct",
        "wandb_run_id": wandb_run_id,
        "checkpoint_sha256": file_sha256(target),
        "eval_protocol_version": EVAL_PROTOCOL,
        "history_encoding": "binary_0_1",
    }
    (destination / "checkpoint.sha256").write_text(f"{provenance['checkpoint_sha256']}  checkpoints/model.pt\n")
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def evaluation_config(
    task: str, model: str, bundle: Path, profile: Path, output_dir: Path,
    backbone: Path, vla_python: Path, *, device: str, episodes: int,
    env_condition: str = "profile", parallel_envs: int = 32,
    inference_devices: list[str] | None = None, max_steps: int | None = None,
) -> dict:
    devices = [device] if inference_devices is None else inference_devices
    config = load_config(ROOT / EVAL_CONFIGS[task])
    provenance = json.loads((bundle / "provenance.json").read_text())
    config["experiment"] = {
        "name": f"{provenance['training_run_id']}-step{provenance['training_step']}-history01-{env_condition}-par{parallel_envs}",
        "seed": 1_000_000,
    }
    config["env"].update({
        "env_fps": RATES[task][0], "obs_fps": RATES[task][1],
        "frame_stack": 1, "action_history_decisions": 8,
    })
    config["executor"].update({
        "mode": "simulated", "simulated_worker_capacity": 1,
        "inference_pool": True,
        "inference_devices": devices, "inference_batch_size": (parallel_envs + len(devices) - 1) // len(devices),
    })
    config["latency"] = {"method": "zero"} if env_condition == "zero" else {
        "method": "temporal", "profile_path": str(profile),
        "profile_worker_slot": 0, "seed": 271828, "add_latency_info": False,
    }
    config["policy"].update({
        "source": "local", "root": str(bundle), "checkpoint": "checkpoints/model.pt",
        "model_config": "config.yaml",
        "backbone_path": str(backbone),
        "device": device,
        "worker_python_executable": str(vla_python),
        "prompt_mode": "latency_neutral",
        "state_source": "transport",
    })
    config["policy"].pop("latency_prompt_map_path", None)
    config["policy"].pop("multibinary_threshold", None)
    config["policy"].pop("state_normalization", None)
    if task in {"atlantis", "asterix"}:
        config["policy"]["task_manifest_path"] = str(bundle / "manifest.json")
    if task == "asterix":
        config["policy"]["action_layout"] = "discrete_9"
    if task == "deadly_corridor":
        config["policy"]["action_layout"] = "multibinary_7"
        config["env"].update({
            "screen_resolution": "RES_160X120", "render_hud": True,
            "render_crosshair": False, "render_weapon": True,
            "render_decals": False, "render_particles": False,
        })
    config["executor"]["parallel_envs"] = parallel_envs
    config["policy"]["deterministic"] = True
    config["evaluation"].update({
        "episodes": episodes,
        "max_env_steps": HORIZONS[task] if max_steps is None else max_steps,
        "eval_raw_reward": True,
    })
    config["logging"].update({
        "output_dir": str(output_dir),
        "save_step_records": True,
        "save_action_records": True,
        "save_latency_records": True,
    })
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("bundle")
    pack.add_argument("--run-dir", type=Path, required=True)
    pack.add_argument("--dataset-dir", type=Path, required=True)
    pack.add_argument("--collection-provenance", type=Path, required=True)
    pack.add_argument("--output-dir", type=Path, required=True)
    pack.add_argument("--step", type=int, default=5000)
    pack.add_argument("--wandb-run-id", required=True)
    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--task", choices=EVAL_CONFIGS, required=True)
    evaluate.add_argument("--model", choices=("qwenoft", "qwengr00t", "qwenpi_v3"), required=True)
    evaluate.add_argument("--bundle-dir", type=Path, required=True)
    evaluate.add_argument("--profile-path", type=Path, required=True)
    evaluate.add_argument("--backbone-path", type=Path, required=True)
    evaluate.add_argument("--vla-python", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--episodes", type=int, default=100)
    evaluate.add_argument("--env-condition", choices=("zero", "profile"), default="profile")
    evaluate.add_argument("--parallel-envs", type=int, default=32)
    evaluate.add_argument("--inference-devices", nargs="+")
    evaluate.add_argument("--max-steps", type=int)
    evaluate.add_argument("--metric-prefix", default=EVAL_PREFIX)
    args = parser.parse_args()
    if args.command == "bundle":
        package_bundle(
            args.run_dir.resolve(), args.dataset_dir.resolve(),
            args.collection_provenance.resolve(), args.output_dir.resolve(), step=args.step,
            wandb_run_id=args.wandb_run_id,
        )
        return
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = evaluation_config(
        args.task, args.model, args.bundle_dir.resolve(), args.profile_path.resolve(),
        output_dir, args.backbone_path.resolve(), args.vla_python.resolve(),
        device=args.device, episodes=args.episodes,
        env_condition=args.env_condition, parallel_envs=args.parallel_envs,
        inference_devices=args.inference_devices, max_steps=args.max_steps,
    )
    config_path = output_dir / "eval_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copyfile(ROOT / "code_snapshot.json", output_dir / "eval_code_snapshot.json")
    provenance = json.loads((args.bundle_dir.resolve() / "provenance.json").read_text())
    receipt = {
        **provenance, "eval_protocol_version": EVAL_PROTOCOL,
        "history_encoding": "binary_0_1", "episodes": args.episodes,
        "env_condition": args.env_condition, "parallel_envs": args.parallel_envs,
        "max_episode_raw_frames": config["evaluation"]["max_env_steps"],
        "inference_devices": config["executor"]["inference_devices"],
        "episode_seeds": [1_000_000, 1_000_000 + args.episodes - 1],
        "profile_sha256": file_sha256(args.profile_path.resolve()) if args.env_condition == "profile" else None,
        "checkpoint_sha256": file_sha256(args.bundle_dir.resolve() / "checkpoints/model.pt"),
    }
    (output_dir / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    run_env = os.environ.copy()
    cuda_home = Path(sys.executable).parents[1]
    compiler = cuda_home / "bin/x86_64-conda-linux-gnu-g++"
    run_env["CUDA_HOME"] = str(cuda_home)
    run_env["CXX"] = str(compiler)
    run_env["NVCC_PREPEND_FLAGS"] = f"-ccbin={compiler}"
    subprocess.run([sys.executable, "-m", "latency_bench.run", "--config", str(config_path)],
                   cwd=ROOT, env=run_env, check=True)
    audit_evaluation(output_dir, args.episodes)
    if args.episodes == 100:
        provenance = json.loads((args.bundle_dir.resolve() / "provenance.json").read_text())
        config["logging"].update({
            "wandb_name": provenance["wandb_run_id"],
            "wandb_display_name": provenance["training_run_id"],
            "wandb_project": "starvla_tasks",
            "wandb_entity": "saberrr-zju",
        })
        label = "0" if args.env_condition == "zero" else "profile_sample"
        summary = json.loads((output_dir / f"queue_eval_latency_{label}.json").read_text())
        summary["checkpoint_global_step"] = provenance["training_step"]
        _log_corrected_eval_summaries_to_wandb(config, [summary], prefix=args.metric_prefix)


if __name__ == "__main__":
    main()

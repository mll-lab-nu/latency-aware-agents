"""Train matched QwenOFT students with the main-paper H8 recipe."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from latency_bench.utils.io import read_json, write_json_atomic
from scripts.experiments.temporal_profiles.teacher import validate_profile
from scripts.tasks.decision_history import train_vla_hist8


BASE_MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
BASE_MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
REFERENCE = {
    "repo_id": "MLL-Lab/LAGEN-models",
    "revision": "96d17ee124b503967bf44dd696579664c8eedd61",
    "path": "latency-aware/flappy/vla/starvla-qwenoft-h1/flappy-qwenoft-hist8-scratch-s42-v2",
}


def collection_record(collection: Path, method: str) -> dict:
    """Reject earlier 7,200-frame collections or a mismatched teacher method."""
    if not (collection / "DONE").is_file():
        raise ValueError(f"Collection/conversion is incomplete: {collection}/DONE is missing")
    protocol = read_json(collection / "protocol.json")
    settings = protocol["settings"]
    expected = {
        "method": method, "env_fps": 10, "obs_fps": 10, "seed": 42,
        "profile_seed": 42, "max_episode_raw_frames": 18000,
        "filter_preset": "none", "val_fraction": .1, "split_seed": 0,
        "deterministic": True,
    }
    if (protocol["profile_set"] != "irregular_20min"
            or any(settings.get(key) != value for key, value in expected.items())
            or settings.get("episode_return_gt") is not None):
        raise ValueError(f"Expected unfiltered {method} irregular_20min data at 10 FPS with an 18,000-frame cap")
    profile = validate_profile(collection / "profile/profile.json", method, profile_set="irregular_20min")
    completed = read_json(collection / "collection_complete.json")
    if (completed["status"] != "collection_complete" or completed["train_rows"] < 320000
            or completed["val_rows"] < 1):
        raise ValueError("Collection must be complete with at least 320,000 training rows and a validation split")
    for split, rows_key in (("lerobot", "train_rows"), ("lerobot__val", "val_rows")):
        manifest = read_json(collection / split / "manifest.json")
        expected_manifest = {
            "robot_type": "rl_games_flappy", "action_dim": 2, "state_dim": 24,
            "action_history_decisions": 8, "state_source": "decision_v1",
            "obs_fps": 10, "rows_unit": "decision_step", "frames": completed[rows_key],
            "state_normalization": {"min": [0.] * 24, "max": [1.] * 24},
        }
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            raise ValueError(f"Wrong H8 training contract: {collection / split / 'manifest.json'}")
        if not any((collection / split / "data").glob("chunk-*/*.parquet")):
            raise FileNotFoundError(f"No converted Parquet data in {collection / split / 'data'}")
    mixture = read_json(collection / "_generated_mixtures/lerobot.json")
    if mixture != {name: [[name, 1., "rl_games_flappy"]] for name in ("lerobot", "lerobot__val")}:
        raise ValueError("Expected a single matching train/validation mixture without other datasets")
    return {
        "collection_dir": str(collection), "collection_run_id": protocol["run_id"],
        "collection_settings": settings, "collection_counts": completed,
        "profile_source_run_id": profile["source_run_id"],
        "reference_training": REFERENCE, "model": "qwenoft", "global_batch": 128,
        "num_processes": 1, "distributed_backend": "none", "per_device_batch_size": 32,
        "gradient_accumulation_steps": 4,
        "base_model_repo": BASE_MODEL_REPO, "base_model_revision": BASE_MODEL_REVISION,
    }


def training_arguments(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        task="flappy", model="qwenoft", dataset_dir=args.collection_dir / "lerobot",
        base_model_dir=args.base_model_dir, output_root=args.output_root,
        run_id=f"flappy-qwenoft-{args.method}-irregular20m-hist8-s42",
        vlm_lr=3e-6, training_condition="profile", stop_after_steps=None,
        preflight=False, resume=args.resume, port=args.port,
        wandb_project=None if os.environ.get("WANDB_MODE") == "disabled" else args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("temporal", "ar1", "arma"), required=True)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29501)
    parser.add_argument("--wandb-project", default="flappy-irregular-vlas")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--prepare-only", action="store_true", help="Validate data and save config without loading a model")
    parser.add_argument("--resume", action="store_true", help="Resume the latest saved training state")
    args = parser.parse_args()
    args.collection_dir = args.collection_dir.resolve()
    args.output_root = args.output_root.resolve()
    record = collection_record(args.collection_dir, args.method)
    training = training_arguments(args)
    run_dir = args.output_root / training.run_id
    record_path = run_dir / "training_data.json"
    if record_path.exists() and read_json(record_path) != record:
        raise ValueError("Dataset or recipe changed; use a new VLA_OUTPUT_ROOT")
    if (run_dir / "run_metadata.json").exists() and not args.resume:
        raise ValueError("This training run already exists; use --resume or a new VLA_OUTPUT_ROOT")
    if args.resume and not any((run_dir / "checkpoints").glob("steps_*_state")):
        raise ValueError("No saved training state to resume")
    if not args.prepare_only:
        import torch
        if torch.cuda.device_count() != 1:
            raise ValueError("Each run requires exactly one visible GPU; set VLA_GPUS=0")
    write_json_atomic(record_path, record)
    print(f"Training {args.method}: {record['collection_counts']['train_rows']:,} rows; "
          "QwenOFT H8, 5,000 updates, global batch 128 (1 GPU x 32 x accumulation 4), seed 42", flush=True)
    train_vla_hist8.run(training, prepare_only=args.prepare_only, save_interval=5000,
                       eval_interval=5000, single_gpu=True)


if __name__ == "__main__":
    main()

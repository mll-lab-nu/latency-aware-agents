"""Download a profile-trained Flappy VLA and use the existing realtime evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from latency_bench.core.config import load_config
from latency_bench.utils.io import write_yaml
from latency_bench.utils.paths import REPO_ROOT


BACKBONE_REPO = "Qwen/Qwen3-VL-4B-Instruct"
BACKBONE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
CHECKPOINT = "checkpoints/steps_5000_pytorch_model.pt"
REQUIRED_FILES = (CHECKPOINT, "config.full.yaml", "dataset_statistics.json")


def prepare(args):
    from huggingface_hub import snapshot_download

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh EVAL_RUN_ID; evaluation output already exists: {output}")
    if args.episodes < 1 or args.max_steps < 1:
        raise ValueError("Episode count and raw-frame cap must be positive")
    print(f"Downloading {args.repo_id}/{args.method} at {args.revision}", flush=True)
    snapshot = Path(snapshot_download(
        repo_id=args.repo_id, revision=args.revision,
        allow_patterns=[f"{args.method}/{name}" for name in (
            *REQUIRED_FILES, "config.yaml", "train.yaml", "training_data.json",
            "training_code_snapshot.json", "run_metadata.json",
        )],
    ))
    model_root = snapshot / args.method
    for name in REQUIRED_FILES:
        path = model_root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Upload the completed {args.method} model first; missing {path}")
    backbone = Path(snapshot_download(repo_id=BACKBONE_REPO, revision=BACKBONE_REVISION))

    # This is the same single-worker realtime entrypoint used for profiling.
    # Keep its environment/action scheduling; replace the checkpoint and H8 clock.
    config = load_config(REPO_ROOT / "configs/examples/flappy/profile_openvla.yaml")
    config.pop("profiling")
    config["experiment"] = {"name": f"flappy-qwenoft-{args.method}-realtime-10fps", "seed": args.seed}
    config["env"].update(env_fps=10, obs_fps=10, frame_stack=1, action_history_decisions=8)
    config["executor"].update(mode="realtime", inference_devices=["cuda:0"], parallel_envs=1,
                               inference_pool=False, sync_cuda=True)
    config["policy"] = {
        "type": "starvla", "source": "local", "root": str(model_root),
        "checkpoint": CHECKPOINT, "model_config": "config.full.yaml",
        "backbone_path": str(backbone), "worker_python_executable": sys.executable,
        "device": "cuda:0", "unnorm_key": "new_embodiment",
        "prompt_mode": "latency_neutral", "state_source": "transport",
        # Training's generated `lerobot` mixture maps to this DataConfig. Its
        # dataset path/registration is unavailable on the profiling server.
        "normalization_robot_type": "rl_games_flappy",
        "deterministic": True, "actions": ["noop", "flap"],
    }
    config["evaluation"] = {
        "episodes": args.episodes, "max_env_steps": args.max_steps,
        "episode_seed_offset": 0, "eval_raw_reward": True, "suites": [],
    }
    config["logging"]["output_dir"] = str(output)
    config["provenance"] = {
        "model_artifact": {
            "source": "hf", "repo_id": args.repo_id, "revision": snapshot.name,
            "path_in_repo": args.method, "checkpoint": CHECKPOINT, "model_config": "config.full.yaml",
        },
        "backbone_artifact": {"source": "hf", "repo_id": BACKBONE_REPO, "revision": BACKBONE_REVISION},
        "evaluation_code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "deployment_condition": "realtime_without_injected_workload",
    }
    output.mkdir(parents=True)
    config_path = output / "eval_config.yaml"
    write_yaml(config_path, config)
    print(f"Prepared {config_path}\n{args.episodes} episodes; {args.max_steps} raw frames maximum "
          f"per episode; 10 FPS; seeds {args.seed}–{args.seed + args.episodes - 1}", flush=True)
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("temporal", "ar1", "arma"), required=True)
    parser.add_argument("--repo-id", default="talha15032/flappy-irregular-qwenoft")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=1000000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config_path = prepare(args)
    if args.prepare_only:
        return
    # Preserve the existing CLI's episode summaries, timing logs and warmup.
    command = [sys.executable, "-m", "latency_bench", "eval", "--config", str(config_path)]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    print(f"Evaluation complete: {config_path.parent}", flush=True)


if __name__ == "__main__":
    main()

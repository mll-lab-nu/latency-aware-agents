"""Train one scratch StarVLA H1 policy from decision-history teacher data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[3]
STARVLA = ROOT / "third_party/starVLA"
sys.path.insert(0, str(STARVLA))

from examples.rl_games.scripts.launch_train import compose_training_config


MODEL_GROUPS = {"qwenoft": "openvla", "qwengr00t": "gr00t", "qwenpi_v3": "pi05"}
HISTORY_DIMS = {"flappy": 24, "demon_attack": 88, "deadly_corridor": 120, "atlantis": 56, "asterix": 136}
ACTION_DIMS = {"flappy": 2, "demon_attack": 6, "deadly_corridor": 7, "atlantis": 4, "asterix": 9}
STATE_RE = re.compile(r"steps_(\d+)_state$")


def _latest_state(checkpoints: Path) -> tuple[int, Path]:
    states = sorted(
        (int(match.group(1)), path)
        for path in checkpoints.glob("steps_*_state")
        if (match := STATE_RE.fullmatch(path.name))
        and (checkpoints / f"steps_{match.group(1)}_pytorch_model.pt").is_file()
    )
    return states[-1]


def training_config(args: argparse.Namespace):
    model_group = MODEL_GROUPS[args.model]
    cfg = compose_training_config("train", model_group, args.task, "scratch", "single", [])
    data_mix = args.dataset_dir.name
    mixture = args.dataset_dir.parent / "_generated_mixtures" / f"{data_mix}.json"
    run_id = args.run_id
    wandb_name = run_id
    OmegaConf.update(cfg, "run_id", run_id)
    OmegaConf.update(cfg, "run_root_dir", str(args.output_root))
    OmegaConf.update(cfg, "seed", 42)
    OmegaConf.update(cfg, "wandb_entity", args.wandb_entity)
    OmegaConf.update(cfg, "wandb_project", args.wandb_project)
    OmegaConf.update(cfg, "wandb_name", wandb_name, force_add=True)
    OmegaConf.update(cfg, "wandb_group", "decision-history-vla-h8-v2", force_add=True)
    OmegaConf.update(
        cfg, "wandb_tags", [args.task, args.model, "hist8-v2", "history01", "scratch", f"{args.training_condition}-latency"],
        force_add=True,
    )
    OmegaConf.update(cfg, "training_latency_condition", f"{args.training_condition}-latency", force_add=True)
    OmegaConf.update(cfg, "base_model.repo_id", "Qwen/Qwen3-VL-4B-Instruct")
    OmegaConf.update(cfg, "paths.base_model_dir", str(args.base_model_dir))
    OmegaConf.update(cfg, "framework.qwenvl.base_vlm", str(args.base_model_dir))
    OmegaConf.update(cfg, "framework.qwenvl.enable_gradient_checkpointing", True)
    OmegaConf.update(cfg, "framework.action_model.action_dim", ACTION_DIMS[args.task])
    OmegaConf.update(cfg, "framework.action_model.action_env_dim", ACTION_DIMS[args.task])
    OmegaConf.update(cfg, "framework.action_model.state_dim", HISTORY_DIMS[args.task])
    OmegaConf.update(cfg, "framework.action_model.action_horizon", 1)
    OmegaConf.update(cfg, "framework.action_model.future_action_window_size", 0)
    OmegaConf.update(cfg, "framework.action_model.past_action_window_size", 0)
    if args.model in {"qwenoft", "qwenpi_v3"}:
        OmegaConf.update(cfg, "framework.action_model.state_encoding", "continuous_projector", force_add=True)
    if args.model == "qwenoft":
        OmegaConf.update(
            cfg, "framework.action_model.loss_type",
            "multibinary_bce" if args.task == "deadly_corridor" else "discrete_ce",
        )
    OmegaConf.update(cfg, "datasets.vla_data.include_state", True)
    OmegaConf.update(cfg, "datasets.vla_data.data_root_dir", str(args.dataset_dir.parent))
    OmegaConf.update(cfg, "datasets.vla_data.data_mix", data_mix)
    OmegaConf.update(cfg, "datasets.vla_data.eval_data_mix", f"{data_mix}__val")
    OmegaConf.update(cfg, "datasets.vla_data.custom_mixtures_path", str(mixture))
    OmegaConf.update(cfg, "datasets.vla_data.per_device_batch_size", 64)
    OmegaConf.update(cfg, "datasets.vla_data.num_workers", 16)
    OmegaConf.update(cfg, "datasets.vla_data.eval_num_workers", 16)
    OmegaConf.update(cfg, "datasets.vla_data.num_obs_frames", 1)
    OmegaConf.update(cfg, "datasets.vla_data.image_mode", "single")
    OmegaConf.update(cfg, "datasets.vla_data.sequential_step_sampling", False)
    OmegaConf.update(cfg, "datasets.vla_data.prompt_mode", "raw")
    OmegaConf.update(cfg, "trainer.max_train_steps", 2 if args.preflight else 5000)
    OmegaConf.update(cfg, "trainer.num_warmup_steps", 0 if args.preflight else 100)
    OmegaConf.update(cfg, "trainer.save_interval", 2 if args.preflight else 500)
    OmegaConf.update(cfg, "trainer.eval_interval", 2 if args.preflight else 500)
    OmegaConf.update(cfg, "trainer.eval_num_batches", 1 if args.preflight else 50)
    OmegaConf.update(cfg, "trainer.gradient_accumulation_steps", 1)
    OmegaConf.update(cfg, "trainer.learning_rate", {
        "base": 3e-5, "qwen_vl_interface": args.vlm_lr, "action_model": 1e-4,
    })
    OmegaConf.update(cfg, "trainer.lr_scheduler_type", "cosine_with_min_lr")
    OmegaConf.update(cfg, "trainer.scheduler_specific_kwargs", {"min_lr_rate": 1 / 30},
                     merge=False, force_add=True)
    OmegaConf.update(cfg, "trainer.gradient_clipping", 1.0)
    OmegaConf.update(cfg, "trainer.distributed_backend", "deepspeed")
    OmegaConf.update(cfg, "trainer.loss_scale", {"vla": 1.0, "vlm": 0.0})
    OmegaConf.update(cfg, "checkpoint.load", "none")
    OmegaConf.update(cfg, "checkpoint.save_best_model", False)
    OmegaConf.update(cfg, "checkpoint.save_final_model", True)
    OmegaConf.update(cfg, "checkpoint.save_pt_file", True)
    OmegaConf.update(cfg, "checkpoint.save_training_state", True)
    OmegaConf.update(cfg, "checkpoint.local.keep_last_n", 2)
    OmegaConf.update(cfg, "rl_games.env_eval.enabled", False)
    OmegaConf.update(cfg, "rl_games.env_eval.mid_train.enabled", False)
    OmegaConf.update(cfg, "rl_games.env_eval.post_train.enabled", False)
    if args.stop_after_steps is not None:
        OmegaConf.update(cfg, "trainer.stop_after_steps", args.stop_after_steps, force_add=True)
    if args.task == "deadly_corridor":
        OmegaConf.update(cfg, "rl_games.env_eval.deadly.action_layout", "multibinary_7")
    if args.resume:
        step, state = _latest_state(args.output_root / run_id / "checkpoints")
        OmegaConf.update(cfg, "trainer.pretrained_checkpoint", str(state))
        OmegaConf.update(cfg, "trainer.is_resume", True)
        OmegaConf.update(cfg, "trainer.resume_step", step)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=HISTORY_DIMS, required=True)
    parser.add_argument("--model", choices=MODEL_GROUPS, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--vlm-lr", type=float, required=True)
    parser.add_argument("--training-condition", choices=("profile", "zero"), default="profile")
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-project", help="Enable W&B in this project")
    parser.add_argument("--wandb-entity", help="Optional W&B account or team")
    args = parser.parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.base_model_dir = args.base_model_dir.resolve()
    args.output_root = args.output_root.resolve()
    cfg = training_config(args)
    run_dir = args.output_root / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "train.yaml"
    OmegaConf.save(cfg, config_path)
    metadata_path = run_dir / "run_metadata.json"
    wandb_run_id = json.loads(metadata_path.read_text())["wandb_run_id"] if args.resume else cfg.run_id
    if not args.resume:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        (run_dir / "training_code_snapshot.json").write_text(json.dumps({
            "revision": revision, "path": "scripts/tasks/decision_history/train_vla_hist8.py",
        }, indent=2) + "\n")
    metadata = {
        "training_run_id": cfg.run_id, "wandb_run_id": wandb_run_id,
        "recipe_version": "hist8-v2", "history_encoding": "binary_0_1",
        "training_condition": args.training_condition,
        "initialization_seed": 42, "vlm_lr": args.vlm_lr,
        "max_train_steps": cfg.trainer.max_train_steps,
        "stop_after_steps": args.stop_after_steps,
        "num_workers_per_rank": cfg.datasets.vla_data.num_workers,
        "eval_num_workers_per_rank": cfg.datasets.vla_data.eval_num_workers,
        "prefetch_factor": cfg.datasets.vla_data.prefetch_factor,
        "base_model_revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "dataset_dir": str(args.dataset_dir),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    env = os.environ.copy()
    env["WANDB_MODE"] = "online" if args.wandb_project is not None and not args.preflight else "disabled"
    if args.wandb_project is not None and not args.preflight:
        env["WANDB_RUN_ID"] = wandb_run_id
        env["WANDB_RESUME"] = "allow" if args.resume else "never"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(STARVLA), str(ROOT / "third_party/sample-factory")])
    cuda_home = Path(sys.executable).parents[1]
    env["CUDA_HOME"] = str(cuda_home)
    env["CXX"] = str(cuda_home / "bin/x86_64-conda-linux-gnu-g++")
    env["NVCC_PREPEND_FLAGS"] = f"-ccbin={env['CXX']}"
    subprocess.run([
        sys.executable, "-m", "accelerate.commands.launch",
        "--config_file", str(STARVLA / "starVLA/config/deepseeds/deepspeed_zero2.yaml"),
        "--num_processes", "2", "--main_process_port", str(args.port),
        str(STARVLA / "starVLA/training/train_starvla.py"),
        "--config_yaml", str(config_path),
    ], cwd=STARVLA, env=env, check=True)


if __name__ == "__main__":
    main()

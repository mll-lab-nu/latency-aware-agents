"""Stage commands for the APPO H1 GR00T experiments."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from latency_bench.core.config import load_config
from scripts.starvla.model_config import make_model_config

ROOT = Path(__file__).resolve().parents[2]
STARVLA = ROOT / "third_party/starVLA"
BACKBONE = {"repo_id": "Qwen/Qwen3-VL-4B-Instruct",
            "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17"}
GATES = {"air_raid": 2500, "inverted_pendulum": 900, "ant": 3000,
         "half_cheetah": 3000, "hopper": 3000, "humanoid": 5000, "walker2d": 3000}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def teacher_config(task: str, run_root: Path, profile: Path | None, source: Path | None = None) -> dict:
    cfg = load_config(source if source is not None else
                      ROOT / f"configs/examples/gymnasium/{task}/small_model_train.yaml")
    if task != "air_raid":
        cfg["env"].update(env_fps=5, obs_fps=5)
        cfg["latency"] = {"method": "zero", "add_latency_info": False}
    condition = "zero_latency" if profile is None else "profile_latency"
    recipe = "appo_h1"
    cfg["experiment"]["name"] = f"{task}_{condition}_{recipe}"
    cfg["backend"]["train_dir"] = str(run_root / "teacher/checkpoints")
    cfg["logging"]["output_dir"] = str(run_root / "teacher/training")
    if profile is not None:
        cfg["latency"] = {"method": "iid", "profile_path": str(profile),
                          "profile_worker_slot": 0, "seed": 0, "add_latency_info": False}
    return cfg


def teacher_evaluation_config(config_path: Path, checkpoint: Path, output: Path,
                              *, profile: Path, episodes: int) -> dict:
    cfg = load_config(config_path)
    cfg["backend"]["run_mode"] = "eval"
    cfg["experiment"]["seed"] = 42
    cfg["executor"] = {"mode": "simulated", "simulated_worker_capacity": 1,
                       "simulated_inference_pool": True, "inference_devices": ["cuda:0"]}
    cfg["latency"] = {"method": "iid", "profile_path": str(profile), "profile_worker_slot": 0,
                      "seed": 0, "add_latency_info": False}
    cfg["policy"] = {"type": "gymnasium_sf", "checkpoint_path": str(checkpoint),
                     "device": "cuda:0", "deterministic": True}
    cfg["evaluation"].update(eval_episodes=episodes, eval_parallel_envs=1, eval_latency_values=None)
    if cfg["env"]["task_name"] == "air_raid":
        cfg["env"]["make_kwargs"].update(noop_max=30, fire_reset=True)
        cfg["evaluation"]["eval_max_steps"] = 3600
    cfg["logging"] = {"output_dir": str(output), "video": {"enabled": False, "num_bins": 1},
                      "save_step_records": True, "save_action_records": True, "save_latency_records": True}
    return cfg


def vla_config(task: str, dataset: Path, backbone: Path, output: Path,
               *, num_gpus: int, micro_batch: int, condition: str, wandb_entity: str) -> dict:
    cfg = yaml.safe_load((STARVLA / "examples/rl_games/config/base/common.yaml").read_text())
    contract = json.loads((dataset / "manifest.json").read_text())
    horizon = 1
    state_dim = contract["state_dim"] if contract["uses_state"] else 0
    steps = 2000 if task == "inverted_pendulum" else 5000
    action_type = "continuous" if "action_space" in contract["gymnasium_task"] else "discrete"
    run_id = f"{task}_{condition}_gr00t_h{horizon}"
    mixture = output / "mixture.json"
    write_json(mixture, {"train": [[str(dataset), 1.0, contract["robot_type"]]],
                         "validation": [[str(dataset.with_name(dataset.name + "__val")), 1.0, contract["robot_type"]]]})
    cfg.update(run_id=run_id, run_root_dir=str(output / "training"), seed=42, is_debug=False,
               version_id="0.21", wandb_project="latency-sensitive-bench", wandb_entity=wandb_entity,
               wandb_name=run_id, wandb_group="gr00t-seven-env", wandb_tags=[task, condition, "GR00T", f"h{horizon}"],
               training_latency_condition=condition,
               backbone=BACKBONE, base_model={"repo_id": BACKBONE["repo_id"]})
    cfg["framework"] = make_model_config("qwengr00t", state_dim=state_dim,
                                         action_dim=contract["action_dim"], backbone_path=str(backbone), action_horizon=horizon)
    cfg["framework"]["qwenvl"]["attn_implementation"] = "flash_attention_2"
    cfg["datasets"]["vla_data"].update(
        lerobot_version="v2.0", include_state=contract["uses_state"], data_root_dir="/", data_mix="train",
        eval_data_mix="validation", custom_mixtures_path=str(mixture), sequential_step_sampling=True,
        per_device_batch_size=micro_batch, num_workers=4, eval_num_workers=2, prefetch_factor=2,
        obs_image_size=[224, 224], video_backend="pyav", action_type=action_type)
    cfg["datasets"]["vla_data"].update(dataset_py="lerobot_datasets",
        gymnasium_task_contract=contract["gymnasium_task"], active_action_dim=contract["active_action_dim"])
    cfg["framework"]["action_model"]["action_env_dim"] = contract["active_action_dim"]
    cfg["trainer"].update(pretrained_checkpoint=None)
    cfg["trainer"].update(max_train_steps=steps, save_interval=500, eval_interval=500,
                         eval_num_batches=200, gradient_accumulation_steps=64 // (num_gpus * micro_batch))
    cfg["checkpoint"] = {"save_best_model": False, "save_final_model": True, "save_pt_file": True,
                         "save_training_state": True, "local": {"keep_last_n": 1}}
    return cfg


def export_bundle(task: str, trained: Path, checkpoint: Path, dataset: Path, destination: Path,
                   *, source_manifest: Path):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "checkpoints").mkdir(exist_ok=True)
    shutil.copyfile(checkpoint, destination / "checkpoints/model.pt")
    config = yaml.safe_load((trained / "config.full.yaml").read_text())
    inference = {key: deepcopy(config[key]) for key in ("framework", "version_id")}
    inference["framework"]["qwenvl"]["base_vlm"] = "${oc.env:GR00T_BACKBONE_DIR}"
    inference["framework"]["action_model"]["task_objective"] = None
    inference["datasets"] = {"vla_data": {key: config["datasets"]["vla_data"][key]
        for key in ("include_state", "obs_image_size", "image_mode", "stitch_grid", "num_obs_frames")}}
    inference["trainer"] = {"pretrained_checkpoint": None,
                            "profile_timing": deepcopy(config["trainer"]["profile_timing"])}
    inference["trainer"]["profile_timing"]["enabled"] = False
    inference["backbone"] = BACKBONE
    manifest = json.loads((dataset / "manifest.json").read_text())
    inference["datasets"]["vla_data"]["gymnasium_task_contract"] = manifest["gymnasium_task"]
    inference["rl_games"] = {"model_alias": "gr00t", "task": "gymnasium", "action_carrier": "native",
        "gymnasium": {"task_contract": manifest["gymnasium_task"]},
        "env_eval": {"image_size": 224, "frameskip": 1, "image_transform": "raw_rgb", "prompt_mode": "raw"}}
    shutil.copyfile(dataset / "latency_prompt_map.json", destination / "latency_prompt_map.json")
    (destination / "config.yaml").write_text(yaml.safe_dump(inference, sort_keys=False))
    # Keep the complete recipe portable without changing its architecture or budget.
    config["framework"]["qwenvl"]["base_vlm"] = "${oc.env:GR00T_BACKBONE_DIR}"
    (destination / "config.full.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copyfile(trained / "dataset_statistics.json", destination / "dataset_statistics.json")
    shutil.copyfile(dataset / "manifest.json", destination / "manifest.json")
    write_json(destination / "provenance.json", {"task": task, "backbone": BACKBONE,
        "checkpoint": checkpoint.name, "training_steps": config["trainer"]["max_train_steps"],
        "seed": config["seed"], "global_batch": 64, "training_run_id": config["run_id"],
        "condition": config["training_latency_condition"],
        "source": json.loads(source_manifest.read_text()),
        "training_config_sha256": file_sha256(trained / "config.full.yaml"),
        "dataset_manifest_sha256": file_sha256(dataset / "manifest.json")})
    checksums = {path.relative_to(destination).as_posix(): file_sha256(path)
                 for path in sorted(destination.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"}
    (destination / "SHA256SUMS").write_text("".join(f"{value}  {path}\n" for path, value in checksums.items()))
    return destination


def evaluation_config(task: str, bundle: Path, backbone: Path, output: Path, *, mode: str,
                      profile: Path | None, episodes: int, profiling: bool, prompt_key: int) -> dict:
    recipe = "starvla_eval.yaml" if task == "air_raid" else "small_model_train.yaml"
    cfg = load_config(ROOT / f"configs/examples/gymnasium/{task}/{recipe}")
    cfg["env"]["obs_resize"] = [224, 224]
    if task != "air_raid":
        from latency_bench.policy.starvla import _canonical_gymnasium_contract_namespace

        contract = _canonical_gymnasium_contract_namespace(
            json.loads((bundle / "manifest.json").read_text())["gymnasium_task"]
        )
        for key in ("env_id", "registration_imports", "make_kwargs", "base_prompt", "frame_stack", "state_space"):
            cfg["env"][key] = deepcopy(contract[key])
        cfg["env"].update(env_fps=5, obs_fps=5)
        if "action_space" in contract:
            cfg["env"].update(action_space=deepcopy(contract["action_space"]), noop_action=contract["noop_action"])
        else:
            cfg["env"].update(action_order=contract["action_labels"],
                action_map=dict(zip(contract["action_labels"], contract["action_values"])),
                noop_action=contract["action_labels"][contract["noop_action_id"]])
    cfg["experiment"] = {"name": f"gr00t_{task}_{mode}", "seed": 42}
    cfg["backend"]["run_mode"] = "eval"
    cfg["executor"] = {"mode": "realtime" if mode == "measured" else "simulated",
        "inference_devices": ["cuda:0"], "inference_batch_size": 1}
    if mode != "measured":
        cfg["executor"].update(simulated_inference_pool=True, simulated_worker_capacity=1)
    cfg["latency"] = {"method": mode, "sync_cuda": mode == "measured"}
    if mode == "iid":
        cfg["latency"].update(profile_path=str(profile.resolve()), profile_worker_slot=0, seed=0)
    cfg["policy"] = {"type": "starvla", "checkpoint_path": str(bundle / "checkpoints/model.pt"),
        "model_config_path": str(bundle / "config.yaml"), "backbone_path": str(backbone), "device": "cuda:0"}
    cfg["policy"].update(task_manifest_path=str(bundle / "manifest.json"),
        latency_prompt_map_path=str(bundle / "latency_prompt_map.json"), latency_prompt_key=prompt_key)
    horizon = (3000 if profiling else 3600) if task == "air_raid" else (900 if profiling else 1000)
    cfg["evaluation"] = {"eval_episodes": episodes, "eval_parallel_envs": 1, "eval_max_steps": horizon,
        "eval_deterministic": True, "eval_latency_values": None}
    cfg["logging"] = {"output_dir": str(output), "video": {"enabled": False, "num_bins": 1},
        "save_step_records": True, "save_action_records": True, "save_latency_records": True,
        "realtime_pipeline_profile": profiling}
    return cfg


def profile_spec(task: str, config_path: Path, bundle: Path, output: Path) -> dict:
    from latency_bench.cross_device.spec import load_profile_spec

    spec = load_profile_spec(ROOT / "configs/profiles.yaml", "gr00t", "flappy")
    spec.update(model_id="gr00t", workload_id=task, base_config=str(config_path), artifact_root=str(output),
        checkpoint_path=str(bundle / "checkpoints/model.pt"), model_config_path=str(bundle / "config.yaml"),
        checkpoint_kind="latest", session_count=5, eval_episodes=10, evaluation_seed=42,
        calibrate_env_fps=False, simulation_parallel_envs=1, simulation_gpu=None, simulation_env={})
    spec["model_artifact"] = {"local_root": str(bundle), "checkpoint": "checkpoints/model.pt",
        "model_config": "config.yaml", "hf_repo": "latency-sensitive-bench/Standard-Pipeline",
        "hf_path": f"{task}/zero_latency/GR00T"}
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    teacher = commands.add_parser("teacher-config")
    teacher.add_argument("--task", choices=tuple(GATES), required=True)
    teacher.add_argument("--run-root", type=Path, required=True)
    teacher.add_argument("--profile", type=Path)
    teacher.add_argument("--source", type=Path, help="Pinned APPO source YAML; defaults to the original task recipe")
    teacher_eval = commands.add_parser("teacher-eval-config")
    for name in ("config", "checkpoint", "profile", "output"):
        teacher_eval.add_argument(f"--{name}", type=Path, required=True)
    teacher_eval.add_argument("--episodes", type=int, default=10)
    vla = commands.add_parser("vla-config")
    vla.add_argument("--task", choices=tuple(GATES), required=True)
    vla.add_argument("--dataset", type=Path, required=True)
    vla.add_argument("--backbone", type=Path, required=True)
    vla.add_argument("--output", type=Path, required=True)
    vla.add_argument("--num-gpus", type=int, choices=(1, 2, 4), required=True)
    vla.add_argument("--micro-batch", type=int, choices=(8, 16, 32), required=True)
    vla.add_argument("--condition", choices=("zero_latency", "profile_latency"), required=True)
    vla.add_argument("--wandb-entity", required=True)
    bundle = commands.add_parser("bundle")
    bundle.add_argument("--task", choices=tuple(GATES), required=True)
    for name in ("trained", "checkpoint", "dataset", "output", "source-manifest"):
        bundle.add_argument(f"--{name}", type=Path, required=True)
    evaluate = commands.add_parser("eval-config")
    evaluate.add_argument("--task", choices=tuple(GATES), required=True)
    for name in ("bundle", "backbone", "output"):
        evaluate.add_argument(f"--{name}", type=Path, required=True)
    evaluate.add_argument("--mode", choices=("zero", "iid", "measured"), required=True)
    evaluate.add_argument("--profile", type=Path)
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--profiling", action="store_true")
    evaluate.add_argument("--prompt-key", type=int, default=0)
    profile = commands.add_parser("profile")
    profile.add_argument("--task", choices=tuple(GATES), required=True)
    for name in ("config", "bundle", "output"):
        profile.add_argument(f"--{name}", type=Path, required=True)
    profile.add_argument("--resume")
    args = parser.parse_args()
    if args.command == "teacher-config":
        cfg = teacher_config(args.task, args.run_root.resolve(), args.profile, args.source)
        args.run_root.mkdir(parents=True, exist_ok=True)
        path = args.run_root / "teacher.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(path)
    elif args.command == "teacher-eval-config":
        cfg = teacher_evaluation_config(args.config.resolve(), args.checkpoint.resolve(), args.output.resolve(),
                                         profile=args.profile.resolve(), episodes=args.episodes)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "eval.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(args.output / "eval.yaml")
    elif args.command == "vla-config":
        cfg = vla_config(args.task, args.dataset.resolve(), args.backbone.resolve(), args.output.resolve(),
                         num_gpus=args.num_gpus, micro_batch=args.micro_batch, condition=args.condition,
                         wandb_entity=args.wandb_entity)
        path = args.output / "training.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(path)
    elif args.command == "bundle":
        print(export_bundle(args.task, args.trained.resolve(), args.checkpoint.resolve(),
                            args.dataset.resolve(), args.output.resolve(), source_manifest=args.source_manifest.resolve()))
    elif args.command == "eval-config":
        cfg = evaluation_config(args.task, args.bundle.resolve(), args.backbone.resolve(), args.output.resolve(),
            mode=args.mode, profile=args.profile, episodes=args.episodes, profiling=args.profiling, prompt_key=args.prompt_key)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "eval.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(args.output / "eval.yaml")
    elif args.command == "profile":
        from latency_bench.cross_device.runner import run_profile

        spec = profile_spec(args.task, args.config.resolve(), args.bundle.resolve(), args.output.resolve())
        print(json.dumps(run_profile(spec, resume_run_id=args.resume, publish=False)))


if __name__ == "__main__":
    main()

"""Train, independently evaluate and publish one CPU H8 profile-aware teacher."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
from latency_bench.policy.model_source import set_model_files
import shutil
import subprocess
import sys

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import torch
import yaml

from latency_bench.core.config import load_config
from latency_bench.utils.io import write_json_atomic
from scripts.tasks.atari_hist8.publish_zero import REPO_ID
from training.common.checkpoints import find_checkpoint


ROOT = Path(__file__).resolve().parents[3]


def teacher_configs(task: str, model: str, profile: Path, output: Path) -> tuple[dict, dict]:
    """Keep the zero teacher's learning recipe and separate GPU evaluation overrides."""
    run_id = f"{task}-{model}-hist8-profile-cpu15m-s0-v1"
    train = load_config(ROOT / f"configs/experiments/{task}/sf_zero_latency_hist8_cpu15m.yaml")
    train["experiment"]["name"] = run_id
    train["backend"]["train_dir"] = str(output / "teachers")
    train["executor"]["simulated_worker_capacity"] = 1
    train["latency"] = {"method": "temporal", "profile_path": str(profile),
                        "profile_worker_slot": 0, "seed": 0, "add_latency_info": False}
    train["logging"].update(output_dir=str(output / "teacher_logs" / run_id),
                            wandb_project="latency-sensitive-bench", wandb_entity="saberrr-zju",
                            wandb_name=run_id, wandb_tags=["atari-hist8-v1", "profile", model])
    evaluation = deepcopy(train)
    evaluation["experiment"]["seed"] = 1_000_000
    evaluation["backend"]["run_mode"] = "eval"
    evaluation["env"].update(simulator="gpu", gpu_env_device="cuda:0", noop_max=30)
    evaluation["latency"]["seed"] = 271828
    evaluation["policy"] = {"type": "sample_factory", "checkpoint_kind": "best", "deterministic": True, "source": "local", "root": str(Path(train["backend"]["train_dir"]) / run_id), "checkpoint": None}
    evaluation["executor"]["inference_devices"] = ['cuda:0']
    evaluation['evaluation'].update(episodes=100, max_env_steps=7200, eval_raw_reward=True, episode_seed_offset=0, suites=[])
    evaluation["executor"]["parallel_envs"] = 32
    evaluation["executor"]["env_backend"] = f"{task}_gpu_batched"
    evaluation["logging"].update(output_dir=str(output / "teacher_evals" / run_id),
                                 save_step_records=True, save_action_records=True, save_latency_records=True)
    return train, evaluation


def prepare(receipt: dict, output: Path) -> Path:
    receipt = {key: receipt[key] for key in (
        "task", "model", "model_publication", "profile", "profile_repo", "profile_revision", "profile_path")}
    receipt["model_publication"] = {key: receipt["model_publication"][key] for key in (
        "task", "model", "repo_id", "path", "revision")}
    task, model = receipt["task"], receipt["model"]
    snapshot = Path(snapshot_download(receipt["profile_repo"], repo_type="dataset",
                                      revision=receipt["profile_revision"],
                                      allow_patterns=[str(Path(receipt["profile_path"]).parent / "*")]))
    profile = snapshot / receipt["profile_path"]
    directory = output / "teacher_configs" / f"{task}-{model}"
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(profile.parent, directory / "profile", dirs_exist_ok=True)
    train, evaluation = teacher_configs(task, model, directory / "profile/profile.json", output)
    for name, config in (("train", train), ("eval", evaluation)):
        (directory / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    zero = json.loads((ROOT / "configs/experiments/atari_hist8/zero_teachers.json").read_text())
    source = Path(hf_hub_download(zero["repo_id"], zero["teachers"][task]["path"] + "/config.json",
                                  revision=zero["revision"]))
    shutil.copyfile(source, directory / "zero_teacher_config.json")
    write_json_atomic(directory / "provenance.json", {
        "task": task, "model": model, "run_id": train["experiment"]["name"],
        "initialization": "scratch", "zero_teacher": zero["teachers"][task],
        "zero_teacher_revision": zero["revision"], "profile_publication": receipt,
    })
    return directory


def evaluate(directory: Path) -> None:
    config = load_config(directory / "eval.yaml")
    checkpoint = find_checkpoint(config["backend"]["train_dir"], config["experiment"]["name"], kind="best")
    set_model_files(config, checkpoint=checkpoint)
    (directory / "eval.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    subprocess.run([sys.executable, "-m", "latency_bench", "eval", "--config", str(directory / "eval.yaml")],
                   cwd=ROOT, check=True)
    evidence = Path(config["logging"]["output_dir"])
    records = [json.loads(line) for line in (evidence / "episode_metrics.jsonl").read_text().splitlines()]
    counts = Counter()
    with (evidence / "actions.jsonl").open() as stream:
        for line in stream:
            counts[json.dumps(json.loads(line)["action"]["value"])] += 1
    write_json_atomic(evidence / "action_audit.json", {"episodes": len(records),
                                                      "episode_seeds": sorted(row["metadata"]["episode_seed"] for row in records),
                                                      "action_counts": dict(counts),
                                                      "issued_actions": sum(counts.values())})
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    write_json_atomic(evidence / "checkpoint.json", {"checkpoint": str(checkpoint),
                      "training_step": state["env_steps"]})


def publish(directory: Path, output: Path) -> dict:
    provenance = json.loads((directory / "provenance.json").read_text())
    run_id, task, model = provenance["run_id"], provenance["task"], provenance["model"]
    evidence = output / "teacher_evals" / run_id
    identity = json.loads((evidence / "checkpoint.json").read_text())
    checkpoint = Path(identity["checkpoint"])
    train = load_config(directory / "train.yaml")
    latest = find_checkpoint(train["backend"]["train_dir"], run_id)
    final_state = torch.load(latest, map_location="cpu", weights_only=False)
    if final_state["env_steps"] < train["training"]["train_for_env_steps"]:
        raise ValueError("Formal teacher publication requires completed training")
    summary = json.loads((evidence / "queue_eval_latency_profile_sample.json").read_text())
    audit = json.loads((evidence / "action_audit.json").read_text())
    evaluation = load_config(directory / "eval.yaml")
    expected_seeds = list(range(evaluation["experiment"]["seed"],
                                evaluation["experiment"]["seed"] + evaluation["evaluation"]["episodes"]))
    if audit["episode_seeds"] != expected_seeds:
        raise ValueError("Formal teacher publication requires the configured evaluation seeds")
    destination = output / "teacher_publication_stage" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, destination / checkpoint.name)
    shutil.copyfile(checkpoint.parent.parent / "config.json", destination / "config.json")
    for name in ("train.yaml", "eval.yaml", "zero_teacher_config.json"):
        shutil.copyfile(directory / name, destination / name)
    shutil.copytree(directory / "profile", destination / "profile", dirs_exist_ok=True)
    for name in ("episode_metrics.jsonl", "action_audit.json", "checkpoint.json"):
        shutil.copyfile(evidence / name, destination / name)
    write_json_atomic(destination / "eval_profile.json", summary)
    profile = provenance["profile_publication"]
    provenance.update(checkpoint=checkpoint.name,
                      best_checkpoint_env_steps=identity["training_step"],
                      final_env_steps=final_state["env_steps"], mean_return=summary["mean_return"],
                      episodes=audit["episodes"], episode_seeds=audit["episode_seeds"],
                      training_status="complete", evaluation_status="complete")
    write_json_atomic(destination / "provenance.json", provenance)
    (destination / "README.md").write_text(
        f"# {task.title()} H8 CPU teacher for {model} RTX 3090 temporal profile\n\n"
        f"CPU/ALE training: {final_state['env_steps']} raw frames ({provenance['training_status']}), "
        "seed 0, four-frame input, eight decision histories, "
        "constant LR 0.000303; GPU network updates, worker slot 0 and capacity 1.\n\n"
        f"Best checkpoint at {identity['training_step']} raw frames. "
        f"GPU evaluation: {audit['episodes']} episodes, mean raw return {summary['mean_return']}, "
        f"seeds {audit['episode_seeds']}, parallel 32, 60/15 Hz, 7200 raw-frame cap.\n\n"
        f"[W&B](https://wandb.ai/saberrr-zju/latency-sensitive-bench/runs/{run_id}). "
        "Profile identity and both training/evaluation configs are included.\n")
    path = f"latency-aware/{task}/small-policy/sample-factory-{model}-hist8-cpu15m-s0-v1"
    api = HfApi(token=os.environ["HF_TOKEN"])
    commit = api.upload_folder(repo_id=REPO_ID, folder_path=destination, path_in_repo=path,
                                commit_message=f"Publish {task} {model} profile-aware H8 teacher")
    teacher = {"path": path, "checkpoint": checkpoint.name,
               "profile_path": profile["profile_path"]}
    catalog = {"revision": commit.oid, "profile_revision": profile["profile_revision"],
               "teachers": {task: {model: teacher}}}
    write_json_atomic(output / "teacher_publications" / f"{task}-{model}.json", catalog)
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "eval", "publish"), required=True)
    args = parser.parse_args()
    receipt = json.loads(args.profile_receipt.read_text())
    output = args.output_root.resolve()
    directory = output / "teacher_configs" / f"{receipt['task']}-{receipt['model']}"
    if args.stage == "prepare":
        prepare(receipt, output)
    elif args.stage == "train":
        subprocess.run([sys.executable, "-m", "training.train", "--config", str(directory / "train.yaml")],
                       cwd=ROOT, check=True)
    elif args.stage == "eval":
        evaluate(directory)
    else:
        print(json.dumps(publish(directory, output), indent=2))


if __name__ == "__main__":
    main()

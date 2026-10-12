"""Publish each Atari H8 VLA, then attach its completed condition evaluations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi
import yaml

from latency_bench.utils.io import write_json_atomic
from scripts.tasks.decision_history.eval_vla_hist8 import audit_evaluation


MODELS = ("qwenoft", "qwengr00t", "qwenpi_v3")
REPO_ID = "MLL-Lab/LAGEN-models"


def publish(task: str, model: str, output: Path, stage: str,
            training_condition: str = "zero", eval_condition: str = "zero") -> dict:
    run_id = f"{task}-{model}-hist8-{training_condition}-s42-v1"
    bundle = output / "bundles" / run_id
    evaluation = output / "evals" / run_id / f"{training_condition}_to_{eval_condition}"
    provenance = json.loads((bundle / "provenance.json").read_text())
    destination = output / "publication_stage" / task / "vla" / f"starvla-{model}-h1" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    subtree = "zero-latency" if training_condition == "zero" else "latency-aware"
    path = f"{subtree}/{task}/vla/starvla-{model}-h1/{run_id}"
    training_config = yaml.safe_load((bundle / "config.full.yaml").read_text())
    if provenance["training_step"] < training_config["trainer"]["max_train_steps"]:
        raise ValueError("Formal publication requires completed training")
    status = {"training_status": "complete", f"{eval_condition}_evaluation_status": "pending"}
    eval_text = f"- Formal {eval_condition} evaluation: pending.\n"
    upload_patterns = None
    if stage == "training":
        shutil.copytree(bundle, destination, dirs_exist_ok=True)
    else:
        status = json.loads((destination / "publication_status.json").read_text())
        audit = audit_evaluation(evaluation)
        identity = json.loads((evaluation / "provenance.json").read_text())
        if (identity["training_run_id"] != provenance["training_run_id"]
                or identity["training_step"] != provenance["training_step"]
                or Path(identity["checkpoint"]).resolve() != (bundle / "checkpoints/model.pt").resolve()):
            raise ValueError("Evaluation belongs to a different model bundle")
        requested = identity["episodes"]
        if audit["episode_seeds"] != list(range(1_000_000, 1_000_000 + requested)):
            raise ValueError("Formal publication requires the configured evaluation seeds")
        label = "0" if eval_condition == "zero" else "profile_sample"
        summary = json.loads((evaluation / f"queue_eval_latency_{label}.json").read_text())
        evidence_dir = "evaluation" if eval_condition == "zero" else "evaluation_profile"
        for name in ("episode_metrics.jsonl", "action_audit.json", "eval_config.yaml", "provenance.json"):
            target = destination / evidence_dir / name
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(evaluation / name, target)
        summary_name = "eval_l0.json" if eval_condition == "zero" else "eval_profile.json"
        write_json_atomic(destination / summary_name, summary)
        status.update({f"{eval_condition}_evaluation_status": "complete",
                       f"{eval_condition}_mean_return": summary["mean_return"],
                       "mean_return": summary["mean_return"], "episodes": audit["episodes"]})
        eval_text = (
            f"- Evaluation: {audit['episodes']} episodes, seeds {audit['episode_seeds']}, GPU simulator, "
            f"parallel 32, capacity 1, 60/15 Hz, 7200 raw-frame cap, {eval_condition} latency\n"
            f"- Mean raw return: {summary['mean_return']}\n"
        )
        upload_patterns = ["README.md", "publication_status.json", summary_name, evidence_dir + "/**"]
        if eval_condition == "profile":
            profile_receipt = output / "profile_receipts" / f"{task}-{model}.json"
            shutil.copyfile(profile_receipt, destination / "profile_publication.json")
            upload_patterns.append("profile_publication.json")
    write_json_atomic(destination / "publication_status.json", status)
    (destination / "README.md").write_text(
        f"# {task.title()} {model} {training_condition}-trained H8 VLA\n\n"
        "Current RGB and eight binary decision-history windows; action horizon one. "
        "Initialized from the pinned Qwen3-VL-4B backbone and random action head.\n\n"
        f"- Training: {provenance['training_step']} steps, seed 42, global batch 128\n"
        f"- W&B: https://wandb.ai/saberrr-zju/starvla_tasks/runs/{provenance['wandb_run_id']}\n"
        + eval_text, encoding="utf-8",
    )
    api = HfApi(token=os.environ["HF_TOKEN"])
    commit = api.upload_folder(repo_id=REPO_ID, repo_type="model", folder_path=destination,
                               path_in_repo=path, allow_patterns=upload_patterns,
                               commit_message=f"Publish {task} {model} H8 {stage} artifacts")
    receipt = {"repo_id": REPO_ID, "revision": commit.oid, "task": task, "model": model,
               "path": path, "run_id": run_id,
               "publication_status": "published", **status}
    suffix = "profile-evaluation" if stage == "evaluation" and eval_condition == "profile" else stage
    write_json_atomic(output / "publications" / f"{run_id}-{suffix}.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("atlantis", "asterix"), required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=("training", "evaluation"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-condition", choices=("zero", "profile"), default="zero")
    parser.add_argument("--eval-condition", choices=("zero", "profile"), default="zero")
    args = parser.parse_args()
    print(json.dumps(publish(args.task, args.model, args.output_root.resolve(), args.stage,
                             args.training_condition, args.eval_condition), indent=2))


if __name__ == "__main__":
    main()

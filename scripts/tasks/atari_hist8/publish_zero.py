"""Publish each Atari H8 VLA, then attach its completed condition evaluations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi, hf_hub_download

from latency_bench.utils.io import write_json_atomic
from scripts.tasks.decision_history.eval_vla_hist8 import audit_evaluation, file_sha256
from scripts.tasks.decision_history.make_hist8_v2_queue import MODELS


REPO_ID = "latency-sensitive-bench/benchmark-models"


def publish(task: str, model: str, output: Path, stage: str,
            training_condition: str = "zero", eval_condition: str = "zero") -> dict:
    run_id = f"{task}-{model}-hist8-{training_condition}-s42-v1"
    bundle = output / "bundles" / run_id
    evaluation = output / "evals" / run_id / f"{training_condition}_to_{eval_condition}"
    provenance = json.loads((bundle / "provenance.json").read_text())
    assert provenance["training_step"] == 5000
    assert provenance["checkpoint_sha256"] == file_sha256(bundle / "checkpoints/model.pt")
    destination = output / "publication_stage" / task / "vla" / f"starvla-{model}-h1" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    subtree = "zero-latency" if training_condition == "zero" else "latency-aware"
    path = f"{subtree}/{task}/vla/starvla-{model}-h1/{run_id}"
    status = {"training_status": "complete", f"{eval_condition}_evaluation_status": "pending"}
    readback = ["config.yaml", "manifest.json", "provenance.json", "publication_status.json", "README.md"]
    eval_text = f"- Formal {eval_condition} evaluation: pending.\n"
    upload_patterns = None
    if stage == "training":
        shutil.copytree(bundle, destination, dirs_exist_ok=True)
        readback.append("checkpoints/model.pt")
    else:
        status = json.loads((destination / "publication_status.json").read_text())
        training = json.loads((output / "publications" / f"{run_id}-training.json").read_text())
        assert training["checkpoint_sha256"] == provenance["checkpoint_sha256"]
        audit_evaluation(evaluation, 100)
        identity = json.loads((evaluation / "provenance.json").read_text())
        assert identity["checkpoint_sha256"] == provenance["checkpoint_sha256"]
        assert identity["training_step"] == 5000
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
                       "mean_return": summary["mean_return"], "episodes": 100})
        eval_text = (
            "- Evaluation: 100 episodes, seeds 1000000–1000099, GPU simulator, "
            f"parallel 32, capacity 1, 60/15 Hz, 7200 raw-frame cap, {eval_condition} latency\n"
            f"- Mean raw return: {summary['mean_return']}\n"
        )
        upload_patterns = ["README.md", "publication_status.json", summary_name, evidence_dir + "/**"]
        readback.extend([summary_name, evidence_dir + "/episode_metrics.jsonl", evidence_dir + "/action_audit.json"])
        if eval_condition == "profile":
            profile_receipt = output / "profile_receipts" / f"{task}-{model}.json"
            shutil.copyfile(profile_receipt, destination / "profile_publication.json")
            upload_patterns.append("profile_publication.json")
            readback.append("profile_publication.json")
    write_json_atomic(destination / "publication_status.json", status)
    (destination / "README.md").write_text(
        f"# {task.title()} {model} {training_condition}-trained H8 VLA\n\n"
        "Current RGB and eight binary decision-history windows; action horizon one. "
        "Initialized from the pinned Qwen3-VL-4B backbone and random action head.\n\n"
        "- Training: 5000 steps, seed 42, global batch 128\n"
        f"- W&B: https://wandb.ai/saberrr-zju/starvla_tasks/runs/{provenance['wandb_run_id']}\n"
        + eval_text, encoding="utf-8",
    )
    api = HfApi(token=os.environ["HF_TOKEN"])
    commit = api.upload_folder(repo_id=REPO_ID, repo_type="model", folder_path=destination,
                               path_in_repo=path, allow_patterns=upload_patterns,
                               commit_message=f"Publish {task} {model} H8 {stage} artifacts")
    for name in readback:
        remote = hf_hub_download(REPO_ID, path + "/" + name,
                                 revision=commit.oid, token=os.environ["HF_TOKEN"])
        assert file_sha256(Path(remote)) == file_sha256(destination / name)
    receipt = {"repo_id": REPO_ID, "revision": commit.oid, "task": task, "model": model,
               "path": path, "run_id": run_id, "checkpoint_sha256": provenance["checkpoint_sha256"],
               "publication_status": "readback_verified", **status}
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

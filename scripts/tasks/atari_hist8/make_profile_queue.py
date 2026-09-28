"""Build the H100 teacher graph or a game's H200 profile-VLA graph."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from latency_bench.utils.io import write_json_atomic
from scripts.tasks.decision_history.make_hist8_v2_queue import ENVIRONMENTS, MODELS


def build_teacher_manifest(root: Path, python_root: Path, gpu_pool: list[int]) -> dict:
    output = root / "outputs/tasks/atari_hist8_v1"
    python = str(python_root / ENVIRONMENTS["qwenoft"] / "bin/python")
    jobs = []
    for task in ("atlantis", "asterix"):
        for model in MODELS:
            identity = f"{task}-{model}"
            receipt = str(output / "profile_receipts" / f"{identity}.json")
            previous = []
            for stage, gpus, priority in (("prepare", 0, 0), ("train", 1, 10), ("eval", 1, 1), ("publish", 0, 0)):
                job_id = f"teacher_{stage}__{identity}"
                job = {"id": job_id, "kind": "atari_teacher", "python": python,
                       "profile_receipt": receipt, "stage": stage, "output_root": str(output),
                       "runtime_root": str(root), "gpus_required": gpus, "gpu_pool": gpu_pool,
                       "priority": priority, "depends_on": previous}
                if stage == "prepare":
                    job["ready_path"] = receipt
                jobs.append(job)
                previous = [job_id]
    return {"version": "atari-hist8-v1-teachers", "runtime_root": str(root), "jobs": jobs}


def build_manifest(task: str, root: Path, python_root: Path, backbone: Path, gpu_pool: list[int]) -> dict:
    output = root / "outputs/tasks/atari_hist8_v1"
    python = str(python_root / ENVIRONMENTS["qwenoft"] / "bin/python")
    jobs = []

    def add(job: dict, gpus: int, priority: int, dependencies: list[str]) -> None:
        jobs.append(job | {"runtime_root": str(root), "gpus_required": gpus, "gpu_pool": gpu_pool,
                           "priority": priority, "depends_on": dependencies})

    for index, model in enumerate(MODELS):
        identity = f"{task}-{model}"
        receipt = output / "profile_receipts" / f"{identity}.json"
        profile = output / "teacher_configs" / identity / "profile/profile.json"
        prepare_id = f"teacher_prepare__{identity}"
        add({"id": prepare_id, "kind": "atari_teacher", "python": python,
             "profile_receipt": str(receipt), "stage": "prepare", "output_root": str(output),
             "ready_path": str(receipt)}, 0, 0, [])
        data = output / "data_profile" / task / model
        collect_id = f"collect_profile__{identity}"
        add({"id": collect_id, "kind": "collect_profile", "python": python,
             "task": task, "model": model, "output_root": str(output / "data_profile"),
             "teacher_catalog": str(output / "teacher_publications" / f"{identity}.json"),
             "ready_path": str(output / "teacher_publications" / f"{identity}.json")},
            1, 30 + index, [prepare_id])
        run_id = f"{identity}-hist8-profile-s42-v1"
        train_id = f"train__{run_id}"
        common = {"task": task, "model": model, "source": "atari-hist8-v1", "backbone": str(backbone),
                  "dataset_dir": str(data / "lerobot"), "collection_provenance": str(data / "provenance.json")}
        add(common | {"id": train_id, "kind": "train", "run_id": run_id,
                      "python": str(python_root / ENVIRONMENTS[model] / "bin/python"),
                      "train_root": str(output / "train"), "vlm_lr": 3e-6, "training_condition": "profile",
                      "resume": False, "stop_after_steps": None, "port": 30200 + index},
            2, 100 + index, [collect_id])
        package_id = f"package__{run_id}"
        add(common | {"id": package_id, "kind": "package_checkpoint", "python": python,
                      "run_dir": str(output / "train" / run_id), "step": 5000,
                      "bundle_dir": str(output / "bundles" / run_id),
                      "wandb_run_id": hashlib.sha256(run_id.encode()).hexdigest()[:8]},
            0, 0, [train_id])
        publication = {"kind": "publish_vla", "python": python, "task": task, "model": model,
                       "output_root": str(output), "env_condition": "profile"}
        training_publish = f"publish_training__{run_id}"
        add(publication | {"id": training_publish, "stage": "training", "training_condition": "profile"},
            0, 0, [package_id])
        for condition in ("zero", "profile"):
            eval_run = f"{identity}-hist8-{condition}-s42-v1"
            eval_id = f"eval_{condition}_to_profile__{eval_run}"
            add(common | {"id": eval_id, "kind": "eval_ready", "python": python,
                          "bundle_dir": str(output / "bundles" / eval_run),
                          "profile_path": str(profile), "env_condition": "profile", "parallel_envs": 32,
                          "metric_prefix": f"eval_atari_hist8_v1/{condition}_to_profile",
                          "eval_dir": str(output / "evals" / eval_run / f"{condition}_to_profile")},
                2, 200 + index, [f"teacher_prepare__{identity}"] if condition == "zero" else [package_id])
            dependencies = [eval_id] + ([training_publish] if condition == "profile" else [])
            add(publication | {"id": f"publish_profile_evaluation__{eval_run}", "stage": "evaluation",
                               "training_condition": condition}, 0, 0, dependencies)
    return {"version": "atari-hist8-v1-profile", "runtime_root": str(root), "task": task, "jobs": jobs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("teachers", "vla"), required=True)
    parser.add_argument("--task", choices=("atlantis", "asterix"))
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--backbone", type=Path)
    parser.add_argument("--gpu-pool", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, python_root = args.runtime_root.resolve(), args.python_root.resolve()
    manifest = (build_teacher_manifest(root, python_root, args.gpu_pool) if args.stage == "teachers" else
                build_manifest(args.task, root, python_root, args.backbone.resolve(), args.gpu_pool))
    write_json_atomic(args.output, manifest)


if __name__ == "__main__":
    main()

"""Build per-model training, publication, and zero-latency evaluation jobs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from latency_bench.utils.io import write_json_atomic
from scripts.tasks.decision_history.make_hist8_v2_queue import ENVIRONMENTS, MODELS


def build_manifest(task: str, root: Path, python_root: Path, backbone: Path, gpu_pool: list[int]) -> dict:
    output = root / "outputs/tasks/atari_hist8_v1"
    data = output / "data_zero" / task
    eval_python = str(python_root / ENVIRONMENTS["qwenoft"] / "bin/python")
    jobs = []

    def add(job: dict, required: int, priority: int, depends: list[str]) -> None:
        jobs.append(job | {"runtime_root": str(root), "gpus_required": required,
                          "priority": priority, "depends_on": depends,
                          "gpu_pool": gpu_pool})

    rollout = f"collect_zero__{task}"
    convert = f"convert_zero__{task}"
    for stage, job_id, cost, dependencies in (("rollout", rollout, 1, []), ("convert", convert, 0, [rollout])):
        add({"id": job_id, "kind": "collect_zero", "task": task, "stage": stage,
             "python": eval_python, "output_root": str(data.parent), "mirror_host": None,
             "teacher_catalog": str(root / "configs/experiments/atari_hist8/zero_teachers.json")},
            cost, 0, dependencies)
    for index, model in enumerate(MODELS):
        run_id = f"{task}-{model}-hist8-zero-s42-v1"
        train_id = f"train__{run_id}"
        common = {"task": task, "model": model, "source": "atari-hist8-v1",
                  "backbone": str(backbone), "dataset_dir": str(data / "lerobot"),
                  "collection_provenance": str(data / "provenance.json")}
        add(common | {"id": train_id, "kind": "train", "run_id": run_id,
                      "python": str(python_root / ENVIRONMENTS[model] / "bin/python"),
                      "train_root": str(output / "train"), "vlm_lr": 3e-6,
                      "training_condition": "zero", "resume": False, "stop_after_steps": None,
                      "port": 30100 + index,
                      "ready_path": str(output / "smoke" / f"{task}.complete.json")},
            2, 100 + index, [convert])
        package_id = f"package__{run_id}"
        publish_id = f"publish_training__{run_id}"
        eval_id = f"eval_zero_to_zero__{run_id}"
        bundle = str(output / "bundles" / run_id)
        add(common | {"id": package_id, "kind": "package_checkpoint", "python": eval_python,
                      "run_dir": str(output / "train" / run_id), "step": 5000,
                      "bundle_dir": bundle,
                      "wandb_run_id": hashlib.sha256(run_id.encode()).hexdigest()[:8]},
            0, 10, [train_id])
        publication = {"kind": "publish_vla", "python": eval_python, "task": task,
                       "model": model, "output_root": str(output)}
        add(publication | {"id": publish_id, "stage": "training"}, 0, 20, [package_id])
        add(common | {"id": eval_id, "kind": "eval_ready",
                      "python": eval_python, "profile_path": str(output / "profiles_pending" / model),
                      "env_condition": "zero", "parallel_envs": 32,
                      "metric_prefix": "eval_atari_hist8_v1/zero_to_zero",
                      "run_dir": str(output / "train" / run_id), "step": 5000,
                      "bundle_dir": bundle,
                      "eval_dir": str(output / "evals" / run_id / "zero_to_zero"),
                      "wandb_run_id": hashlib.sha256(run_id.encode()).hexdigest()[:8]},
            2, 1000 + index, [package_id])
        add(publication | {"id": f"publish_evaluation__{run_id}", "stage": "evaluation"},
            0, 30, [eval_id, publish_id])
    return {"version": "atari-hist8-v1-zero", "runtime_root": str(root), "task": task, "jobs": jobs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("atlantis", "asterix"), required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--gpu-pool", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_atomic(args.output, build_manifest(args.task, args.runtime_root.resolve(),
                      args.python_root.resolve(), args.backbone.resolve(), args.gpu_pool))


if __name__ == "__main__":
    main()

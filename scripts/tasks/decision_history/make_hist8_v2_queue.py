"""Build a host-local job graph from the audited old-run catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from latency_bench.utils.io import write_json_atomic


H200_FIRST = "34.131.223.74"
H200_SECOND = "34.131.91.104"
H100 = "34.6.91.253"
TASKS = ("demon_attack", "flappy", "deadly_corridor")
MODELS = ("qwenoft", "qwengr00t", "qwenpi_v3")
ENVIRONMENTS = {"qwenoft": "starvla_rl_games_openvla", "qwengr00t": "starvla_rl_games_gr00t",
                "qwenpi_v3": "starvla_rl_games_pi05"}


def build_manifest(catalog: list[dict], *, host: str, runtime_root: Path, python_root: Path,
                   old_root: Path, backbone: Path) -> dict:
    output = runtime_root / "outputs/tasks/decision_history_vla_hist8_v2"
    eval_python = str(python_root / "starvla_rl_games_openvla/bin/python")
    entries = {(entry["task"], entry["model"]): entry for entry in catalog}
    jobs = []

    def add(job, *, required, priority, depends=()):
        jobs.append({**job, "runtime_root": str(runtime_root), "gpus_required": required,
                     "priority": priority, "depends_on": list(depends)})

    for task_index, task in enumerate(TASKS):
        for model_index, model in enumerate(MODELS):
            entry = entries[task, model]
            common = {"task": task, "model": model, "backbone": str(backbone)}
            profile = str(old_root / Path(entry["profile_path"]).relative_to(entry["old_root"]))
            if entry["host"] == host:
                base_run_id = f"{task}-{model}-hist8-scratch-s42-v2"
                train_base = {
                    **common, "kind": "train", "source": "v2", "resume": False,
                    "training_condition": "profile",
                    "python": str(python_root / ENVIRONMENTS[model] / "bin/python"),
                    "dataset_dir": entry["dataset_dir"], "collection_provenance": entry["collection_provenance"],
                    "train_root": str(output / "train"),
                }
                evaluation = {
                    **common, "source": "v2", "python": eval_python, "profile_path": profile,
                    "env_condition": "profile", "parallel_envs": 32,
                    "metric_prefix": "eval_hist8_v4/profile_to_profile",
                    "eval_dir": str(output / "evals" / base_run_id),
                }
                priority = 100 + task_index * 100 + model_index * 10
                train_id = f"train__{base_run_id}"
                add({**train_base, "id": train_id, "run_id": base_run_id, "vlm_lr": 3e-6,
                     "stop_after_steps": None, "port": 29800 + task_index * 10 + model_index},
                    required=2, priority=priority)
                add({**evaluation, "id": f"eval_v2__{base_run_id}", "kind": "eval_checkpoint",
                     "run_dir": str(output / "train" / base_run_id), "step": 5000,
                     "dataset_dir": entry["dataset_dir"], "collection_provenance": entry["collection_provenance"],
                     "bundle_dir": str(output / "bundles" / base_run_id),
                     "wandb_run_id": hashlib.sha256(base_run_id.encode()).hexdigest()[:8]},
                    required=2, priority=1000 + task_index, depends=[train_id])

            legacy_host = entry["host"]
            if task == "demon_attack":
                legacy_host = H200_FIRST if model == "qwenoft" else H100
            if legacy_host != host:
                continue
            legacy = {
                **common, "source": "legacy", "python": eval_python,
                "env_condition": "profile", "parallel_envs": 32,
                "metric_prefix": "eval_history01_v2/sf_render_v4" if task == "deadly_corridor" else "eval_history01_v2",
                "id": f"eval_legacy__{task}-{model}", "profile_path": profile,
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "bundle_dir": str(output / "bundles" / f"legacy-{task}-{model}-step{entry['step']}"),
                "eval_dir": str(output / "evals_legacy" / f"{task}-{model}-step{entry['step']}"),
            }
            if task == "demon_attack":
                legacy.update(kind="eval_existing", source_bundle_dir=str(old_root / "outputs/bundles" / task / model))
            else:
                legacy.update(kind="eval_checkpoint", run_dir=entry["run_dir"], step=entry["step"],
                              dataset_dir=entry["dataset_dir"], collection_provenance=entry["collection_provenance"],
                              wandb_run_id=entry["wandb_run_id"])
            add(legacy, required=2, priority=2000 + task_index * 10 + model_index)
    return {"version": "hist8-v2", "host": host, "runtime_root": str(runtime_root), "jobs": jobs}


def add_zero_controls(manifest: dict, catalog: list[dict], *, python_root: Path,
                      old_root: Path, backbone: Path, teacher_catalog: Path) -> None:
    """Append three shared datasets and nine zero-trained controls to the live graph.

    H200 #1 collects on its unpaired GPU 7 and materializes on CPU; Demon data
    is mirrored to H200 #2 before that host's training-ready marker is written.
    """
    host = manifest["host"]
    root = Path(manifest["runtime_root"])
    output = root / "outputs/tasks/decision_history_vla_hist8_v2"
    data_root = output / "data_zero"
    eval_python = str(python_root / "starvla_rl_games_openvla/bin/python")
    entries = {(item["task"], item["model"]): item for item in catalog}
    if host == H200_FIRST:
        for index, task in enumerate(TASKS):
            rollout_id = f"collect_zero__{task}"
            convert_id = f"materialize_zero__{task}"
            for stage, job_id, dependencies in (("rollout", rollout_id, []), ("convert", convert_id, [rollout_id])):
                gpu = stage == "rollout" and task != "deadly_corridor"
                manifest["jobs"].append({
                    "id": job_id, "kind": "collect_zero", "task": task, "stage": stage,
                    "runtime_root": str(root),
                    "python": eval_python, "teacher_catalog": str(teacher_catalog),
                    "output_root": str(data_root), "mirror_host": H200_SECOND if task == "demon_attack" else None,
                    "gpus_required": 1 if gpu else 0, "gpu_pool": [7] if gpu else [],
                    "priority": 50 + index if gpu else 0, "depends_on": dependencies,
                })
    for task_index, task in enumerate(TASKS):
        owner = H200_SECOND if task == "demon_attack" else H200_FIRST
        if host != owner:
            continue
        for model_index, model in enumerate(MODELS):
            entry = entries[task, model]
            run_id = f"{task}-{model}-hist8-zero-scratch-s42-v2"
            train_id = f"train_zero__{run_id}"
            dataset = data_root / task
            profile = old_root / Path(entry["profile_path"]).relative_to(entry["old_root"])
            manifest["jobs"].append({
                "id": train_id, "kind": "train", "source": "v2", "task": task, "model": model,
                "runtime_root": str(root),
                "training_condition": "zero", "resume": False, "run_id": run_id, "vlm_lr": 3e-6,
                "python": str(python_root / ENVIRONMENTS[model] / "bin/python"), "backbone": str(backbone),
                "dataset_dir": str(dataset / "lerobot"), "collection_provenance": str(dataset / "provenance.json"),
                "ready_path": str(dataset / "collection.complete.json"),
                "train_root": str(output / "train"), "stop_after_steps": None,
                "port": 29900 + task_index * 10 + model_index, "gpus_required": 2,
                "priority": 400 + task_index * 10 + model_index,
                "depends_on": [f"materialize_zero__{task}"] if host == H200_FIRST else [],
            })
            package_id = f"package_zero__{run_id}"
            manifest["jobs"].append({
                "id": package_id, "kind": "package_checkpoint", "source": "v2", "python": eval_python,
                "runtime_root": str(root),
                "run_dir": str(output / "train" / run_id), "step": 5000,
                "dataset_dir": str(dataset / "lerobot"), "collection_provenance": str(dataset / "provenance.json"),
                "bundle_dir": str(output / "bundles" / run_id),
                "wandb_run_id": hashlib.sha256(run_id.encode()).hexdigest()[:8],
                "gpus_required": 0, "priority": 0, "depends_on": [train_id],
            })
            for condition in ("zero", "profile"):
                manifest["jobs"].append({
                    "id": f"eval_zero_to_{condition}__{run_id}", "kind": "eval_ready", "source": "v2",
                    "runtime_root": str(root),
                    "task": task, "model": model, "python": eval_python, "backbone": str(backbone),
                    "bundle_dir": str(output / "bundles" / run_id), "profile_path": str(profile),
                    "env_condition": condition, "parallel_envs": 32,
                    "metric_prefix": f"eval_hist8_v4/zero_to_{condition}",
                    "eval_dir": str(output / "evals" / run_id / f"to_{condition}"),
                    "gpus_required": 2, "priority": 1100 + task_index * 10 + model_index,
                    "depends_on": [package_id],
                    **({"aux_gpu": 7} if host == H200_FIRST else {}),
                })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(json.loads(args.catalog.read_text()), host=args.host, runtime_root=args.runtime_root,
                              python_root=args.python_root, old_root=args.old_root, backbone=args.backbone)
    write_json_atomic(args.output, manifest)


if __name__ == "__main__":
    main()

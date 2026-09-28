"""Publish completed MIKASA H8 inference bundles to the benchmark model repo."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi

from scripts.starvla.artifacts import MODEL_REPO_ID, model_path


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "outputs/starvla/runs/mikasa_intercept_grab_fast"
TASK = "mikasa_intercept_grab_fast"
HELDOUT_FIRST = 4_242_424_542
HELDOUT_LAST = 4_242_424_741
CORE_FILES = ("config.yaml", "dataset_statistics.json", "task_contract.json")


def _publication_path(condition: str, model: str, run_id: str, phase: str) -> str:
    section = {
        "zero-latency": "zero-latency",
        "profile-latency": "latency-aware",
        "fixed-2": "latency-aware",
    }[condition]
    if phase == "dagger" and condition != "zero-latency":
        run_id += "-" + {"profile-latency": "profile", "fixed-2": "fixed-2"}[condition]
    return model_path(TASK, model, run_id, condition=section, action_horizon=8)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stage_weight(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.samefile(source):
        return
    target.unlink(missing_ok=True)
    os.link(source, target)


def _source_and_run(condition: str, model: str) -> tuple[Path, Path]:
    if condition == "zero-latency":
        return (RUNS / "h8-zero-latency" / model,
                RUNS / "h8-naive-dagger-20260923" / model)
    arm = Path(condition) / model
    return (RUNS / "h8-conditioned-seed-reset-20260922" / arm,
            RUNS / "h8-conditioned-dagger-20260923" / arm)


def _complete_dagger(run: Path, condition: str) -> dict:
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if result["dagger_steps"] != 5000:
        raise RuntimeError(f"DAgger steps incomplete: {run}")
    evaluations = sorted((run / "evaluations").glob("step_*/result.json"))
    if len(evaluations) != 11 or any(
        (value := json.loads(path.read_text(encoding="utf-8")))["episodes"] != 100
        or not value["audit"]["passed"] for path in evaluations
    ):
        raise RuntimeError(f"DAgger development evaluation incomplete: {run}")
    deltas = sorted((run / "deltas").glob("round-*/raw/collection.json"))
    if len(deltas) != 10 or any(
        json.loads(path.read_text(encoding="utf-8"))["splits"]["train"]["valid_actions"] < 16_000
        for path in deltas
    ) or len(list((run / "deltas").glob("round-*/lerobot/task_contract.json"))) != 10:
        raise RuntimeError(f"DAgger collection incomplete: {run}")
    checkpoint = run / "training/dagger/checkpoints/steps_5000_pytorch_model.pt"
    state = run / "training/dagger/checkpoints/steps_5000_state"
    state_files = [state / "latest", state / "pytorch_model/mp_rank_00_model_states.pt"]
    state_files.extend(state / f"random_states_{rank}.pkl" for rank in range(2))
    state_files.extend(state / "pytorch_model" /
                       f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
                       for rank in range(2))
    if not checkpoint.is_file() or any(not path.is_file() or path.stat().st_size == 0
                                       for path in state_files):
        raise RuntimeError(f"DAgger trainer state incomplete: {run}")
    heldout = result["heldout"]
    groups = heldout.items() if condition == "zero-latency" else ((None, heldout),)
    for setting, group in groups:
        for name, evaluation in group.items():
            if (evaluation["episodes"] != 200 or evaluation["seed"] != HELDOUT_FIRST
                    or not evaluation["audit"]["passed"]):
                raise RuntimeError(f"DAgger heldout evaluation incomplete: {run}")
            directory = run / "heldout" / setting / name if setting else run / "heldout" / name
            seeds = [json.loads(line)["metadata"]["episode_seed"] for line in
                     (directory / "episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
            if (len(seeds) != 200 or len(set(seeds)) != 200
                    or min(seeds) != HELDOUT_FIRST or max(seeds) != HELDOUT_LAST):
                raise RuntimeError(f"DAgger heldout reset seeds mismatch: {directory}")
    return result


def _prepare_stage(condition: str, model: str, phase: str) -> tuple[Path, Path, str, int]:
    source, dagger = _source_and_run(condition, model)
    if phase == "sft":
        if condition == "zero-latency":
            raise ValueError("The naive SFT bundles are already published")
        run = source
        bundle = source / "inference"
        record = json.loads((source / "run.json").read_text(encoding="utf-8"))
        run_id = record["run_id"]
        step = 5000
        training_weight = source / "training/bootstrap/checkpoints/steps_5000_pytorch_model.pt"
        if not bundle.joinpath("checkpoints/model.pt").samefile(training_weight):
            raise RuntimeError(f"SFT inference bundle differs from training weights: {source}")
    else:
        run = dagger
        result = _complete_dagger(dagger, condition)
        bundle = dagger / "bundles/best"
        run_id = ("h8-naive-dagger-20260923" if condition == "zero-latency"
                  else "h8-conditioned-dagger-20260923")
        step = result["best_step"]
    stage = run / "hf_publication" / phase
    stage.mkdir(parents=True, exist_ok=True)
    _stage_weight(bundle / "checkpoints/model.pt", stage / "checkpoints/model.pt")
    for name in CORE_FILES:
        shutil.copyfile(bundle / name, stage / name)
    if phase == "sft":
        for name in ("provenance.json", "README.md"):
            shutil.copyfile(bundle / name, stage / name)
    else:
        if step != 5000:
            _stage_weight(dagger / "bundles/latest/checkpoints/model.pt",
                          stage / "checkpoints/latest_model.pt")
        _write_json(stage / "eval_matrix.json", result["heldout"])
        if condition == "zero-latency":
            best = {"zero": "eval_l0.json", "profile": "eval_profile.json",
                    "fixed-2": "eval_fixed_2f.json"}
            for setting, name in best.items():
                _write_json(stage / name, result["heldout"][setting]["best"])
        else:
            name = "eval_profile.json" if condition == "profile-latency" else "eval_fixed_2f.json"
            _write_json(stage / name, result["heldout"]["best"])
        _write_json(stage / "provenance.json", {
            "task": TASK, "model": model, "condition": condition, "phase": "dagger",
            "source_sft_checkpoint": result["source_checkpoint"],
            "selected_step": step, "latest_step": 5000,
            "dagger_optimizer_steps": result["dagger_steps"],
            "development_seeds": [4_242_424_442, 4_242_424_541],
            "heldout_seeds": [HELDOUT_FIRST, HELDOUT_LAST],
            "checkpoint": "checkpoints/model.pt",
            "latest_checkpoint": ("checkpoints/latest_model.pt" if step != 5000
                                  else "checkpoints/model.pt"),
        })
        (stage / "README.md").write_text(
            f"# MIKASA H8 {condition} {model} DAgger\n\n"
            f"Selected development checkpoint: step {step}. "
            "The independent 200-episode results are in eval_matrix.json.\n",
            encoding="utf-8",
        )
    path = _publication_path(condition, model, run_id, phase)
    return stage, run, path, step


def _verify(api: HfApi, stage: Path, path: str, revision: str) -> None:
    files = [file for file in stage.rglob("*") if file.is_file()]
    paths = [f"{path}/{file.relative_to(stage).as_posix()}" for file in files]
    remote = {item.path: item.size for item in api.get_paths_info(
        MODEL_REPO_ID, paths, revision=revision, repo_type="model"
    )}
    for file, target in zip(files, paths, strict=True):
        if remote[target] != file.stat().st_size:
            raise RuntimeError(f"Hub artifact size mismatch: {target}")


def _publish(api: HfApi, condition: str, model: str, phase: str, *, dry_run: bool) -> dict:
    stage, run, path, step = _prepare_stage(condition, model, phase)
    marker = run / f"publication_hf_{phase}.json"
    files = [file.relative_to(stage).as_posix() for file in stage.rglob("*") if file.is_file()]
    if dry_run:
        return {"path": path, "stage": str(stage), "files": files, "step": step}
    if marker.is_file():
        published = json.loads(marker.read_text(encoding="utf-8"))
        if published["path"] != path:
            # Existing publication markers record the pre-migration path and revision.
            revision = api.repo_info(MODEL_REPO_ID, repo_type="model").sha
            _verify(api, stage, path, revision)
            return {**published, "path": path, "revision": revision}
        _verify(api, stage, path, published["revision"])
        return published
    commit = api.upload_folder(
        repo_id=MODEL_REPO_ID, repo_type="model", folder_path=stage,
        path_in_repo=path, commit_message=f"Publish MIKASA H8 {phase} {condition} {model}",
    )
    _verify(api, stage, path, commit.oid)
    published = {"repo_id": MODEL_REPO_ID, "path": path, "revision": commit.oid,
                 "phase": phase, "selected_step": step, "files": files}
    _write_json(marker, published)
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        help="condition/model; repeat to publish several arms")
    parser.add_argument("--phase", choices=("sft", "dagger", "both"), default="both")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    api = HfApi(endpoint="https://huggingface.co", token=os.environ["HF_TOKEN"])
    for arm in args.arm:
        condition, model = arm.split("/", 1)
        lock = RUNS / "hf_publication.lock"
        with lock.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            phases = ("sft", "dagger") if args.phase == "both" else (args.phase,)
            for phase in phases:
                print(json.dumps(_publish(api, condition, model, phase, dry_run=args.dry_run)),
                      flush=True)


if __name__ == "__main__":
    main()

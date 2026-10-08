"""Stage commits and pinned replay inputs for the held-out sim2real study."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

from latency_bench.utils.io import read_json, write_json_atomic


DATASET_REPO = "MLL-Lab/LAGEN-datasets"
PROFILE_REPO = "MLL-Lab/LAGEN-profiles"
LOCAL_ONLY_FILES = {"contract.json", "hardware.json", "archived_eval.yaml",
                    "generated.yaml", "resolved_config.yaml", "eval_config.yaml"}


class StudyPublication:
    """Publish measurements before experiment stages and retain dataset snapshot receipts."""

    def __init__(self, output: Path, source_ref: str, *, defer_uploads=False):
        self.output = output
        self.source_ref = source_ref
        self.api = HfApi()
        self.defer_uploads = defer_uploads

    def publish(self, stage: str, paths: list[Path]) -> str | None:
        """Return the published revision, or queue a durable stage and return None."""
        receipt = self.output / "uploads" / f"{stage}.json"
        if receipt.exists():
            return read_json(receipt)["revision"]
        if self.defer_uploads:
            pending = self.output / "pending_uploads" / f"{stage}.json"
            if not pending.exists():
                write_json_atomic(pending, {
                    "source_ref": self.source_ref, "stage": stage, "queued_at": time.time(),
                    "files": [file.relative_to(self.output).as_posix() for file in _publication_files(paths)],
                })
                print(f"QUEUED={stage}", flush=True)
            return None
        record = {
            "source_ref": self.source_ref, "stage": stage,
            "files": [file.relative_to(self.output).as_posix() for file in _publication_files(paths)],
        }
        revision = _publish_stages(self.api, [(self.output, record)])
        write_json_atomic(receipt, {
            "repo_id": DATASET_REPO, "source_ref": self.source_ref,
            "stage": stage, "revision": revision,
        })
        print(f"UPLOADED={stage} HF_COMMIT={revision}", flush=True)
        return revision


def _publication_files(paths):
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(file for file in sorted(path.rglob("*"))
                         if not file.is_dir() and file.name not in LOCAL_ONLY_FILES)
        else:
            files.append(path)
    return files


def _measurement_file(name):
    path = Path(name)
    return path.parts[0] in {"calibration", "profile"} or (
        path.parts[0] in {"real", "sim"} and path.name not in {"episode_metrics.jsonl", "DONE"}
    )


def _publish_stages(api, stages):
    """Commit measurement files first; dataset markers establish stage publication."""
    revision = api.repo_info(DATASET_REPO, repo_type="dataset", expand=["sha"]).sha
    profile_revision = api.repo_info(PROFILE_REPO, repo_type="dataset", expand=["sha"]).sha
    markers = [f"{record['source_ref']}/published/{record['stage']}.json" for _, record in stages]
    existing = {entry.path for entry in api.get_paths_info(
        DATASET_REPO, markers, repo_type="dataset", revision=revision,
    )}
    profile_markers = {entry.path for entry in api.get_paths_info(
        PROFILE_REPO, markers, repo_type="dataset", revision=profile_revision,
    )}
    measurements, experiments = {}, {}
    for (output, record), marker in zip(stages, markers):
        if marker in existing:
            continue
        names = []
        for name in record["files"]:
            remote = f"{record['source_ref']}/{name}"
            if _measurement_file(name):
                names.append(name)
                if marker not in profile_markers:
                    measurements[remote] = CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(output / name))
            else:
                experiments[remote] = CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(output / name))
        if names and marker not in profile_markers:
            measurements[marker] = CommitOperationAdd(
                path_in_repo=marker,
                path_or_fileobj=json.dumps({"stage": record["stage"], "files": names}).encode(),
            )
        experiments[marker] = CommitOperationAdd(
            path_in_repo=marker,
            path_or_fileobj=json.dumps({"stage": record["stage"], "files": record["files"]}).encode(),
        )
    if measurements:
        profile_revision = api.create_commit(
            repo_id=PROFILE_REPO, repo_type="dataset", operations=list(measurements.values()),
            commit_message=f"sim2real measurements: {len(stages)} stages",
        ).oid
    if experiments:
        for _, record in stages:
            marker = f"{record['source_ref']}/published/{record['stage']}.json"
            if marker not in existing:
                path = f"{record['source_ref']}/profile_source.json"
                experiments[path] = CommitOperationAdd(path_in_repo=path, path_or_fileobj=json.dumps({
                    "repo_id": PROFILE_REPO, "revision": profile_revision, "source_ref": record["source_ref"],
                }).encode())
        revision = api.create_commit(
            repo_id=DATASET_REPO, repo_type="dataset", operations=list(experiments.values()),
            commit_message=(f"sim2real {stages[0][1]['source_ref']}: {stages[0][1]['stage']}"
                            if len(stages) == 1 else f"sim2real queued stages: {len(stages)}"),
        ).oid
    return revision


def publish_pending(pending_paths):
    """Commit queued measurements and experiments, then persist upload receipts."""
    stages = [(path, read_json(path)) for path in pending_paths]
    outputs = [next(parent.parent for parent in pending.parents if parent.name == "pending_uploads")
               for pending, _ in stages]
    revision = _publish_stages(HfApi(), list(zip(outputs, [record for _, record in stages])))
    for (pending, record), output in zip(stages, outputs):
        write_json_atomic(output / "uploads" / f"{record['stage']}.json", {
            "repo_id": DATASET_REPO, "source_ref": record["source_ref"],
            "stage": record["stage"], "revision": revision,
        })
        pending.unlink()
    return revision


def download_real_experiment(source_ref: str, revision: str, output: Path) -> dict:
    """Restore portable real results from exactly the caller's HF snapshot."""
    snapshot = Path(snapshot_download(
        repo_id=DATASET_REPO, repo_type="dataset", revision=revision,
        allow_patterns=[f"{source_ref}/{name}" for name in (
            "experiment.json", "eval_template.yaml", "REAL_DONE", "real_summary.json",
            "real_runs.csv", "profile_source.json",
            "real/*/episode_metrics.jsonl",
        )],
    )) / source_ref
    (snapshot / "REAL_DONE").read_bytes()
    shutil.copytree(snapshot, output, dirs_exist_ok=True)
    profile = read_json(snapshot / "profile_source.json")
    measurement = Path(snapshot_download(
        repo_id=profile["repo_id"], repo_type="dataset", revision=profile["revision"],
        allow_patterns=[f"{profile['source_ref']}/{name}" for name in (
            "calibration/calibration.json", "profile/*",
        )],
    )) / profile["source_ref"]
    shutil.copytree(measurement / "calibration", output / "calibration", dirs_exist_ok=True)
    shutil.copytree(measurement / "profile", output / "profile", dirs_exist_ok=True)
    return read_json(output / "experiment.json")

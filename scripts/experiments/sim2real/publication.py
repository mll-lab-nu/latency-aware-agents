"""Stage commits and pinned replay inputs for the held-out sim2real study."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

from latency_bench.utils.io import read_json, write_json_atomic


DATASET_REPO = "latency-sensitive-bench/profiles"
LOCAL_ONLY_FILES = {"contract.json", "hardware.json", "archived_eval.yaml",
                    "generated.yaml", "resolved_config.yaml", "eval_config.yaml"}


class StudyPublication:
    """Publish completed stages atomically and retain their snapshot receipts."""

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
        marker = f"{self.source_ref}/published/{stage}.json"
        revision = self.api.repo_info(DATASET_REPO, repo_type="dataset").sha
        # Recover a successful commit if the process died before saving its receipt.
        if not self.api.file_exists(DATASET_REPO, marker, repo_type="dataset", revision=revision):
            files = _publication_files(paths)
            names = [file.relative_to(self.output).as_posix() for file in files]
            operations = [CommitOperationAdd(
                path_in_repo=f"{self.source_ref}/{name}", path_or_fileobj=str(file),
            ) for name, file in zip(names, files)]
            operations.append(CommitOperationAdd(
                path_in_repo=marker,
                path_or_fileobj=json.dumps({"stage": stage, "files": names}).encode(),
            ))
            commit = self.api.create_commit(
                repo_id=DATASET_REPO, repo_type="dataset", operations=operations,
                commit_message=f"sim2real {self.source_ref}: {stage}",
            )
            revision = commit.oid
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


def publish_pending(pending_paths):
    """Commit completed queued stages together, then persist their real upload receipts."""
    api = HfApi()
    stages = [(path, read_json(path)) for path in pending_paths]
    revision = api.repo_info(DATASET_REPO, repo_type="dataset").sha
    markers = [f"{record['source_ref']}/published/{record['stage']}.json" for _, record in stages]
    existing = {entry.path for entry in api.get_paths_info(
        DATASET_REPO, markers, repo_type="dataset", revision=revision,
    )}
    operations = {}
    for (pending, record), marker in zip(stages, markers):
        if marker in existing:
            continue
        output = next(parent.parent for parent in pending.parents if parent.name == "pending_uploads")
        for name in record["files"]:
            remote = f"{record['source_ref']}/{name}"
            operations[remote] = CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(output / name))
        operations[marker] = CommitOperationAdd(
            path_in_repo=marker,
            path_or_fileobj=json.dumps({"stage": record["stage"], "files": record["files"]}).encode(),
        )
    if operations:
        revision = api.create_commit(
            repo_id=DATASET_REPO, repo_type="dataset", operations=list(operations.values()),
            commit_message=f"sim2real queued stages: {len(stages)}",
        ).oid
    for pending, record in stages:
        output = next(parent.parent for parent in pending.parents if parent.name == "pending_uploads")
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
            "real_runs.csv", "calibration/calibration.json", "profile/*",
            "real/*/episode_metrics.jsonl",
        )],
    )) / source_ref
    (snapshot / "REAL_DONE").read_bytes()
    shutil.copytree(snapshot, output, dirs_exist_ok=True)
    return read_json(output / "experiment.json")

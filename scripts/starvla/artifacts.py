"""Upload the task pipeline's canonical datasets and final inference bundles."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError

from latency_bench.utils.hf import resolve_token


MODEL_REPO_ID = "latency-sensitive-bench/benchmark-models"
DATASET_REPO_ID = "latency-sensitive-bench/benchmark-datasets"
MODEL_FILES = [
    "checkpoints/model.pt", "config.yaml", "dataset_statistics.json",
    "task_contract.json", "eval_l0.json", "provenance.json", "README.md",
]
DATASET_FILES = ["lerobot/**", "lerobot__val/**", "provenance.json", "README.md"]


def _remote_path_exists(api: HfApi, repo_id: str, path: str, repo_type: str, revision: str) -> bool:
    try:
        next(iter(api.list_repo_tree(repo_id, path, revision=revision, repo_type=repo_type)))
    except EntryNotFoundError:
        return False
    return True


def _upload_folder(*, repo_id: str, repo_type: str, source_dir: Path, path: str,
                   allow_patterns: list[str], commit_message: str) -> str:
    api = HfApi(token=resolve_token(""))
    head = api.repo_info(repo_id, repo_type=repo_type).sha
    if _remote_path_exists(api, repo_id, path, repo_type, head):
        raise FileExistsError(f"refusing to replace existing artifact at {repo_id}:{path}")
    commit = api.upload_folder(
        repo_id=repo_id, repo_type=repo_type, folder_path=source_dir,
        path_in_repo=path, allow_patterns=allow_patterns,
        ignore_patterns=["**/.cache/**", "**/__pycache__/**", "**/*.tmp"],
        commit_message=commit_message, parent_commit=head,
    )
    return commit.oid


def dataset_path(task: str, *, model: str | None = None, run_id: str | None = None,
                 round_number: int | None = None) -> str:
    prefix = f"zero-latency/{task.replace('_', '-')}"
    if round_number is None:
        return f"{prefix}/teacher-rollouts-h1-state-v1"
    return f"{prefix}/dagger/starvla-{model}-h1/{run_id}/round-{round_number:02d}"


def model_path(task: str, model: str, run_id: str) -> str:
    return f"zero-latency/{task.replace('_', '-')}/vla/starvla-{model}-h1/{run_id}"


def publish_dataset(source_dir: Path, *, task: str, model: str | None = None,
                    run_id: str | None = None, round_number: int | None = None) -> str:
    """Publish a shared bootstrap or one DAgger delta; return its Hub commit."""
    path = dataset_path(task, model=model, run_id=run_id, round_number=round_number)
    return _upload_folder(
        repo_id=DATASET_REPO_ID, repo_type="dataset", source_dir=source_dir, path=path,
        allow_patterns=DATASET_FILES, commit_message=f"Publish {path}",
    )


def publish_model(source_dir: Path, *, task: str, model: str, run_id: str) -> str:
    """Publish the final inference bundle already prepared by the training pipeline."""
    path = model_path(task, model, run_id)
    return _upload_folder(
        repo_id=MODEL_REPO_ID, repo_type="model", source_dir=source_dir, path=path,
        allow_patterns=MODEL_FILES, commit_message=f"Publish {path}",
    )

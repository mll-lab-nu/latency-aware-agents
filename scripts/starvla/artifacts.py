"""Upload the task pipeline's canonical datasets and final inference bundles."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

from latency_bench.utils.hf import resolve_token


MODEL_REPO_ID = "placeholder/benchmark-models"
DATASET_REPO_ID = "placeholder/benchmark-datasets"
MODEL_FILES = [
    "checkpoints/model.pt", "config.yaml", "dataset_statistics.json",
    "task_contract.json", "eval_l0.json", "eval_profile.json", "provenance.json", "README.md",
]
DATASET_FILES = ["lerobot/**", "lerobot__val/**", "provenance.json", "README.md"]


def _upload_folder(*, repo_id: str, repo_type: str, source_dir: Path, path: str,
                   allow_patterns: list[str], commit_message: str) -> str:
    api = HfApi(token=resolve_token(""))
    commit = api.upload_folder(
        repo_id=repo_id, repo_type=repo_type, folder_path=source_dir,
        path_in_repo=path, allow_patterns=allow_patterns,
        ignore_patterns=["**/.cache/**", "**/__pycache__/**", "**/*.tmp"],
        delete_patterns=["**"], commit_message=commit_message,
    )
    return commit.oid


def dataset_path(task: str, *, model: str | None = None, run_id: str | None = None,
                 round_number: int | None = None, condition: str = "zero-latency", action_horizon: int = 1) -> str:
    prefix = f"{condition}/{task.replace('_', '-')}"
    version = "v2" if task == "humanoidbench_balance_simple" and action_horizon == 8 else "v1"
    if round_number is None:
        return f"{prefix}/teacher-rollouts-h{action_horizon}-state-{version}"
    return f"{prefix}/dagger/starvla-{model}-h{action_horizon}/{run_id}/round-{round_number:02d}"


def model_path(task: str, model: str, run_id: str, *, condition: str = "zero-latency", action_horizon: int = 1) -> str:
    return f"{condition}/{task.replace('_', '-')}/vla/starvla-{model}-h{action_horizon}/{run_id}"


def publish_dataset(source_dir: Path, *, task: str, model: str | None = None,
                    run_id: str | None = None, round_number: int | None = None, condition: str = "zero-latency", action_horizon: int = 1) -> str:
    """Publish a shared bootstrap or one DAgger delta; return its Hub commit."""
    path = dataset_path(task, model=model, run_id=run_id, round_number=round_number, condition=condition, action_horizon=action_horizon)
    return _upload_folder(
        repo_id=DATASET_REPO_ID, repo_type="dataset", source_dir=source_dir, path=path,
        allow_patterns=DATASET_FILES, commit_message=f"Publish {path}",
    )


def publish_model(source_dir: Path, *, task: str, model: str, run_id: str, condition: str = "zero-latency", action_horizon: int = 1) -> str:
    """Publish the final inference bundle already prepared by the training pipeline."""
    path = model_path(task, model, run_id, condition=condition, action_horizon=action_horizon)
    return _upload_folder(
        repo_id=MODEL_REPO_ID, repo_type="model", source_dir=source_dir, path=path,
        allow_patterns=MODEL_FILES, commit_message=f"Publish {path}",
    )

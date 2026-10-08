#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Optional


STEP_FILE_RE = re.compile(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$")


def _str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _split_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _has_files(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return True
    return any(path.iterdir())


def _dataset_ready(dataset_local_dir: Path, required_subdirs: Iterable[str]) -> bool:
    if not dataset_local_dir.exists():
        return False
    for subdir in required_subdirs:
        if not _has_files(dataset_local_dir / subdir):
            return False
    return True


def _ensure_dataset(
    mode: str,
    dataset_local_dir: Path,
    dataset_hf_repo_id: Optional[str],
    dataset_allow_patterns: list[str],
    dataset_required_subdirs: list[str],
    force_download: bool,
) -> dict:
    result = {
        "dataset_mode": mode,
        "dataset_local_dir": str(dataset_local_dir),
        "dataset_downloaded": False,
        "dataset_ready": False,
    }
    if mode == "none":
        return result

    if mode == "local":
        result["dataset_ready"] = _dataset_ready(dataset_local_dir, dataset_required_subdirs)
        return result

    if mode != "hf":
        raise ValueError(f"Unsupported dataset mode: {mode}")

    if not dataset_hf_repo_id:
        raise ValueError("dataset_hf_repo_id is required when dataset_mode=hf")

    ready_before = _dataset_ready(dataset_local_dir, dataset_required_subdirs)
    if ready_before and not force_download:
        result["dataset_ready"] = True
        return result

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for dataset_mode=hf") from exc

    dataset_local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=dataset_hf_repo_id,
        repo_type="dataset",
        local_dir=str(dataset_local_dir),
        allow_patterns=dataset_allow_patterns or None,
    )
    result["dataset_downloaded"] = True
    result["dataset_ready"] = _dataset_ready(dataset_local_dir, dataset_required_subdirs)
    return result


def _find_latest_local_checkpoint(checkpoint_local_dir: Path) -> tuple[Optional[Path], int]:
    if not checkpoint_local_dir.exists():
        return None, 0
    candidates: list[tuple[int, Path]] = []
    for item in checkpoint_local_dir.iterdir():
        if not item.is_file():
            continue
        match = STEP_FILE_RE.match(item.name)
        if not match:
            continue
        candidates.append((int(match.group(1)), item))
    if not candidates:
        return None, 0
    candidates.sort(key=lambda x: x[0])
    step, path = candidates[-1]
    return path, step


def _download_latest_hf_checkpoint(checkpoint_hf_repo_id: str, checkpoint_local_dir: Path) -> tuple[Optional[Path], int]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for checkpoint_mode=hf") from exc

    api = HfApi()
    files = api.list_repo_files(repo_id=checkpoint_hf_repo_id, repo_type="model")
    candidates: list[tuple[int, str]] = []
    for file_path in files:
        file_name = os.path.basename(file_path)
        match = STEP_FILE_RE.match(file_name)
        if not match:
            continue
        candidates.append((int(match.group(1)), file_path))

    if not candidates:
        return None, 0

    candidates.sort(key=lambda x: x[0])
    step, chosen_repo_path = candidates[-1]
    checkpoint_local_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=checkpoint_hf_repo_id,
        repo_type="model",
        filename=chosen_repo_path,
        local_dir=str(checkpoint_local_dir),
        local_dir_use_symlinks=False,
    )
    return Path(local_path), step


def _ensure_checkpoint(
    mode: str,
    checkpoint_local_dir: Path,
    checkpoint_hf_repo_id: Optional[str],
) -> dict:
    result = {
        "checkpoint_mode": mode,
        "checkpoint_local_dir": str(checkpoint_local_dir),
        "checkpoint_downloaded": False,
        "resume_checkpoint": None,
        "resume_step": 0,
        "resume_found": False,
    }

    if mode == "none":
        return result

    if mode == "local":
        ckpt_path, step = _find_latest_local_checkpoint(checkpoint_local_dir)
        if ckpt_path is not None:
            result["resume_checkpoint"] = str(ckpt_path)
            result["resume_step"] = int(step)
            result["resume_found"] = True
        return result

    if mode != "hf":
        raise ValueError(f"Unsupported checkpoint mode: {mode}")

    if not checkpoint_hf_repo_id:
        raise ValueError("checkpoint_hf_repo_id is required when checkpoint_mode=hf")

    ckpt_path, step = _download_latest_hf_checkpoint(
        checkpoint_hf_repo_id=checkpoint_hf_repo_id,
        checkpoint_local_dir=checkpoint_local_dir,
    )
    if ckpt_path is not None:
        result["checkpoint_downloaded"] = True
        result["resume_checkpoint"] = str(ckpt_path)
        result["resume_step"] = int(step)
        result["resume_found"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-mode", choices=["none", "local", "hf"], default="none")
    parser.add_argument("--dataset-local-dir", type=str, required=True)
    parser.add_argument("--dataset-hf-repo-id", type=str, default=None)
    parser.add_argument("--dataset-allow-patterns", type=str, default="")
    parser.add_argument("--dataset-required-subdirs", type=str, default="train")
    parser.add_argument("--dataset-force-download", type=str, default="false")

    parser.add_argument("--checkpoint-mode", choices=["none", "local", "hf"], default="none")
    parser.add_argument("--checkpoint-local-dir", type=str, required=True)
    parser.add_argument("--checkpoint-hf-repo-id", type=str, default=None)
    args = parser.parse_args()

    dataset_local_dir = Path(args.dataset_local_dir).expanduser().resolve()
    checkpoint_local_dir = Path(args.checkpoint_local_dir).expanduser().resolve()
    dataset_allow_patterns = _split_csv(args.dataset_allow_patterns)
    dataset_required_subdirs = _split_csv(args.dataset_required_subdirs)
    force_download = _str2bool(args.dataset_force_download)

    dataset_info = _ensure_dataset(
        mode=args.dataset_mode,
        dataset_local_dir=dataset_local_dir,
        dataset_hf_repo_id=args.dataset_hf_repo_id,
        dataset_allow_patterns=dataset_allow_patterns,
        dataset_required_subdirs=dataset_required_subdirs,
        force_download=force_download,
    )
    checkpoint_info = _ensure_checkpoint(
        mode=args.checkpoint_mode,
        checkpoint_local_dir=checkpoint_local_dir,
        checkpoint_hf_repo_id=args.checkpoint_hf_repo_id,
    )

    out = {**dataset_info, **checkpoint_info}
    print(json.dumps(out))


if __name__ == "__main__":
    main()

"""Resolve exact model artifacts and published deployment profiles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from latency_bench.profiling.sync import download_profile
from latency_bench.utils.io import read_json


PROFILE_DONE_RE = re.compile(
    r"^profiles/(?P<model_id>[^/]+)/(?P<gpu_class>[^/]+)/"
    r"(?P<workload_id>[^/]+)/(?P<instance_id>[^/]+)/DONE$"
)


def profile_refs_from_repo_files(files: Iterable[str]) -> list[Path]:
    refs = []
    for filename in files:
        match = PROFILE_DONE_RE.fullmatch(filename)
        if match is not None:
            values = match.groupdict()
            refs.append(
                Path(values["model_id"])
                / values["gpu_class"]
                / values["workload_id"]
                / values["instance_id"]
            )
    return sorted(refs, key=Path.as_posix)


def list_remote_profiles(
    repo_id: str,
    *,
    model_id: str | None = None,
    gpu_class: str | None = None,
    workload_id: str | None = None,
) -> list[Path]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
    )
    refs = profile_refs_from_repo_files(files)
    return [
        ref
        for ref in refs
        if (model_id is None or ref.parts[0] == model_id)
        and (gpu_class is None or ref.parts[1] == gpu_class)
        and (workload_id is None or ref.parts[2] == workload_id)
    ]


def resolve_remote_profile(
    repo_id: str,
    ref: str | Path,
    *,
    output_dir: str | Path,
) -> Path:
    """Download a profile by its published address, without a model registry."""
    ref = Path(ref)
    if ref.is_absolute() or len(ref.parts) != 4 or ".." in ref.parts:
        raise ValueError("Profile reference must be model/deployment/workload/instance")
    local_dir = download_profile(repo_id, ref, local_dir=Path(output_dir) / ref)
    profile_path = local_dir / "profile.json"
    profile = read_json(profile_path)
    expected = dict(
        zip(("model_id", "gpu_class", "workload_id", "instance_id"), ref.parts)
    )
    actual = {key: profile[key] for key in expected}
    if actual != expected:
        raise ValueError(
            f"Remote profile coordinates {actual!r} do not match requested {expected!r}"
        )
    return profile_path

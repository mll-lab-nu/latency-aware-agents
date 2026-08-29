from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable

from latency_bench.cross_device.sync import download_profile
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
    registry: dict[str, Any],
    *,
    hf_commit: str,
    model_id: str | None = None,
    gpu_class: str | None = None,
    workload_id: str | None = None,
) -> list[Path]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(
        repo_id=registry["hf_repo"],
        repo_type="dataset",
        revision=hf_commit,
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
    spec: dict[str, Any],
    *,
    gpu_class: str,
    instance_id: str,
    hf_commit: str,
) -> tuple[Path, Path, str]:
    ref = Path(spec["model_id"]) / gpu_class / spec["workload_id"] / instance_id
    local_dir, resolved_commit = download_profile(
        spec,
        ref,
        hf_commit=hf_commit,
        local_dir=Path(spec["artifact_root"]) / "downloads" / "profiles" / ref,
    )
    profile_path = local_dir / "profile.json"
    profile = read_json(profile_path)
    expected = {
        "model_id": spec["model_id"],
        "gpu_class": gpu_class,
        "workload_id": spec["workload_id"],
        "instance_id": instance_id,
    }
    actual = {key: profile[key] for key in expected}
    if actual != expected:
        raise ValueError(
            f"Remote profile coordinates {actual!r} do not match requested {expected!r}"
        )
    return ref, profile_path, resolved_commit

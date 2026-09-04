from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from latency_bench.cross_device.sync import download_profile
from latency_bench.utils.io import read_json


PROFILE_DONE_RE = re.compile(
    r"^profiles/(?P<model_id>[^/]+)/(?P<gpu_class>[^/]+)/"
    r"(?P<workload_id>[^/]+)/(?P<instance_id>[^/]+)/DONE$"
)


def resolve_model_artifacts(
    spec: dict[str, Any],
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a preset's exact local or Hugging Face model artifacts.

    A caller-provided path replaces only the preset's local model root.  The
    checkpoint relative path and all remote coordinates remain fixed by the
    preset, so a missing local artifact cannot silently select another model.
    """
    artifact = spec.get("model_artifact")
    if artifact is None:
        return spec

    resolved = dict(spec)
    model = _resolve_artifact(
        artifact,
        local_root=Path(model_path) if model_path is not None else None,
    )
    resolved["checkpoint_path"] = str(model["path"])
    resolved["model_provenance"] = model["provenance"]
    if "model_config" in artifact:
        resolved["model_config_path"] = str(
            (model["root"] / artifact["model_config"]).resolve(strict=True)
        )
    if "runtime_artifact" in spec:
        runtime = _resolve_artifact(spec["runtime_artifact"])
        resolved["runtime_checkpoint_path"] = str(runtime["path"])
        resolved["runtime_provenance"] = runtime["provenance"]
    return resolved


def _resolve_artifact(
    artifact: dict[str, Any],
    *,
    local_root: Path | None = None,
) -> dict[str, Any]:
    configured_root = Path(artifact["local_root"])
    root = configured_root if local_root is None else local_root
    checkpoint = Path(artifact["checkpoint"])
    hf_repo = artifact["hf_repo"]
    hf_path = Path(artifact["hf_path"])
    hf_revision = artifact["hf_revision"]

    if root.exists():
        resolved_root = root.resolve(strict=True)
        resolved_path = (resolved_root / checkpoint).resolve(strict=True)
        source = "local"
    else:
        from huggingface_hub import HfApi, snapshot_download

        resolved_revision = HfApi().repo_info(
            repo_id=hf_repo,
            repo_type="model",
            revision=hf_revision,
        ).sha
        snapshot = Path(
            snapshot_download(
                repo_id=hf_repo,
                repo_type="model",
                revision=resolved_revision,
                allow_patterns=[f"{hf_path.as_posix()}/**"],
            )
        )
        resolved_root = (snapshot / hf_path).resolve(strict=True)
        resolved_path = (resolved_root / checkpoint).resolve(strict=True)
        source = "hf"

    return {
        "path": resolved_path,
        "root": resolved_root,
        "provenance": {
            "source": source,
            "repo_id": hf_repo,
            "path_in_repo": hf_path.as_posix(),
            "checkpoint": checkpoint.as_posix(),
            "requested_revision": hf_revision,
            **(
                {"resolved_revision": resolved_revision}
                if source == "hf"
                else {}
            ),
        },
    }


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

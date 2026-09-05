from __future__ import annotations

import os
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
    hf_artifacts: bool = False,
) -> dict[str, Any]:
    """Resolve a preset's exact local or Hugging Face model artifacts.

    A caller-provided path replaces only the preset's local model root.  The
    checkpoint relative path and all remote coordinates remain fixed by the
    preset, so a missing local artifact cannot silently select another model.
    """
    artifact = spec["model_artifact"]

    resolved = dict(spec)
    model = _resolve_artifact(
        artifact,
        local_root=Path(model_path) if model_path is not None else None,
        force_hf=hf_artifacts,
    )
    resolved["checkpoint_path"] = str(model["path"])
    resolved["model_provenance"] = model["provenance"]
    if "model_config" in artifact:
        resolved["model_config_path"] = str(
            _contained_path(
                model["root"],
                artifact["model_config"],
                allow_hf_cache_symlink=model["provenance"]["source"] == "hf",
            )
        )
    if "runtime_artifact" in spec:
        runtime = _resolve_artifact(
            spec["runtime_artifact"], force_hf=hf_artifacts
        )
        resolved["runtime_checkpoint_path"] = str(runtime["path"])
        resolved["runtime_provenance"] = runtime["provenance"]
    return resolved


def resolve_profile_artifacts(
    spec: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the immutable model/runtime revisions recorded by a profile."""
    resolved = dict(spec)
    model_provenance = profile["provenance"]["model_artifact"]
    model = _download_profile_artifact(model_provenance)
    resolved["checkpoint_path"] = str(model["path"])
    resolved["model_provenance"] = model_provenance
    if "model_config" in model_provenance:
        resolved["model_config_path"] = str(
            _contained_path(
                model["root"],
                model_provenance["model_config"],
                allow_hf_cache_symlink=True,
            )
        )
    if "runtime_artifact" in profile["provenance"]:
        runtime_provenance = profile["provenance"]["runtime_artifact"]
        runtime = _download_profile_artifact(runtime_provenance)
        resolved["runtime_checkpoint_path"] = str(runtime["path"])
        resolved["runtime_provenance"] = runtime_provenance
    return resolved


def _download_profile_artifact(provenance: dict[str, Any]) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    # A published profile is external input, not a live registry revision.
    if provenance["source"] != "hf" or not re.fullmatch(
        r"[0-9a-f]{40}", provenance["resolved_revision"]
    ):
        raise ValueError("profile replay requires source='hf' and a fixed 40-hex commit")
    path_in_repo = Path(provenance["path_in_repo"])
    if Path(provenance["checkpoint"]).is_absolute():
        raise ValueError(
            f"Artifact path must be relative: {provenance['checkpoint']}"
        )
    download_kwargs = {
        "repo_id": provenance["repo_id"],
        "repo_type": "model",
        "revision": provenance["resolved_revision"],
    }
    if path_in_repo != Path("."):
        download_kwargs["allow_patterns"] = [f"{path_in_repo.as_posix()}/**"]
    snapshot = Path(snapshot_download(**download_kwargs))
    root = (
        snapshot
        if path_in_repo == Path(".")
        else _contained_path(snapshot, path_in_repo)
    )
    path = _contained_path(
        root,
        provenance["checkpoint"],
        allow_hf_cache_symlink=True,
    )
    return {"root": root, "path": path}


def _contained_path(
    root: Path,
    artifact_path: str | Path,
    *,
    allow_hf_cache_symlink: bool = False,
) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        raise ValueError(f"Artifact path must be relative: {artifact_path}")
    resolved_root = root.resolve(strict=True)
    lexical_root = Path(os.path.abspath(root))
    lexical = Path(os.path.normpath(os.fspath(root / path))).absolute()
    if not lexical.is_relative_to(lexical_root):
        raise ValueError(f"Artifact path escapes the snapshot root: {artifact_path}")
    resolved = lexical.resolve(strict=True)
    if resolved.is_relative_to(resolved_root):
        return resolved
    if allow_hf_cache_symlink and _is_hf_blob_path(resolved_root, lexical, resolved):
        return lexical
    raise ValueError(f"Artifact path escapes the snapshot root: {artifact_path}")


def _is_hf_blob_path(root: Path, lexical: Path, resolved: Path) -> bool:
    if not lexical.is_symlink() or not lexical.is_file():
        return False
    # HF files may link out of snapshots/<commit>, but only into this repo's
    # sibling blobs directory, never a similarly named ancestor directory.
    return any(
        snapshot.parent.name == "snapshots"
        and re.fullmatch(r"[0-9a-f]{40}", snapshot.name)
        and snapshot.parent.parent.name.startswith("models--")
        and snapshot.parent.parent / "blobs" == resolved.parent
        for snapshot in (root, *root.parents)
    )


def _resolve_artifact(
    artifact: dict[str, Any],
    *,
    local_root: Path | None = None,
    force_hf: bool = False,
) -> dict[str, Any]:
    configured_root = Path(artifact["local_root"])
    root = configured_root if local_root is None else local_root
    checkpoint = Path(artifact["checkpoint"])
    hf_repo = artifact["hf_repo"]
    hf_path = Path(artifact["hf_path"])
    hf_revision = artifact["hf_revision"]
    if local_root is not None:
        resolved_root = root.resolve(strict=True)
        source = "local"
        resolved_revision = None
    elif not force_hf and root.exists():
        resolved_root = root.resolve(strict=True)
        source = "local"
        resolved_revision = None
    else:
        if re.fullmatch(r"[0-9a-f]{40}", hf_revision):
            resolved_revision = hf_revision
        else:
            from huggingface_hub import HfApi

            resolved_revision = HfApi().repo_info(
                repo_id=hf_repo,
                repo_type="model",
                revision=hf_revision,
            ).sha
        from huggingface_hub import snapshot_download

        download_kwargs = {
            "repo_id": hf_repo,
            "repo_type": "model",
            "revision": resolved_revision,
        }
        if hf_path != Path("."):
            download_kwargs["allow_patterns"] = [f"{hf_path.as_posix()}/**"]
        snapshot = Path(snapshot_download(**download_kwargs))
        resolved_root = (
            snapshot
            if hf_path == Path(".")
            else _contained_path(snapshot, hf_path)
        )
        source = "hf"

    resolved_path = _contained_path(
        resolved_root,
        checkpoint,
        allow_hf_cache_symlink=source == "hf",
    )
    artifact_provenance = {
        "source": source,
        "repo_id": hf_repo,
        "path_in_repo": hf_path.as_posix(),
        "checkpoint": checkpoint.as_posix(),
        "requested_revision": hf_revision,
    }
    if resolved_revision is not None:
        artifact_provenance["resolved_revision"] = resolved_revision
    if "model_config" in artifact:
        artifact_provenance["model_config"] = artifact["model_config"]

    return {
        "path": resolved_path,
        "root": resolved_root,
        "provenance": artifact_provenance,
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

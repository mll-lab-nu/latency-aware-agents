"""Resolve exact model artifacts and published deployment profiles."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Iterable

from latency_bench.core.config import load_config
from latency_bench.profiling.sync import download_profile
from latency_bench.utils.io import read_json


PROFILE_DONE_RE = re.compile(
    r"^profiles/(?P<model_id>[^/]+)/(?P<gpu_class>[^/]+)/"
    r"(?P<workload_id>[^/]+)/(?P<instance_id>[^/]+)/DONE$"
)


_MODEL_SIDECARS = (
    "config.yaml", "config.full.yaml", "dataset_statistics.json",
    "task_contract.json", "manifest.json", "latency_prompt_map.json",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_files(root: Path) -> list[Path]:
    files = []
    for directory, children, names in os.walk(root, followlinks=True):
        children[:] = [name for name in children if name not in {".cache", ".git", "__pycache__"}]
        files.extend(Path(directory) / name for name in names)
    return sorted(files)


def _artifact_digests(root: Path, artifact: dict[str, Any], *, from_hf: bool) -> dict[str, str]:
    checkpoint = root / artifact["checkpoint"]
    files = _directory_files(checkpoint) if checkpoint.is_dir() else [checkpoint]
    files.extend(root / name for name in _MODEL_SIDECARS if (root / name).is_file())
    if "model_config" in artifact:
        files.append(root / artifact["model_config"])
    return {
        path.relative_to(root).as_posix(): _file_sha256(_contained_path(
            root, path.relative_to(root), allow_hf_cache_symlink=from_hf,
        ))
        for path in sorted(set(files))
    }


def profile_input_identity(spec: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint the actual model bundle and final evaluation input paths."""
    from latency_bench.profiling.runtime_configs import _base_config

    identity = {}
    for role in ("model", "runtime"):
        if role == "runtime" and "runtime_provenance" not in spec:
            continue
        provenance = spec[f"{role}_provenance"]
        identity[role] = _artifact_digests(
            Path(spec[f"{role}_artifact_root"]), provenance,
            from_hf=provenance["source"] == "hf",
        )
    config = _base_config(spec, Path("__latency_bench_contract_output__"))
    for key in ("model_config_path", "task_contract_path", "task_manifest_path", "latency_prompt_map_path"):
        if key in config["policy"]:
            identity[key] = _file_sha256(Path(config["policy"][key]))
    if "backbone_path" in config["policy"]:
        root = Path(config["policy"]["backbone_path"])
        # Resolve a missing directory through the filesystem, not an empty identity.
        root = root.resolve(strict=True)
        identity["backbone"] = {
            path.relative_to(root).as_posix(): _file_sha256(path)
            for path in _directory_files(root)
        }
    return identity


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
    resolved["model_artifact_root"] = str(model["root"])
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
        resolved["runtime_artifact_root"] = str(runtime["root"])
        resolved["runtime_provenance"] = runtime["provenance"]
    _prepare_backbone(resolved)
    return resolved


def resolve_profile_artifacts(
    spec: dict[str, Any],
    profile: dict[str, Any],
    *,
    model_path: str | Path | None = None,
    backbone_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve the model/runtime artifacts recorded by a profile."""
    resolved = dict(spec)
    if "model_artifact" not in profile["provenance"]:
        raise ValueError("Profile has no verified model identity; do not infer it from current HF contents")
    model_provenance = profile["provenance"]["model_artifact"]
    model = _resolve_recorded_artifact(
        model_provenance,
        local_root=Path(model_path) if model_path is not None else (
            Path(spec["model_artifact"]["local_root"])
            if model_provenance["source"] == "local" else None
        ),
    )
    resolved["checkpoint_path"] = str(model["path"])
    resolved["model_artifact_root"] = str(model["root"])
    resolved["model_provenance"] = model_provenance
    if "model_config" in model_provenance:
        resolved["model_config_path"] = str(
            _contained_path(
                model["root"],
                model_provenance["model_config"],
                allow_hf_cache_symlink=model["from_hf"],
            )
        )
    if "runtime_artifact" in profile["provenance"]:
        runtime_provenance = profile["provenance"]["runtime_artifact"]
        runtime = _resolve_recorded_artifact(
            runtime_provenance,
            local_root=Path(spec["runtime_artifact"]["local_root"])
            if runtime_provenance["source"] == "local" else None,
        )
        resolved["runtime_checkpoint_path"] = str(runtime["path"])
        resolved["runtime_artifact_root"] = str(runtime["root"])
        resolved["runtime_provenance"] = runtime_provenance
    if backbone_path is not None:
        resolved["backbone_path"] = str(backbone_path)
    _prepare_backbone(resolved, recorded=(
        profile["provenance"]["backbone_artifact"]
        if "backbone_artifact" in profile["provenance"] else None
    ))
    return resolved


def _prepare_backbone(spec: dict[str, Any], *, recorded: dict[str, Any] | None = None) -> None:
    """Prepare explicit bundle backbone metadata before inference workers start."""
    uses_starvla = False
    if "base_config" in spec:
        policy = load_config(spec["base_config"])["policy"]
        uses_starvla = policy["type"] == "starvla"
        if "backbone_path" not in spec and "backbone_path" in policy:
            spec["backbone_path"] = policy["backbone_path"]
    if "backbone_path" in spec:
        spec["backbone_path"] = str(Path(spec["backbone_path"]).expanduser().resolve())
        spec["backbone_provenance"] = {"source": "local"}
        return
    if recorded is not None:
        if recorded["source"] == "local":
            raise ValueError("This profile used a local backbone; supply --backbone-path for replay")
        backbone = recorded
    else:
        if "model_config_path" not in spec and not uses_starvla:
            return
        # StarVLA constructs the model from config.yaml, even when benchmark
        # preprocessing uses config.full.yaml.
        config_path = Path(spec["model_artifact_root"]) / "config.yaml"
        model_config = load_config(config_path)
        if "backbone" not in model_config:
            if uses_starvla:
                raise ValueError("StarVLA profiling requires --backbone-path or config.yaml backbone metadata to bind the actual model/processor files")
            return
        backbone = model_config["backbone"]
    from huggingface_hub import snapshot_download

    download_args = {"repo_id": backbone["repo_id"]}
    for field in ("revision", "allow_patterns"):
        if field in backbone:
            download_args[field] = backbone[field]
    snapshot = Path(snapshot_download(**download_args))
    spec["backbone_path"] = str(snapshot)
    spec["backbone_provenance"] = {
        "source": "hf", "repo_id": backbone["repo_id"], "revision": snapshot.name,
        **({"allow_patterns": backbone["allow_patterns"]} if "allow_patterns" in backbone else {}),
    }


def _resolve_recorded_artifact(provenance: dict[str, Any], *, local_root: Path | None) -> dict[str, Any]:
    if "files_sha256" not in provenance:
        raise ValueError("Profile has no verified model identity; do not infer it from current HF contents")
    if local_root is not None:
        root = local_root.expanduser().resolve(strict=True)
        model = {"root": root, "path": _contained_path(root, provenance["checkpoint"])}
    else:
        model = _download_profile_artifact(provenance)
    from_hf = local_root is None
    actual = _artifact_digests(model["root"], provenance, from_hf=from_hf)
    if actual != provenance["files_sha256"]:
        raise ValueError("Model artifact identity does not match the checkpoint/config/sidecars measured by this profile")
    return {**model, "from_hf": from_hf}


def _download_profile_artifact(provenance: dict[str, Any]) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    path_in_repo = Path(provenance["path_in_repo"])
    if Path(provenance["checkpoint"]).is_absolute():
        raise ValueError(
            f"Artifact path must be relative: {provenance['checkpoint']}"
        )
    download_kwargs = {
        "repo_id": provenance["repo_id"],
        "repo_type": "model",
    }
    download_kwargs["revision"] = provenance["revision"]
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
    if local_root is not None:
        resolved_root = root.resolve(strict=True)
        source = "local"
    elif not force_hf and root.exists():
        resolved_root = root.resolve(strict=True)
        source = "local"
    else:
        from huggingface_hub import snapshot_download

        hf_repo = artifact["hf_repo"]
        hf_path = Path(artifact["hf_path"])
        download_kwargs = {
            "repo_id": hf_repo,
            "repo_type": "model",
        }
        if "revision" in artifact:
            download_kwargs["revision"] = artifact["revision"]
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
        "checkpoint": checkpoint.as_posix(),
        "files_sha256": _artifact_digests(resolved_root, artifact, from_hf=source == "hf"),
    }
    if "model_config" in artifact:
        artifact_provenance["model_config"] = artifact["model_config"]
    if source == "hf":
        artifact_provenance.update(repo_id=hf_repo, path_in_repo=hf_path.as_posix(), revision=snapshot.name)

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
    model_id: str | None = None,
    gpu_class: str | None = None,
    workload_id: str | None = None,
) -> list[Path]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(
        repo_id=registry["hf_repo"],
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
    spec: dict[str, Any],
    *,
    gpu_class: str,
    instance_id: str,
) -> tuple[Path, Path]:
    ref = Path(spec["model_id"]) / gpu_class / spec["workload_id"] / instance_id
    local_dir = download_profile(
        spec,
        ref,
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
    return ref, profile_path

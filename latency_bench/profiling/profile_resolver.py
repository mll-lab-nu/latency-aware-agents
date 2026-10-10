"""Resolve exact model artifacts and published deployment profiles."""

from __future__ import annotations

import os
import re
from pathlib import Path
from copy import deepcopy
from typing import Any, Iterable

from latency_bench.core.config import load_config
from latency_bench.profiling.sync import download_profile
from latency_bench.utils.io import read_json


PROFILE_DONE_RE = re.compile(
    r"^profiles/(?P<model_id>[^/]+)/(?P<gpu_class>[^/]+)/"
    r"(?P<workload_id>[^/]+)/(?P<instance_id>[^/]+)/DONE$"
)


def resolve_model_artifacts(
    config: dict[str, Any], *, profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve configured artifacts, selecting recorded revisions for profile replay.

    Ordinary policy paths need no artifact declaration. Resolved roots remain
    local, while profile provenance records the selected artifact source.
    """
    config = deepcopy(config)
    options = config["profiling"]
    provenance = config.setdefault("provenance", {})
    recorded = profile["provenance"] if profile is not None else None
    for role in ("model", "runtime"):
        key = f"{role}_artifact"
        if recorded is None:
            if key not in options:
                continue
            artifact = options[key]
        else:
            if key not in recorded:
                continue
            origin = recorded[key]
            selected = options[key] if key in options else {"source": origin["source"]}
            artifact = {"source": selected["source"], "checkpoint": origin["checkpoint"]}
            if "model_config" in origin:
                artifact["model_config"] = origin["model_config"]
            if artifact["source"] == "local":
                artifact["local_root"] = selected["local_root"]
            elif artifact["source"] == "hf":
                if origin["source"] != "hf":
                    raise ValueError("A locally measured artifact requires an explicit local copy for replay")
                artifact.update(hf_repo=origin["repo_id"], hf_path=origin["path_in_repo"],
                                revision=origin["revision"])
            options[key] = artifact
        resolved = _resolve_artifact(artifact)
        portable = resolved["provenance"] if recorded is None else {
            name: value for name, value in recorded[key].items()
            if name in {"source", "checkpoint", "model_config", "repo_id", "path_in_repo", "revision"}
        }
        provenance[key] = {**portable, "local_root": str(resolved["root"])}
        if role == "runtime":
            config["env"]["runtime_checkpoint_path"] = str(resolved["path"])
            continue
        policy = config["policy"]
        policy["checkpoint_path"] = str(resolved["path"])
        if "model_config" in artifact:
            policy["model_config_path"] = str(_contained_path(
                resolved["root"], artifact["model_config"],
                allow_hf_cache_symlink=artifact["source"] == "hf",
            ))
            for field in ("task_contract_path", "task_manifest_path"):
                if field in policy and not Path(policy[field]).is_absolute():
                    policy[field] = str(_contained_path(
                        Path(policy["model_config_path"]).parent, policy[field],
                        allow_hf_cache_symlink=artifact["source"] == "hf",
                    ))
    _prepare_backbone(config, recorded=recorded.get("backbone_artifact") if recorded is not None else None)
    return config


def _prepare_backbone(config: dict[str, Any], *, recorded: dict[str, Any] | None = None) -> None:
    """Bind the actual StarVLA model and processor inputs before workers start."""
    policy = config["policy"]
    provenance = config.setdefault("provenance", {})
    if "backbone_path" in policy:
        policy["backbone_path"] = os.path.abspath(Path(policy["backbone_path"]).expanduser())
        provenance["backbone_artifact"] = {"source": "local"}
        return
    if policy["type"] != "starvla":
        return
    if recorded is not None:
        if recorded["source"] == "local":
            raise ValueError("This profile used a local backbone; configure policy.backbone_path for replay")
        backbone = recorded
    else:
        if "model_artifact" in provenance:
            root = Path(provenance["model_artifact"]["local_root"])
        else:
            return
        model_config = load_config(root / "config.yaml")
        if "backbone" not in model_config:
            return
        backbone = model_config["backbone"]
    from huggingface_hub import snapshot_download

    download_args = {"repo_id": backbone["repo_id"]}
    for field in ("revision", "allow_patterns"):
        if field in backbone:
            download_args[field] = backbone[field]
    snapshot = Path(snapshot_download(**download_args))
    policy["backbone_path"] = str(snapshot)
    provenance["backbone_artifact"] = {
        "source": "hf", "repo_id": backbone["repo_id"], "revision": snapshot.name,
        **({"allow_patterns": backbone["allow_patterns"]} if "allow_patterns" in backbone else {}),
    }


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


def _resolve_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    source = artifact["source"]
    checkpoint = Path(artifact["checkpoint"])
    if source == "local":
        resolved_root = Path(artifact["local_root"]).expanduser().resolve(strict=True)
    elif source == "hf":
        from huggingface_hub import snapshot_download

        hf_repo = artifact["hf_repo"]
        hf_path = Path(artifact["hf_path"])
        download_kwargs = {"repo_id": hf_repo, "repo_type": "model"}
        if "revision" in artifact:
            download_kwargs["revision"] = artifact["revision"]
        if hf_path != Path("."):
            download_kwargs["allow_patterns"] = [f"{hf_path.as_posix()}/**"]
        snapshot = Path(snapshot_download(**download_kwargs))
        resolved_root = snapshot if hf_path == Path(".") else _contained_path(snapshot, hf_path)
    else:
        raise ValueError(f"Unknown artifact source: {source!r}; use local or hf")
    resolved_path = _contained_path(resolved_root, checkpoint, allow_hf_cache_symlink=source == "hf")
    provenance = {"source": source, "checkpoint": checkpoint.as_posix()}
    if "model_config" in artifact:
        provenance["model_config"] = artifact["model_config"]
    if source == "hf":
        provenance.update(repo_id=hf_repo, path_in_repo=hf_path.as_posix(), revision=snapshot.name)
    return {"path": resolved_path, "root": resolved_root, "provenance": provenance}


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
    repo_id: str, ref: str | Path, *, output_dir: str | Path,
) -> Path:
    """Download a profile by its published address, without a model registry."""
    ref = Path(ref)
    if ref.is_absolute() or len(ref.parts) != 4 or ".." in ref.parts:
        raise ValueError("Profile reference must be model/deployment/workload/instance")
    local_dir = download_profile(repo_id, ref, local_dir=Path(output_dir) / ref)
    profile_path = local_dir / "profile.json"
    profile = read_json(profile_path)
    expected = dict(zip(("model_id", "gpu_class", "workload_id", "instance_id"), ref.parts))
    actual = {key: profile[key] for key in expected}
    if actual != expected:
        raise ValueError(f"Remote profile coordinates {actual!r} do not match requested {expected!r}")
    return profile_path

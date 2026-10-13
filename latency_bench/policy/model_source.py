"""Resolve model sources and preserve artifact identity before workers start."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from latency_bench.core.config import load_config


def model_file(policy: dict[str, Any], field: str) -> str:
    """Locate a model file under the already resolved local root."""
    return str(Path(policy["root"]) / policy[field])


def set_model_files(
    config: dict[str, Any],
    *,
    checkpoint: str | Path | None = None,
    model_config: str | Path | None = None,
) -> None:
    """Bind explicit file overrides to the same flat local source declaration."""
    if config["policy"].get("source") == "hf":
        config.update(resolve_model_source(config))
    policy = config["policy"]
    files = {
        key: os.path.abspath(model_file(policy, key))
        for key in ("checkpoint", "model_config")
        if key in policy and policy[key] is not None
    }
    metadata = {
        key: os.path.abspath(Path(policy.get("root", ".")) / policy[key])
        for key in (
            "task_contract_path",
            "task_manifest_path",
            "latency_prompt_map_path",
        )
        if key in policy and not Path(policy[key]).is_absolute()
    }
    for key, value in (("checkpoint", checkpoint), ("model_config", model_config)):
        if value is not None:
            files[key] = os.path.abspath(Path(value).expanduser())
    root = os.path.commonpath([str(Path(value).parent) for value in files.values()])
    policy.update(source="local", root=root)
    policy.update({key: os.path.relpath(value, root) for key, value in files.items()})
    policy.update(metadata)
    if "provenance" in config:
        config["provenance"].pop("model_artifact", None)


def resolve_model_source(
    config: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the one model declaration used by evaluation, collection and profiling."""
    config = deepcopy(config)
    policy = config["policy"]
    provenance = config.setdefault("provenance", {})
    if "source" not in policy:
        return config
    if policy["type"] == "sample_factory":
        policy.setdefault("checkpoint", None)
    source = policy["source"]
    if source == "local":
        root = Path(policy["root"]).expanduser().absolute()
    elif source == "hf":
        from huggingface_hub import snapshot_download

        args = {"repo_id": policy["hf_repo"], "repo_type": "model"}
        if "revision" in policy:
            args["revision"] = policy["revision"]
        directory = Path(policy["root"])
        if directory != Path("."):
            args["allow_patterns"] = [f"{directory.as_posix()}/**"]
        snapshot = Path(snapshot_download(**args))
        root = (
            snapshot if directory == Path(".") else _contained_path(snapshot, directory)
        )
    else:
        raise ValueError(f"Unknown policy.source: {source!r}; use local or hf")
    root = Path(os.path.abspath(root))
    artifact = {"source": source, "checkpoint": policy["checkpoint"]}
    if "model_config" in policy:
        artifact["model_config"] = policy["model_config"]
    if source == "hf":
        artifact.update(
            repo_id=policy["hf_repo"],
            path_in_repo=policy["root"],
            revision=snapshot.name,
        )
    if "model_artifact" in provenance:
        artifact = {
            key: value
            for key, value in provenance["model_artifact"].items()
            if key != "local_root"
        }
    if policy["checkpoint"] is not None:
        _contained_path(root, policy["checkpoint"], allow_hf_cache_symlink=True)
    if "model_config" in policy and policy["model_config"] is not None:
        _contained_path(root, policy["model_config"], allow_hf_cache_symlink=True)
    provenance["model_artifact"] = {**artifact, "local_root": str(root)}
    policy.update(source="local", root=str(root))
    # Relative task metadata belongs to the model bundle. Absolute inputs may
    # explicitly refer to user-owned metadata outside that bundle.
    for key in ("task_contract_path", "task_manifest_path", "latency_prompt_map_path"):
        if key in policy and not Path(policy[key]).is_absolute():
            policy[key] = str(
                _contained_path(root, policy[key], allow_hf_cache_symlink=True)
            )
    _prepare_backbone(config)
    return config


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


def _prepare_backbone(
    config: dict[str, Any]
) -> None:
    """Bind the actual StarVLA model and processor inputs before workers start."""
    policy = config["policy"]
    provenance = config.setdefault("provenance", {})
    if "backbone_path" in policy:
        policy["backbone_path"] = os.path.abspath(
            Path(policy["backbone_path"]).expanduser()
        )
        if "backbone_artifact" not in provenance:
            provenance["backbone_artifact"] = {"source": "local"}
        return
    if policy["type"] != "starvla":
        return
    if "model_config" not in policy or policy["model_config"] is None:
        return
    model_config = load_config(model_file(policy, "model_config"))
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
        "source": "hf",
        "repo_id": backbone["repo_id"],
        "revision": snapshot.name,
        **(
            {"allow_patterns": backbone["allow_patterns"]}
            if "allow_patterns" in backbone
            else {}
        ),
    }

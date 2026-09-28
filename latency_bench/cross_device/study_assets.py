"""Resolve the exact model, training profile and archived eval for one study case."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from latency_bench.core.config import load_config
from latency_bench.utils.io import read_json


REPO_ROOT = Path(__file__).resolve().parents[2]


def prepare_assets(registry, entry):
    """Relocate archived file references without changing policy/task semantics."""
    config = load_config(REPO_ROOT / entry["eval_config"])
    source = entry["training_profile"]
    profile_path = hf_hub_download(
        repo_id=source["repo_id"], repo_type=source["repo_type"],
        revision=source["revision"], filename=source["path"],
    )
    training_profile = read_json(profile_path)
    required_files = {
        entry["checkpoint"], entry["model_config"], "config.yaml", "dataset_statistics.json",
        *entry["policy_files"].values(),
    }
    snapshot = Path(snapshot_download(
        repo_id=registry["model_repo"], revision=registry["model_revision"],
        allow_patterns=[f"{entry['bundle']}/{name}" for name in sorted(required_files)],
    ))
    model_root = snapshot / entry["bundle"]
    backbone = registry["backbone"]
    backbone_root = snapshot_download(repo_id=backbone["repo_id"], revision=backbone["revision"])
    config["policy"]["checkpoint_path"] = str(model_root / entry["checkpoint"])
    config["policy"]["model_config_path"] = str(model_root / entry["model_config"])
    config["policy"]["backbone_path"] = backbone_root
    for key, filename in entry["policy_files"].items():
        config["policy"][key] = str(model_root / filename)
    config["provenance"] = {
        "archived_eval": entry["eval_source"],
        "archived_provenance": config["provenance"] if "provenance" in config else None,
        "model_artifact": {
            "repo_id": registry["model_repo"], "revision": registry["model_revision"],
            "path_in_repo": entry["bundle"], "checkpoint": entry["checkpoint"],
            "model_config": entry["model_config"],
        },
        "training_profile": source,
    }
    return config, training_profile

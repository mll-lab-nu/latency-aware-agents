"""Resolve the exact model, training profile and archived eval for one study case."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from latency_bench.core.config import load_config
from latency_bench.eval.artifacts import read_runtime_snapshot
from latency_bench.utils.io import read_json
from latency_bench.utils.paths import REPO_ROOT


def portable_eval_config(config, entry, registry):
    """Keep task semantics while replacing machine-local execution coordinates."""
    config = read_runtime_snapshot(config)
    config.pop("training", None)
    if "backend" in config:
        config["backend"].pop("train_dir", None)
    config["logging"].pop("output_dir", None)
    config["policy"].pop("worker_python_executable", None)
    config["policy"].pop("backbone_path", None)
    config["policy"].update(source="hf", hf_repo=registry["model_repo"], revision=registry["model_revision"],
                            root=entry["bundle"], checkpoint=entry["checkpoint"],
                            model_config=entry["model_config"])
    for key, value in entry["policy_files"].items():
        config["policy"][key] = value
    if "latency" in config:
        config["latency"].pop("profile_path", None)
    config["provenance"] = {"archived_eval": entry["eval_source"]}
    return config


def prepare_assets(registry, entry, *, eval_config_path=None):
    """Resolve real-stage calibration inputs and the exact model/eval assets."""
    source = entry["training_profile"]
    profile_path = hf_hub_download(
        repo_id=source["repo_id"], repo_type=source["repo_type"],
        revision=source["revision"], filename=source["path"],
    )
    training_profile = read_json(profile_path)
    return prepare_model_assets(registry, entry, eval_config_path=eval_config_path), training_profile


def prepare_model_assets(registry, entry, *, eval_config_path=None):
    """Relocate the pinned model/eval without depending on the training profile."""
    config = read_runtime_snapshot(load_config(REPO_ROOT / entry["eval_config"] if eval_config_path is None else eval_config_path))
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
    config["policy"].update(source="local", root=str(model_root), checkpoint=entry["checkpoint"],
                            model_config=entry["model_config"])
    config["policy"]["backbone_path"] = backbone_root
    for key, filename in entry["policy_files"].items():
        config["policy"][key] = str(model_root / filename)
    if "task_contract_path" in entry["policy_files"]:
        # These archived H1 policies predate action-prefix conditioning.
        config["policy"]["action_prefix"] = {"mode": "none"}
    config["provenance"] = {
        "archived_eval": entry["eval_source"],
        "archived_provenance": config["provenance"] if "provenance" in config else None,
        "model_artifact": {
            "source": "hf", "repo_id": registry["model_repo"], "revision": registry["model_revision"],
            "path_in_repo": entry["bundle"], "checkpoint": entry["checkpoint"],
            "model_config": entry["model_config"],
        },
        "training_profile": entry["training_profile"],
    }
    return config

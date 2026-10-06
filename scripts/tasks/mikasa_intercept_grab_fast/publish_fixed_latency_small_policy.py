#!/usr/bin/env python3
"""Publish one fixed-latency InterceptGrabFast small PPO policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi

from latency_bench.utils.hf import resolve_token


MODEL_REPO = "latency-sensitive-bench/benchmark-models"
TASK_PATH = "latency-aware/mikasa-intercept-grab-fast/small-policy"


def _latency_spec(latency: int) -> dict[str, object]:
    if latency == 0:
        return {"method": "zero"}
    return {"method": "fixed", "fixed_latency_ms": latency * 50.0}


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _copy_tree_payload(*, latency: int, output_root: Path, publish_dir: Path, prefix: str) -> dict[str, object]:
    train_dir = output_root / f"fixed_l{latency}"
    eval_dir = output_root / f"fixed_l{latency}_eval"
    required = {
        "checkpoint": train_dir / "best_policy.pt",
        "training_config": train_dir / "config.yaml",
        "best_evaluation": train_dir / "best_evaluation.json",
        "final_evaluation": eval_dir / "evaluation.json",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot publish; missing required files:\n" + "\n".join(missing))

    publish_dir.mkdir(parents=True, exist_ok=True)
    for child in publish_dir.iterdir():
        if child.name == "publication.json":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    _link_or_copy(required["checkpoint"], publish_dir / "checkpoint.pt")
    shutil.copy2(required["training_config"], publish_dir / "training_config.yaml")
    shutil.copy2(required["best_evaluation"], publish_dir / "best_evaluation.json")
    shutil.copy2(required["final_evaluation"], publish_dir / "evaluation" / "evaluation.json")

    manifest = {
        "task": "mikasa_intercept_grab_fast",
        "environment": "InterceptGrabFast-VLA-v0",
        "policy": "small PPO",
        "latency": latency,
        "latency_config": _latency_spec(latency),
        "control_hz": 20,
        "training_timesteps": 150000000,
        "source": {
            "train_dir": str(train_dir),
            "eval_dir": str(eval_dir),
        },
        "huggingface": {
            "repo_id": MODEL_REPO,
            "repo_type": "model",
            "path": prefix,
        },
    }
    (publish_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (publish_dir / "README.md").write_text(
        "\n".join(
            [
                f"# InterceptGrabFast fixed-latency L{latency} small PPO policy",
                "",
                "This artifact contains the selected native PPO teacher checkpoint for",
                "`InterceptGrabFast-VLA-v0` under the fixed-latency small-policy sweep.",
                "",
                f"- Latency: L{latency}",
                f"- Latency config: `{json.dumps(_latency_spec(latency), sort_keys=True)}`",
                "- Control/observation rate: 20 Hz",
                "- Training budget: 150,000,000 environment timesteps",
                "",
                "Files:",
                "",
                "- `checkpoint.pt`: selected `best_policy.pt`, renamed for publication",
                "- `training_config.yaml`: resolved training config",
                "- `best_evaluation.json`: periodic-training best checkpoint evidence",
                "- `evaluation/evaluation.json`: independent post-training evaluation",
                "- `manifest.json`: provenance and source paths",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def _publish_folder(source: Path, *, repo_id: str, prefix: str) -> str:
    files = {
        f"{prefix}/{path.relative_to(source).as_posix()}": path.stat().st_size
        for path in source.rglob("*")
        if path.is_file() and ".cache" not in path.parts and path.name != "publication.json"
    }
    api = HfApi(token=resolve_token(""))
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=source,
        path_in_repo=prefix,
        ignore_patterns=["**/.cache/**", "publication.json"],
        commit_message=f"Publish {prefix}",
    )
    confirmed: dict[str, int | None] = {}
    paths = list(files)
    for start in range(0, len(paths), 100):
        confirmed.update(
            {
                entry.path: entry.size
                for entry in api.get_paths_info(
                    repo_id,
                    paths[start : start + 100],
                    repo_type="model",
                    revision=commit.oid,
                )
            }
        )
    if confirmed != files:
        raise RuntimeError(f"Hugging Face publication verification failed for {prefix}@{commit.oid}")
    print(f"HF_PUBLISHED={repo_id}/{prefix}@{commit.oid}", flush=True)
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency", type=int, choices=range(5), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/mikasa_fixed_latency_ppo"))
    parser.add_argument("--repo-id", default=MODEL_REPO)
    parser.add_argument("--path-in-repo")
    args = parser.parse_args()

    output_root = args.output_root
    prefix = args.path_in_repo or f"{TASK_PATH}/ppo-fixed-latency-l{args.latency}-v1"
    publish_dir = output_root / "publish" / f"fixed_l{args.latency}"
    manifest = _copy_tree_payload(latency=args.latency, output_root=output_root, publish_dir=publish_dir, prefix=prefix)
    revision = _publish_folder(publish_dir, repo_id=args.repo_id, prefix=prefix)
    receipt = {
        "repo_id": args.repo_id,
        "repo_type": "model",
        "path": prefix,
        "revision": revision,
        "latency": args.latency,
        "manifest": manifest,
    }
    (publish_dir / "publication.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for receipt_dir in (output_root / f"fixed_l{args.latency}", output_root / f"fixed_l{args.latency}_eval"):
        if receipt_dir.is_dir():
            (receipt_dir / "hf_publish_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

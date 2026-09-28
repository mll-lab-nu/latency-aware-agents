#!/usr/bin/env python3
"""Verify formal MIKASA profile-v2 datasets, runs, and optional Hub publications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from latency_bench.utils.hf import resolve_token
from scripts.starvla import artifacts


MODELS = ("qwenoft", "qwenpi_v3", "qwengr00t")
PREFIX_FRAMES = {"qwenoft": 3, "qwenpi_v3": 4, "qwengr00t": 3}


def verify(data_root: Path, output_root: Path, *, hub: bool) -> dict:
    report = {"datasets": {}, "models": {}}
    for model in MODELS:
        dataset_root = data_root / "profile-latency" / model
        audit = json.loads((dataset_root / "dataset_audit.json").read_text())
        assert audit["passed"] is True
        assert audit["collection_num_envs"] == 128
        assert audit["latency_stream_scope"] == "collector_slot"
        assert audit["action_prefix"]["mode"] == "profile_p95"
        assert audit["action_prefix"]["frames"] == PREFIX_FRAMES[model]
        assert audit["splits"]["train"]["valid_actions"] >= 64_288
        assert audit["splits"]["validation"]["valid_actions"] >= 6_884

        run_root = output_root / "profile-latency" / model
        run = json.loads((run_root / "run.json").read_text())
        training = run["training"]
        assert run["run_id"] == "h8-conditioned-seed0-profile"
        assert training["sft_updates"] == 5_000
        assert training["num_gpus"] == 2
        assert training["micro_batch"] == 64
        assert training["gradient_accumulation_steps"] == 1
        assert training["global_batch"] == 128
        assert training["gradient_checkpointing"] is True
        assert training["oom_retry"] is False
        assert (
            run_root
            / "training/bootstrap/checkpoints/steps_2500_pytorch_model.pt"
        ).is_file()
        assert (
            run_root
            / "training/bootstrap/checkpoints/steps_5000_pytorch_model.pt"
        ).is_file()
        assert (run_root / "inference/checkpoints/model.pt").is_file()
        evaluation = json.loads((run_root / "inference/eval_profile.json").read_text())
        acceptance = evaluation["acceptance"]
        assert acceptance["episode_count"] == 200
        assert run["evaluation_matrix"]["profile"]["latency_seed"] == 0
        assert set(run["evaluation_matrix"]) == {"profile"}
        report["datasets"][model] = {
            "path": run["datasets"][0]["path"],
            "revision": run["datasets"][0]["revision"],
            "audit": audit["splits"],
        }
        report["models"][model] = {
            "success_count": acceptance["success_count"],
            "episode_count": acceptance["episode_count"],
            "passed": acceptance["passed"],
        }

        if hub:
            dataset_ref = run["datasets"][0]
            for filename in ("provenance.json", "lerobot/task_contract.json"):
                hf_hub_download(
                    dataset_ref["repo_id"],
                    filename=f"{dataset_ref['path']}/{filename}",
                    repo_type="dataset",
                    revision=dataset_ref["revision"],
                    token=resolve_token(""),
                )
            publication = json.loads((run_root / "publication.json").read_text())
            for filename in (
                "checkpoints/model.pt",
                "task_contract.json",
                "eval_profile.json",
                "provenance.json",
            ):
                hf_hub_download(
                    publication["repo_id"],
                    filename=f"{publication['path']}/{filename}",
                    revision=publication["revision"],
                    token=resolve_token(""),
                )
            report["models"][model]["path"] = publication["path"]
            report["models"][model]["revision"] = publication["revision"]

    if hub:
        (output_root / "publication-verification.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hub", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.data_root, args.output_root, hub=args.hub), indent=2))


if __name__ == "__main__":
    main()

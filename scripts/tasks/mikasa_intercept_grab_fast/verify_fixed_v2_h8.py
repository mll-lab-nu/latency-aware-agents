#!/usr/bin/env python3
"""Verify formal MIKASA fixed-v2 data, training, and matched evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.tasks.mikasa_intercept_grab_fast.fixed_latency import FIXED_LATENCIES


MODELS = ("qwenoft", "qwenpi_v3", "qwengr00t")


def verify(data_root: Path, output_root: Path, *, condition: str, models: list[str]) -> dict:
    frames = FIXED_LATENCIES[condition]
    dataset_root = data_root / condition
    audit = json.loads((dataset_root / "dataset_audit.json").read_text())
    assert audit["passed"] is True
    assert audit["collection_num_envs"] == 128
    assert audit["latency_stream_scope"] == "collector_slot"
    assert audit["action_prefix"]["mode"] == "fixed_frames"
    assert audit["action_prefix"]["frames"] == frames
    assert audit["splits"]["train"]["valid_actions"] >= 64_288
    assert audit["splits"]["validation"]["valid_actions"] >= 6_884

    report = {"dataset": audit["splits"], "models": {}}
    for model in models:
        run_root = output_root / condition / model
        run = json.loads((run_root / "run.json").read_text())
        training = run["training"]
        assert training["sft_updates"] == 5_000
        assert training["num_gpus"] == 2
        assert training["micro_batch"] == 64
        assert training["gradient_accumulation_steps"] == 1
        assert training["global_batch"] == 128
        assert training["gradient_checkpointing"] is (model != "qwenoft")
        assert training["oom_retry"] is False
        assert (
            run_root
            / "training/bootstrap/checkpoints/steps_5000_state/latest"
        ).is_file()
        assert (
            run_root
            / "training/bootstrap/checkpoints/steps_5000_pytorch_model.pt"
        ).is_file()
        assert (run_root / "inference/checkpoints/model.pt").is_file()

        evaluation = json.loads(
            (run_root / f"inference/eval_fixed_{frames}f.json").read_text()
        )
        acceptance = evaluation["acceptance"]
        execution = evaluation["chunk_execution_audit"]
        assert acceptance["episode_count"] == 200
        assert execution["fixed_latency_ms"] == frames * 50.0
        assert execution["latency_raw_frames"] == frames
        assert execution["passed"] is True
        assert set(run["evaluation_matrix"]) == {condition}
        report["models"][model] = {
            "success_count": acceptance["success_count"],
            "episode_count": acceptance["episode_count"],
            "passed": acceptance["passed"],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", choices=FIXED_LATENCIES, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.data_root, args.output_root, condition=args.condition, models=args.models), indent=2))


if __name__ == "__main__":
    main()

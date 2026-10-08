#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset import (
    _load_train_split,
    build_latency_prompt_map,
)


def _parse_contract(value: dict[str, Any] | str | None) -> dict[str, Any]:
    if value in (None, ""):
        raise ValueError("gymnasium_task_contract is required for Gymnasium dataset verification")
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


def _action_labels(contract: dict[str, Any]) -> list[str]:
    if "action_labels" in contract:
        return [str(value) for value in contract["action_labels"]]
    action_space = contract.get("action_space") or {}
    if "labels" in action_space:
        return [str(value) for value in action_space["labels"]]
    raise ValueError("gymnasium_task_contract must contain action_labels or action_space.labels")


def _state_labels(contract: dict[str, Any]) -> list[str]:
    state_space = contract.get("state_space") or {}
    labels = state_space.get("labels") or []
    return [str(value) for value in labels]


def verify_dataset(
    dataset_name: str,
    *,
    rows: int = 200,
    cache_dir: str | None = None,
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
    strict: bool = False,
    allow_mixed_latency_prompts: bool = False,
    latencies: list[int] | None = None,
    source_latency_column: str,
    target_latency_unit: str,
    obs_stride_raw_frames: int,
    gymnasium_task_contract: dict[str, Any] | str | None = None,
) -> bool:
    contract = _parse_contract(gymnasium_task_contract)
    expected_actions = _action_labels(contract)
    expected_state_labels = _state_labels(contract)

    try:
        for columns in (
            ["prompt", "action_id", "action_text", "state", source_latency_column, "latency_ms"],
            ["prompt", "action_id", "state", source_latency_column, "latency_ms"],
            ["prompt", "action_id", "state"],
            None,
        ):
            try:
                ds = _load_train_split(
                    dataset_name,
                    cache_dir,
                    columns=columns,
                    dataset_config_name=dataset_config_name,
                    dataset_source_subdir=dataset_source_subdir,
                    latencies=latencies,
                )
                break
            except Exception:
                if columns is None:
                    raise
    except Exception as exc:
        print(f"ERROR: could not load Gymnasium dataset {dataset_name}: {exc}")
        if strict:
            raise
        return False

    if len(ds) == 0:
        print("ERROR: Gymnasium dataset has zero train rows.")
        if strict:
            raise ValueError("Gymnasium dataset has zero train rows")
        return False

    sample_n = min(rows, len(ds))
    ok = True

    prompts = {str(ds[i]["prompt"]) for i in range(sample_n)}
    if any(not prompt.strip() for prompt in prompts):
        print("ERROR: prompt must be a non-empty string.")
        ok = False

    if allow_mixed_latency_prompts:
        try:
            mapping = build_latency_prompt_map(
                ds,
                latency_column=source_latency_column,
                target_latency_unit=target_latency_unit,
                obs_stride_raw_frames=obs_stride_raw_frames,
            )
            if latencies and len({int(v) for v in latencies}) > 1 and len(mapping) <= 1:
                raise ValueError(f"expected more than one latency prompt, got {len(mapping)}")
            print("Latency prompt map:")
            print(json.dumps(mapping, indent=2))
        except Exception as exc:
            print(f"ERROR: invalid mixed-latency prompt mapping: {exc}")
            ok = False

    action_id_to_text: dict[int, set[str]] = defaultdict(set)
    seen_ids = set()
    has_action_text = "action_text" in ds.column_names
    has_state = "state" in ds.column_names
    for i in range(sample_n):
        action_id = int(ds[i]["action_id"])
        seen_ids.add(action_id)
        if has_action_text:
            action_id_to_text[action_id].add(str(ds[i]["action_text"]))
        if expected_state_labels and has_state:
            state = ds[i]["state"]
            if state is None or len(state) != len(expected_state_labels):
                print(
                    "ERROR: state vector length mismatch: "
                    f"saw {None if state is None else len(state)}, expected {len(expected_state_labels)}"
                )
                ok = False
                break

    if not seen_ids or min(seen_ids) < 0 or max(seen_ids) >= len(expected_actions):
        print(f"ERROR: action_id values must be in [0, {len(expected_actions) - 1}], saw {sorted(seen_ids)}")
        ok = False

    if has_action_text:
        for action_id, texts in sorted(action_id_to_text.items()):
            expected = expected_actions[action_id] if action_id < len(expected_actions) else None
            if texts != {expected}:
                print(f"ERROR: action_id={action_id} maps to {sorted(texts)}, expected {expected!r}")
                ok = False

    if ok:
        print(f"Gymnasium dataset verification passed for {dataset_name} ({sample_n} sampled rows).")
    elif strict:
        raise ValueError("Gymnasium dataset verification failed")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", "--dataset_name", required=True)
    parser.add_argument("--dataset-config-name", "--dataset_config_name", default=None)
    parser.add_argument("--dataset-source-subdir", "--dataset_source_subdir", default=None)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--cache-dir", "--cache_dir", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow-mixed-latency-prompts", "--allow_mixed_latency_prompts", action="store_true")
    parser.add_argument("--latencies", default=None)
    parser.add_argument("--source-latency-column", choices=["latency", "latency_raw_frames"], required=True)
    parser.add_argument("--target-latency-unit", choices=["raw_frames", "observation_steps"], required=True)
    parser.add_argument("--obs-stride-raw-frames", type=int, required=True)
    parser.add_argument("--gymnasium-task-contract", "--gymnasium_task_contract", required=True)
    args = parser.parse_args()

    latencies = None
    if args.latencies not in (None, ""):
        latencies = [int(item) for item in str(args.latencies).split(",") if item.strip()]

    try:
        ok = verify_dataset(
            args.dataset_name,
            rows=args.rows,
            cache_dir=args.cache_dir,
            dataset_config_name=args.dataset_config_name,
            dataset_source_subdir=args.dataset_source_subdir,
            strict=args.strict,
            allow_mixed_latency_prompts=args.allow_mixed_latency_prompts,
            latencies=latencies,
            source_latency_column=args.source_latency_column,
            target_latency_unit=args.target_latency_unit,
            obs_stride_raw_frames=args.obs_stride_raw_frames,
            gymnasium_task_contract=args.gymnasium_task_contract,
        )
    except Exception:
        if args.strict:
            raise
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

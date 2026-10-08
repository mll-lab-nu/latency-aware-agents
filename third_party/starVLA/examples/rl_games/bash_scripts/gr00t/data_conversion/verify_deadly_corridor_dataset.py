#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from examples.rl_games.bash_scripts.gr00t.data_conversion.verify_flappy_dataset import (
    _load_train_split,
    build_latency_prompt_map,
)


EXPECTED_ACTIONS = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
]
FACTORIZED_11_ACTIONS = [
    "TURN_NONE",
    "TURN_LEFT",
    "TURN_RIGHT",
    "MOVE_NONE",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "STRAFE_NONE",
    "STRAFE_LEFT",
    "STRAFE_RIGHT",
    "ATTACK_OFF",
    "ATTACK_ON",
]
ACTION_LAYOUT_MULTIBINARY_7 = "multibinary_7"
ACTION_LAYOUT_FACTORIZED_11 = "factorized_11"
REQUIRED_PROMPT_PARTS = ["Deadly Corridor", *EXPECTED_ACTIONS]


def _load_first_available(
    dataset_name: str,
    cache_dir: str | None,
    column_options: tuple[list[str] | None, ...],
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
    latencies: list[int] | None = None,
):
    last_exc: Exception | None = None
    for columns in column_options:
        try:
            return _load_train_split(
                dataset_name,
                cache_dir,
                columns=columns,
                dataset_config_name=dataset_config_name,
                dataset_source_subdir=dataset_source_subdir,
                latencies=latencies,
            )
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("no column options provided")


def _normalize_action_layout(action_layout: str) -> str:
    layout = str(action_layout or ACTION_LAYOUT_MULTIBINARY_7).lower()
    if layout in {ACTION_LAYOUT_MULTIBINARY_7, ACTION_LAYOUT_FACTORIZED_11}:
        return layout
    raise ValueError(f"Unsupported action_layout={action_layout!r}; expected multibinary_7 or factorized_11")


def _factorized_one_hot(action_tuple: Any) -> list[float]:
    turn, move, strafe, attack = [int(value) for value in action_tuple]
    values = [0.0] * len(FACTORIZED_11_ACTIONS)
    values[turn] = 1.0
    values[3 + move] = 1.0
    values[6 + strafe] = 1.0
    values[9 + attack] = 1.0
    return values


def _action_vector(row: dict[str, Any], action_layout: str = ACTION_LAYOUT_MULTIBINARY_7) -> list[float]:
    if _normalize_action_layout(action_layout) == ACTION_LAYOUT_FACTORIZED_11:
        if "action_tuple" in row and row["action_tuple"] is not None:
            return _factorized_one_hot(row["action_tuple"])
        raw_action = row.get("action", row.get("actions"))
        if raw_action is not None:
            values = np.asarray(raw_action, dtype=np.float32).reshape(-1).tolist()
            if len(values) != len(FACTORIZED_11_ACTIONS):
                raise ValueError(
                    f"Deadly Corridor factorized action must have {len(FACTORIZED_11_ACTIONS)} values, got {len(values)}"
                )
            if any(value not in (0.0, 1.0) for value in values):
                raise ValueError(f"Deadly Corridor factorized action values must be binary, got {values}")
            return values
        raise ValueError("Deadly Corridor factorized rows must include either `action_tuple` or 11D `action`")

    raw_action = row.get("action", row.get("actions"))
    if raw_action is not None:
        values = np.asarray(raw_action, dtype=np.float32).reshape(-1).tolist()
        if len(values) != len(EXPECTED_ACTIONS):
            raise ValueError(f"Deadly Corridor action must have {len(EXPECTED_ACTIONS)} values, got {len(values)}")
        if any(value not in (0.0, 1.0) for value in values):
            raise ValueError(f"Deadly Corridor action values must be binary, got {values}")
        return values

    if "action_text" in row and row.get("action_text") is not None:
        text = str(row["action_text"]).upper()
        return [1.0 if label in text else 0.0 for label in EXPECTED_ACTIONS]

    raise ValueError("Deadly Corridor rows must include either multibinary `action` or `action_text`")


def verify_dataset(
    dataset_name: str,
    *,
    rows: int = 200,
    cache_dir: str | None = None,
    dataset_config_name: str | None = None,
    dataset_source_subdir: str | None = None,
    strict: bool = False,
    allow_mixed_latency_prompts: bool = False,
    action_layout: str = ACTION_LAYOUT_MULTIBINARY_7,
    latencies: list[int] | None = None,
    source_latency_column: str,
    target_latency_unit: str,
    obs_stride_raw_frames: int,
) -> bool:
    action_layout = _normalize_action_layout(action_layout)
    if action_layout == ACTION_LAYOUT_FACTORIZED_11:
        column_options = (
            ["prompt", "action_tuple", "latency", "latency_ms"],
            ["prompt", "action_tuple", "latency_raw_frames", "latency_ms"],
            ["prompt", "action", "latency", "latency_ms"],
            ["prompt", "action", "latency_raw_frames", "latency_ms"],
            ["prompt", "actions", "latency", "latency_ms"],
            ["prompt", "actions", "latency_raw_frames", "latency_ms"],
            ["prompt", "action_tuple"],
            ["prompt", "action"],
            ["prompt", "actions"],
            None,
        )
    else:
        column_options = (
            ["prompt", "action_text", "latency", "latency_ms"],
            ["prompt", "action_text", "latency_raw_frames", "latency_ms"],
            ["prompt", "action", "latency", "latency_ms"],
            ["prompt", "action", "latency_raw_frames", "latency_ms"],
            ["prompt", "actions", "latency", "latency_ms"],
            ["prompt", "actions", "latency_raw_frames", "latency_ms"],
            ["prompt", "action_text"],
            ["prompt", "action"],
            ["prompt", "actions"],
            None,
        )
    try:
        ds = _load_first_available(
            dataset_name,
            cache_dir,
            column_options,
            dataset_config_name=dataset_config_name,
            dataset_source_subdir=dataset_source_subdir,
            latencies=latencies,
        )
    except Exception as exc:
        print(f"ERROR: could not load dataset {dataset_name}: {exc}")
        if strict:
            raise
        return False

    if len(ds) == 0:
        print("ERROR: dataset has zero train rows.")
        if strict:
            raise ValueError("dataset has zero train rows")
        return False

    sample_n = min(rows, len(ds))
    ok = True

    prompts = {str(ds[i]["prompt"]) for i in range(sample_n)}
    if any(not prompt.strip() for prompt in prompts):
        print("ERROR: prompt must be a non-empty string.")
        ok = False
    if allow_mixed_latency_prompts:
        try:
            prompt_ds = _load_first_available(
                dataset_name,
                cache_dir,
                (
                    ["split", "prompt", source_latency_column, "latency_ms"],
                    ["prompt", source_latency_column, "latency_ms"],
                    None,
                ),
                dataset_config_name=dataset_config_name,
                dataset_source_subdir=dataset_source_subdir,
                latencies=latencies,
            )
            mapping = build_latency_prompt_map(
                prompt_ds,
                latency_column=source_latency_column,
                target_latency_unit=target_latency_unit,
                obs_stride_raw_frames=obs_stride_raw_frames,
            )
            if len(mapping) <= 1:
                raise ValueError(f"expected more than one latency prompt, got {len(mapping)}")
            print("Latency prompt map:")
            print(json.dumps(mapping, indent=2))
        except Exception as exc:
            print(f"ERROR: invalid mixed-latency prompt mapping: {exc}")
            ok = False

    seen_active = set()
    for i in range(sample_n):
        try:
            action = _action_vector(ds[i], action_layout)
        except Exception as exc:
            print(f"ERROR: invalid action at row {i}: {exc}")
            ok = False
            break
        seen_active.update(idx for idx, value in enumerate(action) if value)

    if not seen_active:
        print("WARNING: sampled rows contain only no-op Deadly Corridor actions.")

    if ok:
        print(f"Deadly Corridor dataset verification passed for {dataset_name} ({sample_n} sampled rows).")
    elif strict:
        raise ValueError("Deadly Corridor dataset verification failed")
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
    parser.add_argument("--source-latency-column", choices=["latency", "latency_raw_frames"], required=True)
    parser.add_argument("--target-latency-unit", choices=["raw_frames", "observation_steps"], required=True)
    parser.add_argument("--obs-stride-raw-frames", type=int, required=True)
    parser.add_argument(
        "--action-layout",
        "--action_layout",
        choices=[ACTION_LAYOUT_MULTIBINARY_7, ACTION_LAYOUT_FACTORIZED_11],
        default=ACTION_LAYOUT_MULTIBINARY_7,
    )
    args = parser.parse_args()

    try:
        ok = verify_dataset(
            args.dataset_name,
            rows=args.rows,
            cache_dir=args.cache_dir,
            dataset_config_name=args.dataset_config_name,
            dataset_source_subdir=args.dataset_source_subdir,
            strict=args.strict,
            allow_mixed_latency_prompts=args.allow_mixed_latency_prompts,
            action_layout=args.action_layout,
            source_latency_column=args.source_latency_column,
            target_latency_unit=args.target_latency_unit,
            obs_stride_raw_frames=args.obs_stride_raw_frames,
        )
    except Exception:
        if args.strict:
            raise
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
